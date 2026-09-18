# RL Chess Agent

An AlphaZero-inspired reinforcement learning chess engine built from scratch. Uses a **Transformer neural network** as the evaluation backbone, **Monte Carlo Tree Search (MCTS)** for move selection, and a deterministic **Alpha-Beta endgame solver** that takes over in simplified positions to force checkmate.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Board State                      │
│                  (42 × 8 × 8 tensor)                │
│  turn + pieces + attacks + castling + en passant    │
│                  + 8-move history                   │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│              1×1 Conv Projection (42 → 512)         │
│           + Learnable Positional Encoding           │
│                  (64 square tokens)                 │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│            Transformer Encoder (×10 layers)         │
│         d_model=512, 8 heads, FFN=2048              │
│              Pre-Norm, batch_first                  │
└──────────┬────────────────────────────┬─────────────┘
           │                            │
           ▼                            ▼
┌──────────────────────┐  ┌──────────────────────────┐
│     Policy Head      │  │       Value Head          │
│  Linear(32768→4096)  │  │   Linear(32768→512)      │
│  Linear(4096→4672)   │  │   Linear(512→1)          │
│   → raw logits       │  │   → tanh ∈ [-1, 1]       │
└──────────────────────┘  └──────────────────────────┘
```

- **Policy Head**: Outputs logits over 4672 possible actions (4096 normal moves + queen promotions, 144 underpromotion slots for Knight/Bishop/Rook)
- **Value Head**: Outputs a single scalar predicting the game outcome from the current player's perspective
- **~30M parameters** — designed for an RTX 3080

### MCTS (Opening + Middlegame)

Standard AlphaZero-style MCTS with PUCT selection, Dirichlet noise at root for exploration, and 400 simulations per move. The neural network provides both the prior probabilities for edge selection and the leaf evaluation.

### Alpha-Beta Solver (Endgame)

When ≤ 7 pieces remain on the board, the engine automatically switches from MCTS to a deterministic **alpha-beta search with iterative deepening** (up to depth 20). This solver uses MVV-LVA move ordering, king-cornering heuristics, and passed-pawn proximity bonuses to force checkmate reliably.

## Project Structure

```
RLChessAgent/
├── Chess/
│   ├── board.py              # Board state representation (42×8×8 tensor encoding + 8-move history)
│   ├── action_encoding.py    # Shared move ↔ index mapping (handles underpromotions)
│   └── endgame_solver.py     # Alpha-beta pruning solver for simplified positions
├── Models/
│   ├── net.py                # RLModel — shared Transformer backbone
│   ├── policy_net.py         # PolicyHead — linear layers outputting raw logits
│   └── value_net.py          # ValueHead — linear layers outputting scalar value
├── MCTS/
│   ├── mcts.py               # Monte Carlo Tree Search with PUCT
│   ├── node.py               # Tree node (wraps Board state)
│   ├── edge.py               # Tree edge (action, visit count, value stats)
│   └── constant.py           # Hyperparameters (C_base, C_init, Dirichlet α)
├── KingBase/                 # KingBase2019 PGN dataset (grandmaster games, 2000+ ELO)
├── main.py                   # Play against the agent (CLI)
├── train_supervised.py       # Supervised pre-training on KingBase PGN data
├── train.py                  # RL self-play training loop
├── play_self.py              # Watch the agent play itself (with PGN log)
├── arena.py                  # Pit two checkpoints against each other
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.8+ and PyTorch with CUDA support for GPU training.

## Training

### Step 1: Supervised Pre-Training (recommended first)

Download the [KingBase2019](https://archive.org/details/KingBase2019) dataset and place the `.pgn` files in the `KingBase/` directory, then run:

```bash
python3 train_supervised.py
```

This parses millions of grandmaster moves and trains the network using cross-entropy loss on the policy and MSE loss on the value. Checkpoints are saved every 5000 steps. The training uses a 100k-position replay buffer with random sampling to prevent catastrophic forgetting across opening systems.

### Step 2: RL Self-Play Fine-Tuning (optional)

Once supervised training converges, load the checkpoint into the RL self-play loop:

```bash
python3 train.py
```

The self-play loop includes reward shaping bonuses:
- **+0.1** for capturing an undefended (hanging) piece
- **+0.05** for moving to a square defended by a friendly piece
- **+0.2** for promoting a pawn (+0.1 extra if the promotion square is undefended)

### Monitoring

```bash
tensorboard --logdir runs/
```

Tracks `Supervised/ValueLoss`, `Supervised/PolicyLoss`, `SelfPlay/GameLength`, and more.

## Usage

### Play Against the Agent

```bash
python3 main.py --checkpoint supervised_model_step_10000.pth
```

Enter moves in SAN (`Nf3`, `O-O`) or UCI (`e2e4`) notation. The agent uses MCTS with 400 simulations in the opening/middlegame and switches to the alpha-beta solver in simplified endgames.

### Watch Agent vs. Itself

```bash
python3 play_self.py --checkpoint supervised_model_step_10000.pth
```

Outputs a move-by-move log to `self_play_log.txt` with standard PGN at the bottom (paste into [Lichess](https://lichess.org/paste) to replay).

### Pit Two Checkpoints Against Each Other

```bash
python3 arena.py --white supervised_model_step_5000.pth --black supervised_model_step_10000.pth --sims 400
```

Useful for evaluating whether a later checkpoint is objectively stronger. Log saved to `arena_log.txt`.

---

## Development Log

This section documents the key observations made during development and the corresponding fixes applied. It serves as a record of the iterative debugging process that shaped the final agent.

### Phase 1: Initial Implementation

The project started as an incomplete skeleton with stubbed-out MCTS functions and basic Conv2d neural networks. The initial work completed:
- Implemented missing MCTS functions (`expand`, `backpropagation`) with AlphaZero PUCT formula
- Fixed neural network input dimensions to accept the `(42, 8, 8)` board tensor
- Fixed board encoding to output `float32` (was using integers, causing overflow with negative piece values)
- Created CLI gameplay (`main.py`), RL training loop (`train.py`), and self-play logger (`play_self.py`)
- Added TensorBoard integration and GPU auto-detection (CUDA / MPS / CPU)

### Phase 2: Architecture Upgrade

**Observation**: *"The agent seems to be pretty stupid"* — the simple Conv2d architecture couldn't capture long-range dependencies across the board (e.g., a Bishop on a1 threatening h8).

**Changes**:
- Replaced Conv2d blocks with a **10-layer Transformer Encoder** (d_model=512, 8 heads) treating the 64 squares as sequence tokens — similar to Vision Transformers
- Unified the Policy and Value networks to share the Transformer backbone (matching true AlphaZero methodology)
- Scaled up to ~30M parameters for the RTX 3080
- Added reward shaping for capturing undefended pieces (+0.1) and moving to defended squares (+0.05)

### Phase 3: Supervised Pre-Training

**Observation**: *"Can I use some dataset to train?"* — pure self-play from random weights requires enormous compute.

**Changes**:
- Created `train_supervised.py` to pre-train on the KingBase2019 grandmaster dataset (1.5GB+ of PGN files)
- Added error handling for corrupted PGN entries, garbage collection for memory management, and `utf-8` encoding fallbacks
- Added epoch-level logging to track dataset coverage

### Phase 4: MCTS Hardening

**Observation**: The agent was getting stuck in infinite move-repetition loops and showing extreme EV variance on repeated positions (0.93 vs -0.72 for the same position).

**Root Causes Identified**:
1. **No repetition detection inside MCTS** — the tree treated looping back to a previous position as free value instead of scoring it as a draw
2. **History amnesia** — MCTS was initialized from a FEN string on every turn, wiping the 8-move history queue. The neural network saw empty history planes every single move
3. **Low simulation budget** — 30-50 simulations weren't enough to distinguish a real plan from a repeating threat

**Changes**:
- Refactored `Node` to hold a deep-copied `Board` object (with full move stack) instead of a FEN string — `python-chess` natively detects threefold repetition and 50-move draws via `is_game_over(claim_draw=True)`
- Added `Board.copy()` that preserves both the board state and the 8-move history queue
- Raised default simulation budget to 400 across all scripts
- Added Dirichlet noise and temperature-based exploration at root during self-play

### Phase 5: Critical Bug Audit

**Observation**: *"The agents are not performing well"* — even after architectural upgrades and dataset training, play quality remained poor.

A systematic audit of every file revealed **8 issues**, 4 of which were outright crashers:

| # | Severity | Bug | Impact |
|---|----------|-----|--------|
| 1 | 🔴 Crash | `torch.softmax()` missing required `dim` argument | Policy head crashed at runtime |
| 2 | 🔴 Crash | `board` variable referenced after rename to `b` | RL self-play crashed after every game |
| 3 | 🔴 Crash | `chess.Board(mcts.root.state)` — state was a Board object, not a string | Policy extraction crashed |
| 4 | 🔴 Silent | `[np.zeros(...)]*8` created 8 references to the **same** array | Network saw no history — always zeros |
| 5 | 🟠 Design | MSE loss on 4672-dim one-hot policy target | Gradient drowned in 4671 zeros → random policy |
| 6 | 🟠 Design | `from_sq*64 + to_sq` doesn't distinguish promotion types | Cannot learn to promote to Knight/Rook/Bishop |
| 7 | 🟡 Logic | Terminal value signs inverted in MCTS expand | Agent learned to **lose** instead of win |
| 8 | 🟡 Design | Sequential data feeding in supervised training | Catastrophic forgetting across opening systems |

**Fixes Applied**:
- **#1**: Policy head now outputs raw logits; softmax applied externally where needed
- **#2**: Fixed `board` → `b.board.move_stack`
- **#3**: Fixed `chess.Board(state)` → `state.board.legal_moves`
- **#4**: Changed `[np.zeros(...)]*8` to `[np.zeros(...) for _ in range(8)]`
- **#5**: Switched from `F.mse_loss` to `F.cross_entropy` (RL uses soft cross-entropy with MCTS visit distribution as target)
- **#6**: Created `Chess/action_encoding.py` with dedicated slots for underpromotions (indices 4096–4239)
- **#7**: Corrected terminal value sign convention in `mcts.py`
- **#8**: Replaced sequential feeding with a 100k-position replay buffer using `random.sample()`

### Phase 6: Endgame Improvements

**Observation**: *"The endgame is quite noisy"* and *"the pawn isn't promoting even when the square is undefended"*.

**Changes**:
- Created `Chess/endgame_solver.py` — a deterministic alpha-beta search with iterative deepening that activates when ≤7 pieces remain
- The solver uses MVV-LVA move ordering, king-cornering heuristics, and passed-pawn proximity bonuses
- Added promotion reward shaping: +0.2 for any promotion, +0.1 extra for undefended promotion squares
- Fixed promotion action encoding so all promotion types (Q/R/B/N) map to distinct network outputs

---

## References

- [Silver et al., 2017 — Mastering Chess and Shogi by Self-Play (AlphaZero)](https://arxiv.org/abs/1712.01815)
- [KingBase2019 Dataset](https://archive.org/details/KingBase2019)
- [python-chess](https://python-chess.readthedocs.io/)

## License

MIT