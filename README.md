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

This is a record of how this project evolved — the things I noticed while watching the agent play, the bugs I tracked down, and the fixes that actually moved the needle.

### Phase 1: Initial Implementation

I started this project a while back wanting to build an AlphaZero-style chess engine from scratch. I had the skeleton in place — board representation, neural network stubs, a half-written MCTS — but nothing actually worked end to end. The first task was just getting it to run:
- I filled in the missing MCTS functions (`expand`, `backpropagation`) and wired up the AlphaZero PUCT formula
- Fixed the neural network input dimensions — they were expecting flat vectors but the board encoding produces a `(42, 8, 8)` spatial tensor
- The board encoding was using integer dtypes, which silently overflowed when encoding negative piece values. Switched to `float32`
- Built out the CLI (`main.py`) so I could actually sit down and play against the thing, plus a training loop (`train.py`) and a self-play logger (`play_self.py`)
- Added TensorBoard so I could watch training metrics, and auto-detection for CUDA / MPS / CPU

At this point the agent played, but its moves were essentially random.

### Phase 2: Architecture Upgrade

After playing a few games against it, my observation was blunt: **the agent was pretty stupid.** It had no sense of long-range threats — it couldn't see that a Bishop on a1 was threatening something on h8, for example. The simple Conv2d architecture just didn't have the receptive field for it.

So I replaced the entire neural network backbone:
- Swapped the Conv2d blocks for a **10-layer Transformer Encoder** (d_model=512, 8 attention heads), treating the 64 board squares as sequence tokens — basically a Vision Transformer for chess
- Unified the Policy and Value networks to share this backbone, branching only at the final linear heads. This is what AlphaZero actually does and it's much more parameter-efficient
- Scaled the model up to ~30M parameters since I have an RTX 3080 on a desktop-cooled system and it can handle it
- Added reward shaping to give the agent some chess intuition faster: +0.1 for capturing an undefended piece, +0.05 for moving to a square covered by a friendly piece

### Phase 3: Supervised Pre-Training

I realized that training purely from self-play with random weights would take an absurd amount of compute — this is how the original AlphaZero did it, but they had thousands of TPUs. I needed a shortcut.

I downloaded the KingBase2019 dataset (millions of grandmaster games, 1.5GB+ of PGN files) and wrote `train_supervised.py` to pre-train the network on real human moves. The idea is to get the network to a baseline where it understands basic chess principles, and *then* switch to self-play to push it further.

The dataset loading crashed a few times initially — some PGN files had corrupted entries or non-UTF8 characters in player names. I added try/except guards, forced `utf-8` encoding with error replacement, and sprinkled in `gc.collect()` calls to keep memory under control. Also added epoch-level logging so I could tell how far through the dataset I actually was.

### Phase 4: MCTS Hardening

After some training, I watched the agent play itself and noticed something weird: **it was getting stuck in infinite loops.** It would find a threatening move, the opponent would block, it would retreat, then play the exact same threat again. Over and over. The expected values were also wildly inconsistent — the same position would evaluate as 0.93 on one turn and -0.72 two moves later.

I dug into this and found three overlapping problems:
1. **No repetition detection inside the MCTS tree** — the search treated "loop back to a position I've seen before" as having real value instead of recognizing it as a draw
2. **History amnesia** — I was initializing MCTS from a FEN string every turn, which wiped the 8-move history queue. The neural network was seeing zeroed-out history planes on every single move, so it had no concept of "I've been here before"
3. **Simulation budget was way too low** — 30-50 simulations weren't enough to search past immediate threats and find actual plans

I fixed all three:
- Refactored `Node` to carry a deep copy of the `Board` object (with full move stack) instead of just a FEN string. Now `python-chess` natively catches threefold repetition and 50-move draws
- Added a proper `Board.copy()` that preserves both state and history
- Cranked the simulation budget to 400 everywhere

### Phase 5: The Big Bug Audit

Even after all the architectural changes and dataset training, **the agent still wasn't performing well.** The moves looked slightly more reasonable than random, but it wasn't learning at the rate I expected. Something deeper was wrong.

I went through every single file line by line. What I found was humbling — there were 8 separate issues, and 4 of them were hard crashes that meant parts of the code had literally never worked.

The first thing I found was that `torch.softmax()` in the policy head was missing its required `dim` argument. This is a `TypeError` — the policy network couldn't even produce output. I have no idea how I missed this, but it meant that any training run that actually hit this code path would have crashed immediately. I fixed it by having the policy head output raw logits instead, and applying softmax externally only where needed.

Next, I found a stale variable name. During an earlier refactor I had renamed `board` to `b` throughout the self-play function, but missed one line at the very end — `return training_data, reward, len(board.move_stack)`. This threw a `NameError` after every single game finished. Same kind of thing with `chess.Board(mcts.root.state)` — after the MCTS refactor, `state` was no longer a FEN string but a custom Board object, so passing it to `chess.Board()` would crash.

Then I found the sneakiest bug of the whole project. The history initialization was `self.history = [np.zeros((4,64), np.float32)]*8`. In Python, `[x]*8` doesn't create 8 independent copies — it creates 8 references to the *same* object. So when `push()` appended a new history frame and popped the oldest, 7 out of 8 slots were still pointing to the original zeros. The neural network was seeing empty history planes on every move regardless of what had actually happened in the game. It was playing with complete amnesia. I'd been staring at the history code for a while before I caught this. One-line fix — change to a list comprehension — but massive impact.

On the design side, the biggest problem was the loss function. I was using MSE loss on a 4672-dimensional one-hot policy target. Think about what that means: the target has a single `1.0` at the correct move index and `0.0` everywhere else. MSE computes the gradient for all 4672 outputs, and 4671 of them are pushing toward zero while only 1 is pushing toward one. The gradient from the correct move gets completely drowned out. The network learns to output near-zero everywhere, which is effectively a uniform random policy. Switching to cross-entropy loss was the single biggest quality improvement — suddenly the network could actually learn which move to play.

I also found that the action encoding `from_square * 64 + to_square` couldn't distinguish between promotion types. A pawn push to e8 promoting to a Queen and the same push promoting to a Knight mapped to the exact same index. The network had no way to express a preference.

The value signs in the MCTS were inverted too. In checkmate positions, the side that got checkmated was receiving a *positive* value. The agent was literally learning to lose.

Finally, the supervised training was feeding data sequentially — all games from file `A00-A39.pgn`, then `A40-A79.pgn`, and so on. Since KingBase files are grouped by ECO opening code, the network would overfit to whatever opening system it was currently seeing, then catastrophically forget it when the next file loaded. I replaced this with a 100k-position replay buffer that samples randomly.

### Phase 6: Endgame Improvements

With the bugs fixed, the games started looking genuinely reasonable through the opening and middlegame. But the **endgame was still noisy** — the agent would have a completely winning position with a King and Rook vs a lone King but couldn't find the mate. It would shuffle pieces around aimlessly.

I also noticed that **pawns weren't promoting** even when they had a clear path to the back rank with no defenders in the way. This was partly the action encoding issue (Bug #6 — the network couldn't distinguish promotion types) and partly that there was no incentive to promote.

I made three changes:
- Built a deterministic **alpha-beta endgame solver** (`Chess/endgame_solver.py`) that automatically takes over when ≤7 pieces remain on the board. It uses iterative deepening up to depth 20 with MVV-LVA move ordering, king-cornering heuristics, and passed-pawn bonuses. No neural network noise — just clean, exhaustive search that finds forced checkmates
- Added **promotion reward shaping**: +0.2 for any promotion, +0.1 extra if the promotion square is undefended
- Created a proper **action encoding module** (`Chess/action_encoding.py`) with dedicated index slots for underpromotions (Knight/Rook/Bishop), so the network can actually express a preference for promotion type

The endgame solver was the single most satisfying addition. When it kicks in, you can see `[SOLVER]` in the logs, and the agent goes from aimless shuffling to precise, clinical mating sequences.

---

## References

- [Silver et al., 2017 — Mastering Chess and Shogi by Self-Play (AlphaZero)](https://arxiv.org/abs/1712.01815)
- [KingBase2019 Dataset](https://archive.org/details/KingBase2019)
- [python-chess](https://python-chess.readthedocs.io/)

## License

MIT
