import chess
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from Models.net import RLModel
from MCTS.mcts import MCTS
from Chess.board import Board
from Chess.action_encoding import move_to_index

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def save(self, data):
        self.buffer.append(data)
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
        
    def __len__(self):
        return len(self.buffer)

def get_mcts_policy(mcts, temp=1.0):
    """
    Returns the action probabilities from MCTS root edges based on visit counts.
    temp: Temperature parameter to control exploration.
    """
    edges = mcts.root.edges
    counts = np.array([edge.N for edge in edges])
    actions = [edge.action for edge in edges]
    
    # Guard: if MCTS produced no edges (draw/terminal detected inside search),
    # return a uniform policy over legal moves.
    if len(edges) == 0:
        legal_moves = list(mcts.root.state.board.legal_moves)
        if not legal_moves:
            return np.zeros(4672, dtype=np.float32), [], np.array([])
        uniform = np.ones(len(legal_moves), dtype=np.float32) / len(legal_moves)
        full_policy = np.zeros(4672, dtype=np.float32)
        for move, p in zip(legal_moves, uniform):
            full_policy[move_to_index(move) % 4672] = p
        return full_policy, legal_moves, uniform
    
    if temp == 0:
        best_idx = np.argmax(counts)
        probs = np.zeros(len(counts))
        probs[best_idx] = 1.0
    else:
        counts_temp = counts ** (1.0 / temp)
        total = np.sum(counts_temp)
        probs = counts_temp / total if total > 0 else np.ones(len(counts)) / len(counts)
        
    full_policy = np.zeros(4672, dtype=np.float32)
    
    for idx, a in enumerate(actions):
        move_idx = move_to_index(a)
        full_policy[move_idx % 4672] = probs[idx]
        
    return full_policy, actions, probs

def self_play(model, num_simulations=50):
    """
    Plays a single game against itself and returns the training examples.
    """
    b = Board() # Use custom board to maintain history across turns
    examples = []
    
    while not b.board.is_game_over(claim_draw=True) and len(b.board.move_stack) < 200:
        mcts = MCTS(agent=model, state=b.copy(), stochastic=True)
        mcts.simulate(num_simulations)
        
        # In early game, use temp=1 for exploration. Later use temp=0.
        temp = 1.0 if len(b.board.move_stack) < 30 else 0.0
        
        pi, actions, probs = get_mcts_policy(mcts, temp=temp)
        
        # Save state and policy (from the perspective of the current player)
        state_input = b.board_to_input()
        
        # Choose action
        action_idx = np.random.choice(len(actions), p=probs)
        action = actions[action_idx]
        
        # --- REWARD SHAPING ---
        extra_reward = 0.0
        
        # 1. Bonus for capturing an undefended piece
        if b.board.is_capture(action):
            captured_square = action.to_square
            if b.board.is_en_passant(action):
                if b.board.turn == chess.WHITE:
                    captured_square -= 8
                else:
                    captured_square += 8
                    
            opponent = not b.board.turn
            defenders = b.board.attackers(opponent, captured_square)
            if not defenders:
                extra_reward += 0.1
                
        # 2. Bonus for moving to a defended square (covered by friendly piece)
        b_next = b.board.copy()
        b_next.push(action)
        friendly = not b_next.turn # The player who just moved
        if b_next.is_attacked_by(friendly, action.to_square):
            extra_reward += 0.05
        
        # 3. Bonus for pawn promotion
        if action.promotion:
            extra_reward += 0.2  # Strong incentive to promote
            # Extra bonus if promoting to an undefended square (free promotion)
            opponent = not b.board.turn
            if not b.board.attackers(opponent, action.to_square):
                extra_reward += 0.1
            
        examples.append([state_input, pi, b.board.turn, extra_reward])
        b.push(action) # Automatically updates history and pushes to board
        
    # Game over, determine reward
    result = b.board.result(claim_draw=True)
    if result == '1-0':
        reward = 1.0  # White wins
    elif result == '0-1':
        reward = -1.0 # Black wins
    else:
        reward = 0.0  # Draw
        
    # Assign rewards to examples (1 for win, -1 for loss)
    training_data = []
    for state, pi, turn, step_reward in examples:
        r = reward if turn == chess.WHITE else -reward
        r += step_reward
        # Clamp reward to [-1.0, 1.0] to maintain value stability
        r = max(min(r, 1.0), -1.0)
        training_data.append((state, pi, r))
        
    return training_data, reward, len(b.board.move_stack)

def train(model, optimizer, buffer, batch_size=32):
    if len(buffer) < batch_size:
        return 0.0, 0.0, 0.0
        
    device = next(model.parameters()).device
        
    batch = buffer.sample(batch_size)
    states = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32).to(device)
    pis = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.float32).to(device)
    vs = torch.tensor(np.array([x[2] for x in batch]), dtype=torch.float32).unsqueeze(1).to(device)
    
    model.train()
    optimizer.zero_grad()
    
    out_pi_logits, out_v = model(states)
    
    # Value loss: MSE
    value_loss = F.mse_loss(out_v, vs)
    
    # Policy loss: Cross-entropy with soft MCTS targets
    # pis is a probability distribution from MCTS visit counts, not one-hot.
    # Cross-entropy: -sum(pi * log_softmax(logits))
    log_probs = F.log_softmax(out_pi_logits, dim=-1)
    policy_loss = -torch.mean(torch.sum(pis * log_probs, dim=-1))
    
    loss = value_loss + policy_loss
    loss.backward()
    optimizer.step()
    
    return value_loss.item(), policy_loss.item(), loss.item()

def main():
    print("Starting RL Self-Play Training Loop...")
    
    # Check for GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. Training on CUDA.")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS is available. Training on Apple Silicon GPU.")
    else:
        device = torch.device("cpu")
        print("WARNING: GPU is not available. Training on CPU. This will be very slow!")
        
    # Initialize TensorBoard writer
    log_dir = f"runs/rl_chess_experiment_{int(time.time())}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to {log_dir}")
    
    model = RLModel([(1, 42, 8, 8), (1, 10, 8, 8)], [(73*8*8,), 1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    buffer = ReplayBuffer(capacity=10000)
    
    num_episodes = 100
    mcts_simulations = 400
    batch_size = 128
    epochs = 20
    
    global_step = 0
    
    for episode in range(1, num_episodes + 1):
        print(f"\n--- Episode {episode}/{num_episodes} ---")
        model.eval()
        
        # 1. Self Play
        print("Playing game...")
        examples, final_reward, game_length = self_play(model, num_simulations=mcts_simulations)
        for ex in examples:
            buffer.save(ex)
        print(f"Game finished in {game_length} moves. Result: {final_reward}. Buffer size: {len(buffer)}")
        
        # Log self-play metrics
        writer.add_scalar('SelfPlay/GameLength', game_length, episode)
        writer.add_scalar('SelfPlay/FinalReward (White=1, Black=-1, Draw=0)', final_reward, episode)
        
        # 2. Train
        if len(buffer) >= batch_size:
            print("Training network...")
            total_loss = 0.0
            total_v_loss = 0.0
            total_p_loss = 0.0
            
            for epoch in range(epochs):
                v_loss, p_loss, loss = train(model, optimizer, buffer, batch_size=batch_size)
                total_v_loss += v_loss
                total_p_loss += p_loss
                total_loss += loss
                
                # Log batch-level losses
                writer.add_scalar('Loss/Value', v_loss, global_step)
                writer.add_scalar('Loss/Policy', p_loss, global_step)
                writer.add_scalar('Loss/Total', loss, global_step)
                global_step += 1
                
            avg_loss = total_loss / epochs
            print(f"Avg Total Loss: {avg_loss:.4f} (Value: {total_v_loss/epochs:.4f}, Policy: {total_p_loss/epochs:.4f})")
            
        # 3. Save Checkpoint (every 10 episodes)
        if episode % 10 == 0:
            torch.save(model.state_dict(), f"checkpoint_ep{episode}.pth")
            print(f"Saved checkpoint: checkpoint_ep{episode}.pth")
            
    writer.close()
    print("Training finished!")

if __name__ == "__main__":
    main()
