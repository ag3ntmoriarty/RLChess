import chess
import chess.pgn
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import os
import glob
from torch.utils.tensorboard import SummaryWriter
from Models.net import RLModel
from Chess.board import Board
from Chess.action_encoding import move_to_index

def process_game(game):
    """
    Parses a single game and returns a list of (state, target_move_index, result)
    """
    result_str = game.headers.get("Result", "*")
    if result_str == "1-0":
        final_result = 1.0
    elif result_str == "0-1":
        final_result = -1.0
    elif result_str == "1/2-1/2":
        final_result = 0.0
    else:
        return [] # unknown result

    examples = []
    b = Board() # custom Board tracks history correctly
    
    for move in game.mainline_moves():
        # Get state input with history
        state_input = b.board_to_input()
        
        # Determine policy target as an integer index (for cross-entropy)
        target_idx = move_to_index(move)
        
        # Append from perspective of the player to move
        examples.append([state_input, target_idx, b.board.turn])
        
        # Push move to internal board and history queue
        b.push(move)
        
    training_data = []
    for state, target_idx, turn in examples:
        r = final_result if turn == chess.WHITE else -final_result
        training_data.append((state, target_idx, r))
        
    return training_data

def train_batch(model, optimizer, batch, device):
    states = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32).to(device)
    # Policy targets are now integer indices for cross-entropy
    target_indices = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.long).to(device)
    vs = torch.tensor(np.array([x[2] for x in batch]), dtype=torch.float32).unsqueeze(1).to(device)
    
    model.train()
    optimizer.zero_grad()
    
    out_pi_logits, out_v = model(states)
    
    value_loss = F.mse_loss(out_v, vs)
    # Cross-entropy loss: F.cross_entropy expects raw logits and integer class targets
    policy_loss = F.cross_entropy(out_pi_logits, target_indices)
    
    loss = value_loss + policy_loss
    loss.backward()
    optimizer.step()
    
    return value_loss.item(), policy_loss.item(), loss.item()

def main():
    print("Starting Supervised Learning on KingBase Dataset...")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU (CUDA).")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS).")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
        
    log_dir = f"runs/supervised_chess_{int(time.time())}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to {log_dir}")
    
    # Initialize the scaled up model
    model = RLModel([(1, 42, 8, 8), (1, 10, 8, 8)], [(73*8*8,), 1], d_model=512, nhead=8, num_layers=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Lower LR for large models
    
    pgn_files = glob.glob("KingBase/*.pgn")
    if not pgn_files:
        print("No PGN files found in KingBase/ directory!")
        return
        
    print(f"Found {len(pgn_files)} PGN files.")
    
    batch_size = 128 # Reduced to prevent potential OOM
    replay_buffer = []
    max_buffer_size = 100000 # Hold up to 100k positions for random sampling
    global_step = 0
    games_processed = 0
    num_epochs = 3 # Can be adjusted
    
    import gc
    import random
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n=== Starting Epoch {epoch}/{num_epochs} ===")
        for pgn_file in pgn_files:
            print(f"Processing {pgn_file}...")
            with open(pgn_file, "r", encoding="utf-8", errors="replace") as pgn:
                while True:
                    try:
                        game = chess.pgn.read_game(pgn)
                    except Exception as e:
                        print(f"Skipping corrupted game in PGN: {e}")
                        continue
                        
                    if game is None:
                        break # End of file
                        
                    try:
                        data = process_game(game)
                        replay_buffer.extend(data)
                        games_processed += 1
                        
                        # Cap buffer size by removing oldest entries
                        if len(replay_buffer) > max_buffer_size:
                            replay_buffer = replay_buffer[-max_buffer_size:]
                    except Exception as e:
                        # Catch any illegal moves or board issues
                        pass
                    
                    # Free memory
                    del game
                    
                    if games_processed % 100 == 0:
                        print(f"[Epoch {epoch}] Parsed {games_processed} total games... Buffer size: {len(replay_buffer)}")
                        gc.collect() # Force garbage collection
                    
                    # Train by sampling randomly from the buffer (Fix #8: shuffling)
                    if len(replay_buffer) >= batch_size:
                        train_batch_data = random.sample(replay_buffer, batch_size)
                        
                        v_loss, p_loss, loss = train_batch(model, optimizer, train_batch_data, device)
                        
                        writer.add_scalar('Supervised/ValueLoss', v_loss, global_step)
                        writer.add_scalar('Supervised/PolicyLoss', p_loss, global_step)
                        writer.add_scalar('Supervised/TotalLoss', loss, global_step)
                        global_step += 1
                        
                        if global_step % 100 == 0:
                            print(f"[Epoch {epoch}] Step {global_step} - Loss: {loss:.4f} (V: {v_loss:.4f}, P: {p_loss:.4f})")
                            
                        if global_step % 5000 == 0:
                            torch.save(model.state_dict(), f"supervised_model_step_{global_step}.pth")
                            print(f"Checkpoint saved: supervised_model_step_{global_step}.pth")

    # Save final model
    torch.save(model.state_dict(), "supervised_model_final.pth")
    print("Training complete! Model saved to supervised_model_final.pth")
    writer.close()

if __name__ == "__main__":
    main()
