import chess
import chess.pgn
import torch
import os
import argparse
from Models.net import RLModel
from MCTS.mcts import MCTS
from Chess.board import Board

def main():
    parser = argparse.ArgumentParser(description="Run Agent Self-Play")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to load")
    args = parser.parse_args()

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Running agent self-play simulation on {device}...")
    
    # Initialize model
    model = RLModel([(1, 42, 8, 8), (1, 10, 8, 8)], [(73*8*8,), 1]).to(device)
    
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
            print(f"Successfully loaded checkpoint: {args.checkpoint}")
        else:
            print(f"Warning: Checkpoint file '{args.checkpoint}' not found. Using random weights.")
    else:
        print("No checkpoint provided. Using randomly initialized weights.")
        
    model.eval()

    b = Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Agent Self-Play Debug"
    game.headers["White"] = "RL Agent (White)"
    game.headers["Black"] = "RL Agent (Black)"
    
    node = game
    
    print("Starting game...")
    move_count = 1
    
    # We log the game both to terminal and to a file
    log_file = open("self_play_log.txt", "w")
    
    def log_print(text):
        print(text)
        log_file.write(text + "\n")
    
    # To prevent infinite games, limit to 200 half-moves (100 full turns)
    while not b.board.is_game_over(claim_draw=True) and b.board.fullmove_number <= 100:
        log_print(f"\n--- Move {move_count} ({'White' if b.board.turn == chess.WHITE else 'Black'} to play) ---")
        log_print(str(b.board))
        
        # Initialize MCTS
        mcts = MCTS(agent=model, state=b.copy(), stochastic=True) 
        mcts.simulate(400) 
        
        # Get the best move based on visits
        edges = mcts.root.edges
        best_edge = max(edges, key=lambda e: e.N)
        best_move = best_edge.action
        
        q_val = best_edge.W / best_edge.N if best_edge.N > 0 else 0
        log_print(f"Agent chooses: {b.board.san(best_move)} (Visits: {best_edge.N}, Expected Value: {q_val:.3f})")
        
        b.push(best_move)
        node = node.add_variation(best_move)
        
        move_count += 1
        
    log_print("\n--- Final Board ---")
    log_print(str(b.board))
    
    result = b.board.result(claim_draw=True)
    if b.board.fullmove_number > 100 and not b.board.is_game_over(claim_draw=True):
        result = "1/2-1/2"
        log_print("\nGame Over! Result: Draw (Stopped at 100 move limit)")
    else:
        log_print(f"\nGame Over! Result: {result}")
    
    game.headers["Result"] = result
    
    log_file.write("\n\n=== PGN FORMAT ===\n")
    log_file.write(str(game))
    log_file.close()
    
    print("\nGame log and PGN successfully saved to 'self_play_log.txt'!")
    print("Tip: You can copy the PGN output at the bottom of the file into https://lichess.org/paste to watch an animated replay of the game.")

if __name__ == "__main__":
    main()
