import chess
import chess.pgn
import torch
import os
import argparse
from Models.net import RLModel
from MCTS.mcts import MCTS
from Chess.board import Board
from Chess.endgame_solver import is_endgame, solve_endgame

def load_model(checkpoint_path, device):
    model = RLModel([(1, 42, 8, 8), (1, 10, 8, 8)], [(73*8*8,), 1]).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        print(f"Successfully loaded checkpoint: {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint '{checkpoint_path}' not found or not provided. Using random weights.")
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Arena: Pit two RL Chess Agents against each other")
    parser.add_argument("--white", type=str, required=True, help="Path to White's model checkpoint")
    parser.add_argument("--black", type=str, required=True, help="Path to Black's model checkpoint")
    parser.add_argument("--sims", type=int, default=400, help="Number of MCTS simulations per move")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Running Arena simulation on {device}...\n")
    
    print("Loading White Agent...")
    model_white = load_model(args.white, device)
    
    print("\nLoading Black Agent...")
    model_black = load_model(args.black, device)

    b = Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Agent vs Agent Arena"
    game.headers["White"] = f"Agent ({os.path.basename(args.white)})"
    game.headers["Black"] = f"Agent ({os.path.basename(args.black)})"
    
    node = game
    
    print("\nStarting the clash...")
    move_count = 1
    
    log_file = open("arena_log.txt", "w")
    def log_print(text):
        print(text)
        log_file.write(text + "\n")
    
    while not b.board.is_game_over(claim_draw=True) and b.board.fullmove_number <= 100:
        is_white_turn = b.board.turn == chess.WHITE
        current_model = model_white if is_white_turn else model_black
        player_name = "White" if is_white_turn else "Black"
        
        log_print(f"\n--- Move {move_count} ({player_name} to play) ---")
        log_print(str(b.board))
        
        # Check if we should switch to the endgame solver
        if is_endgame(b.board):
            best_move, value, depth = solve_endgame(b.board)
            log_print(f"[SOLVER] {player_name} plays: {b.board.san(best_move)} (Depth: {depth}, Value: {value:.3f})")
        else:
            mcts = MCTS(agent=current_model, state=b.copy(), stochastic=True) 
            mcts.simulate(args.sims) 
            
            edges = mcts.root.edges
            best_edge = max(edges, key=lambda e: e.N)
            best_move = best_edge.action
            
            q_val = best_edge.W / best_edge.N if best_edge.N > 0 else 0
            log_print(f"[MCTS] {player_name} plays: {b.board.san(best_move)} (Visits: {best_edge.N}, EV: {q_val:.3f})")
        
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
    
    print("\nArena match finished! Game log and PGN successfully saved to 'arena_log.txt'.")

if __name__ == "__main__":
    main()
