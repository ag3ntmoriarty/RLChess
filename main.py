import chess
import torch
import sys
import os
import argparse
from Models.net import RLModel
from MCTS.mcts import MCTS
from Chess.board import Board

def get_best_move(mcts, n_simulations=100):
    mcts.simulate(n_simulations)
    best_edge = max(mcts.root.edges, key=lambda edge: edge.N)
    return best_edge.action

def print_board(board):
    print("\n----------------")
    print(board)
    print("----------------\n")

def main():
    parser = argparse.ArgumentParser(description="Play against RL Chess Agent")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to load")
    args = parser.parse_args()
    
    print("Loading RL Chess Agent...")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU (CUDA).")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS).")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
        
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

    b = Board() # Use custom board to maintain history
    
    print("Welcome to RL Chess!")
    print("Enter moves in standard algebraic notation (e.g., e2e4, Nf3, O-O).")
    
    user_color = None
    while user_color not in ['w', 'b']:
        user_color = input("Do you want to play as White (w) or Black (b)? ").strip().lower()

    is_user_turn = (user_color == 'w')

    while not b.board.is_game_over():
        print_board(b.board)
        
        if is_user_turn:
            move = None
            while move is None:
                move_str = input("Your move: ").strip()
                try:
                    move = b.board.parse_san(move_str)
                    if move not in b.board.legal_moves:
                        print("Illegal move.")
                        move = None
                except ValueError:
                    try:
                        move = chess.Move.from_uci(move_str)
                        if move not in b.board.legal_moves:
                            print("Illegal move.")
                            move = None
                    except ValueError:
                        print("Invalid format. Please use SAN (e.g. e4) or UCI (e.g. e2e4).")
                        move = None
            b.push(move)
        else:
            print("Agent is thinking...")
            mcts = MCTS(agent=model, state=b.copy(), stochastic=False)
            best_move = get_best_move(mcts, n_simulations=400) 
            print(f"Agent plays: {b.board.san(best_move)}")
            b.push(best_move)

        is_user_turn = not is_user_turn

    print_board(b.board)
    print("Game Over!")
    print("Result:", b.board.result())

if __name__ == "__main__":
    main()
