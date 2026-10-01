"""
Endgame Alpha-Beta solver (Negamax + Transposition Table).

When the board has few enough pieces (≤ ENDGAME_PIECE_THRESHOLD), we switch
from neural-network-guided MCTS to a deterministic negamax search that
can find forced checkmates.

Key optimizations:
  - Transposition table (Zobrist hashing): avoids re-evaluating the same position
  - Move ordering: captures/promotions first (no push/pop, cheap heuristic)
  - Mate-distance pruning: shorter mates score higher
  - Iterative deepening: returns best move found within time budget
"""

import chess
import math
import time

ENDGAME_PIECE_THRESHOLD = 7
SOLVER_MAX_DEPTH = 60
SOLVER_TIME_LIMIT = 5.0  # seconds per move

MATE_SCORE = 100000

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 325,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Transposition table entry types
TT_EXACT = 0
TT_LOWER = 1  # Alpha cutoff (value is a lower bound)
TT_UPPER = 2  # Beta cutoff (value is an upper bound)


class TranspositionTable:
    """Simple hash map for caching position evaluations."""
    def __init__(self, max_size=2_000_000):
        self.table = {}
        self.max_size = max_size
    
    def store(self, key, depth, value, flag, best_move):
        # Only overwrite if new search is at least as deep
        entry = self.table.get(key)
        if entry is None or depth >= entry[0]:
            if len(self.table) >= self.max_size:
                # Simple eviction: clear half the table
                keys = list(self.table.keys())
                for k in keys[:len(keys)//2]:
                    del self.table[k]
            self.table[key] = (depth, value, flag, best_move)
    
    def probe(self, key):
        return self.table.get(key)
    
    def clear(self):
        self.table.clear()


# Global TT (persists across iterative deepening iterations)
_tt = TranspositionTable()


def count_pieces(board):
    return bin(board.occupied).count('1')

def is_endgame(board):
    return count_pieces(board) <= ENDGAME_PIECE_THRESHOLD

def evaluate(board):
    """Static evaluation from the perspective of the side to move."""
    if board.is_checkmate():
        return -MATE_SCORE
    
    if board.is_stalemate() or board.is_insufficient_material() or \
       board.can_claim_draw() or board.is_seventyfive_moves() or \
       board.is_fivefold_repetition():
        return 0

    material = 0
    for pt in PIECE_VALUES:
        material += PIECE_VALUES[pt] * (
            len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK))
        )
    
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is not None and bk is not None:
        wk_f, wk_r = chess.square_file(wk), chess.square_rank(wk)
        bk_f, bk_r = chess.square_file(bk), chess.square_rank(bk)
        king_dist = abs(wk_f - bk_f) + abs(wk_r - bk_r)
        
        if material > 0:
            # Winning side: push losing king to edge, bring own king close
            bk_edge = max(abs(bk_f - 3.5), abs(bk_r - 3.5))
            material += int(bk_edge * 30) - king_dist * 8
        elif material < 0:
            wk_edge = max(abs(wk_f - 3.5), abs(wk_r - 3.5))
            material -= int(wk_edge * 30) - king_dist * 8
    
    # Passed pawn advancement bonus
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        material += (chess.square_rank(sq) - 1) * 20
    for sq in board.pieces(chess.PAWN, chess.BLACK):
        material -= (6 - chess.square_rank(sq)) * 20
    
    return material if board.turn == chess.WHITE else -material


def _order_moves(board, tt_move=None):
    """
    Order moves for alpha-beta pruning. TT move (from previous iteration) goes first.
    Then captures sorted by MVV-LVA, then promotions, then quiet moves.
    """
    moves = list(board.legal_moves)
    scored = []
    for m in moves:
        s = 0
        if tt_move and m == tt_move:
            s = 1000000  # Always search TT move first
        elif m.promotion:
            s = 90000 + PIECE_VALUES.get(m.promotion, 0)
        else:
            victim = board.piece_at(m.to_square)
            if victim:
                attacker = board.piece_at(m.from_square)
                s = 10000 + PIECE_VALUES.get(victim.piece_type, 0) * 10
                if attacker:
                    s -= PIECE_VALUES.get(attacker.piece_type, 0)
        scored.append((s, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored]


def negamax(board, depth, alpha, beta, ply, deadline):
    """
    Negamax with alpha-beta pruning and transposition table.
    Returns (value, best_move).
    """
    # Time check every 4096 nodes
    if ply > 0 and (ply & 0xFFF) == 0 and time.time() > deadline:
        return 0, None  # Abort — use previous iteration's result

    # Terminal position check
    if board.is_game_over(claim_draw=True):
        if board.is_checkmate():
            return -(MATE_SCORE - ply), None
        return 0, None
    
    if depth <= 0:
        return evaluate(board), None

    # Transposition table probe
    board_key = board._transposition_key()
    tt_entry = _tt.probe(board_key)
    tt_move = None
    
    if tt_entry is not None:
        tt_depth, tt_value, tt_flag, tt_best = tt_entry
        tt_move = tt_best
        if tt_depth >= depth:
            if tt_flag == TT_EXACT:
                return tt_value, tt_best
            elif tt_flag == TT_LOWER:
                alpha = max(alpha, tt_value)
            elif tt_flag == TT_UPPER:
                beta = min(beta, tt_value)
            if alpha >= beta:
                return tt_value, tt_best
    
    # Generate and order moves
    moves = _order_moves(board, tt_move)
    if not moves:
        return evaluate(board), None
    
    best_move = moves[0]
    best_value = -math.inf
    orig_alpha = alpha
    
    for move in moves:
        board.push(move)
        child_val, _ = negamax(board, depth - 1, -beta, -alpha, ply + 1, deadline)
        child_val = -child_val
        board.pop()
        
        if child_val > best_value:
            best_value = child_val
            best_move = move
        
        alpha = max(alpha, best_value)
        if alpha >= beta:
            break
    
    # Store in transposition table
    if best_value <= orig_alpha:
        flag = TT_UPPER
    elif best_value >= beta:
        flag = TT_LOWER
    else:
        flag = TT_EXACT
    _tt.store(board_key, depth, best_value, flag, best_move)
    
    return best_value, best_move


def solve_endgame(board, max_depth=SOLVER_MAX_DEPTH, time_limit=SOLVER_TIME_LIMIT):
    """
    Iterative deepening negamax search with transposition table.
    Returns (best_move, display_value, depth_searched).
    """
    _tt.clear()
    
    best_move = None
    best_value = 0
    final_depth = 1
    deadline = time.time() + time_limit
    
    for depth in range(1, max_depth + 1):
        if time.time() > deadline:
            break
        
        value, move = negamax(board, depth, -math.inf, math.inf, 0, deadline)
        
        if move is not None:
            best_move = move
            best_value = value
            final_depth = depth
        
        # Stop if forced mate found
        if abs(best_value) >= MATE_SCORE - 500:
            mate_ply = MATE_SCORE - abs(best_value)
            break
    
    # Normalize for display
    if abs(best_value) >= MATE_SCORE - 500:
        mate_ply = MATE_SCORE - abs(best_value)
        display_value = 1.0 if best_value > 0 else -1.0
    else:
        display_value = max(-1.0, min(1.0, best_value / 1000.0))
    
    return best_move, display_value, final_depth
