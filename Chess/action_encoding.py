"""
Shared action encoding utilities.

Maps chess moves to/from a flat action index for the policy network.

Encoding scheme (total 4672 slots):
  - Normal moves + queen promotions: from_sq * 64 + to_sq  → indices 0–4095
  - Knight underpromotion:  4096 + pawn_file * 6 + side * 3 + direction  → 4096–4143
  - Rook underpromotion:    4144 + pawn_file * 6 + side * 3 + direction  → 4144–4191
  - Bishop underpromotion:  4192 + pawn_file * 6 + side * 3 + direction  → 4192–4239

where:
  - pawn_file = from_square % 8  (0–7)
  - side = 0 if White promotes (from rank 7), 1 if Black promotes (from rank 2)
  - direction = (to_file - pawn_file) + 1  → 0=left capture, 1=straight, 2=right capture
"""

import chess

POLICY_SIZE = 4672

def move_to_index(move):
    """Convert a chess.Move to a policy index (0–4671)."""
    if move.promotion and move.promotion != chess.QUEEN:
        # Underpromotion: use dedicated slots so the network can distinguish them
        pawn_file = move.from_square % 8
        to_file = move.to_square % 8
        direction = (to_file - pawn_file) + 1  # 0, 1, or 2
        side = 0 if move.from_square >= 48 else 1  # White promotes from rank 7 (sq 48-55)
        compact = pawn_file * 6 + side * 3 + direction  # 0–47
        
        # KNIGHT=2, BISHOP=3, ROOK=4
        promo_type = move.promotion - 2  # 0=Knight, 1=Bishop, 2=Rook
        return 4096 + promo_type * 48 + compact
    else:
        # Normal move or queen promotion
        return (move.from_square * 64 + move.to_square) % 4096

def index_to_promotion_type(idx):
    """Given a policy index, return the promotion piece type or None."""
    if idx >= 4096:
        offset = idx - 4096
        promo_type = offset // 48  # 0=Knight, 1=Bishop, 2=Rook
        return promo_type + 2  # chess.KNIGHT=2, chess.BISHOP=3, chess.ROOK=4
    return None
