import chess
import numpy as np

class Board():
    def __init__(self, board = None):
        if board is None:
            self.board = chess.Board()
        else:
            if isinstance(board, str):
                self.board = chess.Board(board)
            else:
                self.board = board
        
        self.history = [np.zeros((4, 64), np.float32) for _ in range(8)]

    def board_to_input(self):
        matrix = []        
        #1 turn representation
        matrix.append(np.ones(64, np.float32) if self.board.turn else np.zeros(64, np.float32))
        
        #2 board reprensentation in numpy 
        matrix.append(self.get_numpy_representation(chess.WHITE))
        matrix.append(self.get_numpy_representation(chess.BLACK))
        
        #3 Attckers representation 
        matrix.append(self.get_attacked_squares(chess.WHITE))
        matrix.append(self.get_attacked_squares(chess.BLACK))

        #4 castling representation for each ie. 4x64
        matrix.append(np.ones(64, np.float32) if self.board.has_kingside_castling_rights(chess.WHITE) else np.zeros(64, np.float32))
        matrix.append(np.ones(64, np.float32) if self.board.has_queenside_castling_rights(chess.WHITE) else np.zeros(64, np.float32))
        matrix.append(np.ones(64, np.float32) if self.board.has_kingside_castling_rights(chess.BLACK) else np.zeros(64, np.float32))
        matrix.append(np.ones(64, np.float32) if self.board.has_queenside_castling_rights(chess.BLACK) else np.zeros(64, np.float32))

        #5 En passant 
        en_passant = np.zeros(64, np.float32)
        if self.board.ep_square is not None:
            en_passant[self.board.ep_square] = 1
        matrix.append(en_passant)

        #6 History for the board
        for i in self.history:
            for j in i:
                matrix.append(j)
                
        # matrix is 42 x 64. Reshape to 42 x 8 x 8
        return np.reshape(np.array(matrix, dtype=np.float32), (42, 8, 8))

    def copy(self):
        new_b = Board(self.board.copy())
        new_b.history = [np.copy(h) for h in self.history]
        return new_b

    def get_attacked_squares(self, color):
        attacked = chess.SquareSet()
        for attacker in chess.SquareSet(self.board.occupied_co[color]):
            attacked |= self.board.attacks(attacker)
        
        return np.array(attacked.tolist(), np.float32)

    def get_numpy_representation(self, color):
        nstate = {'p':1, 'r':2, 'n':3, 'b':4, 'q':5, 'k':6, 'P':-1, 'R':-2, 'N':-3, 'B':-4, 'Q':-5, 'K':-6}
        
        reps = np.zeros(64, np.float32)
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece is not None and piece.color == color:
                reps[i] = nstate[piece.symbol()]
        return reps
    
    def push(self, move):
        self.history.append(np.array([self.get_numpy_representation(chess.WHITE), 
                            self.get_numpy_representation(chess.BLACK),
                            self.get_attacked_squares(chess.WHITE),
                            self.get_attacked_squares(chess.BLACK)]))
        if len(self.history) > 8:
            self.history.pop(0)
        self.board.push(move)

    def push_san(self, move):
        self.push(self.board.parse_san(move))

    def numpy_to_fen(self, numpy_board):    
        pass

    def get_legal_moves(self):
        return list(self.board.legal_moves)
