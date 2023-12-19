import chess
import numpy as np

class Board():
    def __init__(self, board = None):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board
        
        self.history = [np.zeros(64, np.uint8)]*8

    def board_to_input(self):
        """Define different states for each piece.
            My representation:
            Turn done
            White done
            Black done
            Attackers x 2 done
            History(no idea how yet)(past 8 frames) done
            Casteling x 4 done
            En Passant done """
        
        
        matrix = []        
        #1 turn representation
        matrix.append(np.ones(64, np.uint8) if self.board.turn else np.zeros(64, np.uint8))
        
        #2 board reprensentation in numpy 
        matrix.append(self.get_numpy_representation(chess.WHITE))
        matrix.append(self.get_numpy_representation(chess.BLACK))
        
        #4 Attckers representation 
        matrix.append(self.get_attacked_squares(chess.WHITE))
        matrix.append(self.get_attacked_squares(chess.BLACK))

        #5 castling representation for each ie. 4x64
        castle = np.asarray(
            [np.ones(64, np.uint8) if self.board.has_kingside_castling_rights(chess.WHITE) else np.zeros(64, np.uint8),
            np.ones(64, np.uint8) if self.board.has_queenside_castling_rights(chess.WHITE) else np.zeros(64, np.uint8),
            np.ones(64, np.uint8) if self.board.has_kingside_castling_rights(chess.BLACK) else np.zeros(64, np.uint8),
            np.ones(64, np.uint8) if self.board.has_queenside_castling_rights(chess.BLACK) else np.zeros(64, np.uint8)], 
            np.uint8)
        matrix.append(castle)
        del castle
        
        
        #6 En passant 
        en_passant = np.zeros(64, np.uint8)
        if self.board.ep_square is not None:
            en_passant[self.board.ep_square] = 1
        matrix.append(en_passant)
        del en_passant

        #7 History for the board
        matrix.append(self.history)
        return matrix

    def get_attacked_squares(self, color):
        attacked = chess.SquareSet()
        for attacker in chess.SquareSet(self.board.occupied_co[color]):
            attacked |= self.board.attacks(attacker)
        
        return np.array(attacked.tolist(), np.uint8)

    def get_numpy_representation(self, color):
        nstate = {'p':1, 'r':2, 'n':3, 'b':4, 'q':5, 'k':6, 'P':-1, 'R':-2, 'N':-3, 'B':-4, 'Q':-5, 'K':-6}
        
        reps = np.zeros(64, np.uint8)
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece is not None and piece.color == color:
                reps[i] = nstate[piece.symbol()]
        return reps
    
    def push_san(self, move):
        self.history.append(np.array([self.get_numpy_representation(chess.WHITE), 
                            self.get_numpy_representation(chess.BLACK),
                            self.get_attacked_squares(chess.WHITE),
                            self.get_attacked_squares(chess.BLACK)]))
        if len(self.history) > 8:
            self.history.pop(0)
        self.board.push_san(move)

    def numpy_to_fen(self, numpy_board):    
        pass

    def get_legal_moves(self):
        pass

