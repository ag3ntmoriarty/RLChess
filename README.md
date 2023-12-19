# RLChessAgent
Make a at first a chess engine with lil previous knowledge
Train a similar 0 knowldge chess engine 


* Establish board
    Pieces:
    King
    Queen
    Rook
    Bishop
    Knight
    Pawn

TO DO:
My representation:
Turn done
White done
Black done
Attackers x 2 done
History(no idea how yet)(past 8 frames) done
Casteling x 4 done
En Passant done




Using alphago's representation of board position but may lead to memory overflow
Current board size = 8x8x(12*8+8)

* Make a search tree for games possible combinations
* Reduce search tree with neural net (Value Net).. V = f(board)
* RL Agent (Policy Net) to evaluate the net and take the best possible path
* Self play to improve policy net and the value net is trained on classic SL approach