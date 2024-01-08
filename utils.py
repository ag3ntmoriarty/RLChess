from Chess import Board
import chess
b =Board()
b.push_san("e4")
matrix = b.board_to_input()
print(len(matrix))
b.push_san("e5")
matrix = b.board_to_input()
print(len(matrix))

b.push_san("Nf3")
b.push_san("Nc6")
matrix = b.board_to_input()
print(len(matrix))

b.push_san("Bb5")
b.push_san("a6")

b.push_san("Ba4")
matrix = b.board_to_input()
print(len(matrix))
print(matrix)
b.push_san("Nf6")
matrix = b.board_to_input()
print(matrix.size)
b.push_san("O-O")
print(b.board)
