import chess
from edge import Edge

class Node:
    def __init__(self, state):
        self.state = state
        self.turn  = chess.Board(state).turn
        self.edges = []
        
        # Number of times node has been visited
        self.N = 0 
        
        # Value of node
        self.value = 0 
    
    def __eq__(self, node: object) -> bool:
        """
        Checks if two nodes are equal
        """
        if isinstance(node, Node):
            return self.state == Node.state
    
    def step(self, action)-> str:
        """ 
        Returns the new state of board after taking action
        in the form of a FEN string
        """
        board = chess.Board(self.state)
        board.push(action)
        new_state = board.fen()
        del board
        return new_state
    
    def game_over(self)-> bool:
        """ 
        Checks if the game is over
        """
        board = chess.Board(self.state)
        return board.is_game_over()
    
    def add_child(self, child, action, priority)-> Edge:
        """
        Adds a child node to the current node
        """

        edge = Edge(input_node=self, output_node=child, action=action, priority=priority)
        self.edges.append(edge)
        return edge
    
    def is_leaf(self)-> bool:
        """
        Checks if the current node is a leaf node
        """
        return self.N == 0
    
    def get_children(self)-> list:
        """
        Returns a list of child nodes
        """
        children = []
        for edge in self.edges:
            children.append(edge.output_node)
            children.extend(edge.output_node.get_children())
        return children
    
    def get_edge(self, action)-> Edge:
        """
        Returns the edge corresponding to the action
        """
        for edge in self.edges:
            if edge.action == action:
                return edge
        return None

