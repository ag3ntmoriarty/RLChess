import chess
import math
from edge import Edge
import constant

class Edge:
    def __init__(self, input_node, output_node, action, priority):
        self.input_node = input_node
        self.output_node = output_node
        self.action = action

        # MCTS variables
        self.N = 0
        self.W = 0
        self.priority = priority

    
    def get_uct(self, noise)-> float:
        """
        Returns the Upper Confidence Bound for Trees (UCT) value of the edge
        """
        er = math.log((self.input_node.N + constant.C_Base + 1) / constant.C_Base) + constant.C_init
        ucb = er * (self.P * noise) * (math.sqrt(self.input_node.N) / (1 + self.N))
        if self.input_node.turn:
            return self.W / (self.W + 1) + ucb
        else:
            return -(self.W / (self.W - 1)) + ucb
    
    def __eq__(self, edge: object) -> bool:
        """
        Checks if two edges are equal
        """
        if isinstance(edge, Edge):
            return self.action == edge.action and self.input_node.state == edge.input_node.state
        else:
            return NotImplemented

    def __str__(self):
        return f"Edge: {self.action} | N: {self.N} | W: {self.W} | P: {self.priority} | UCT: {self.get_uct()}"
    
    def __repr__(self):
        return f"Edge: {self.action} | N: {self.N} | W: {self.W} | P: {self.priority} | UCT: {self.get_uct()}"