import math
from MCTS import constant

class Edge:
    def __init__(self, input_node, output_node, action, priority):
        self.input_node = input_node
        self.output_node = output_node
        self.action = action

        # MCTS variables
        self.N = 0
        self.W = 0
        self.priority = priority

    def get_uct(self, noise=1.0)-> float:
        """
        Returns the Upper Confidence Bound for Trees (UCT) value of the edge
        """
        Q = self.W / self.N if self.N > 0 else 0.0
        
        # AlphaZero PUCT formula
        c_puct = 1.0 # Base constant
        U = c_puct * (self.priority * noise) * (math.sqrt(self.input_node.N) / (1 + self.N))
        
        # In our implementation, W is updated from the perspective of the node that just moved.
        # But during selection, input_node is choosing the max UCT. Q represents the value of taking this edge.
        return Q + U
    
    def __eq__(self, edge: object) -> bool:
        if isinstance(edge, Edge):
            return self.action == edge.action and self.input_node.state == edge.input_node.state
        return NotImplemented

    def __str__(self):
        return f"Edge: {self.action} | N: {self.N} | W: {self.W} | P: {self.priority} | UCT: {self.get_uct()}"
    
    def __repr__(self):
        return f"Edge: {self.action} | N: {self.N} | W: {self.W} | P: {self.priority} | UCT: {self.get_uct()}"