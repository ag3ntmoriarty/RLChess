import chess
from MCTS.edge import Edge

class Node:
    def __init__(self, state):
        self.state = state # Custom Board object
        self.turn  = state.board.turn
        self.edges = []
        
        # Number of times node has been visited
        self.N = 0 
        
        # Value of node
        self.value = 0 
    
    def __eq__(self, node: object) -> bool:
        if isinstance(node, Node):
            return self.state.board.fen() == node.state.board.fen()
        return False
    
    def step(self, action):
        new_state = self.state.copy()
        new_state.push(action)
        return new_state
    
    def game_over(self)-> bool:
        return self.state.board.is_game_over(claim_draw=True)
    
    def add_child(self, child, action, priority)-> Edge:
        edge = Edge(input_node=self, output_node=child, action=action, priority=priority)
        self.edges.append(edge)
        return edge
    
    def is_leaf(self)-> bool:
        return self.N == 0
    
    def get_children(self)-> list:
        children = []
        for edge in self.edges:
            children.append(edge.output_node)
            children.extend(edge.output_node.get_children())
        return children
    
    def get_edge(self, action)-> Edge:
        for edge in self.edges:
            if edge.action == action:
                return edge
        return None
