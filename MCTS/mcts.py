import chess
import numpy as np
from MCTS.edge import Edge
from MCTS.node import Node
import constant

class MCTS:
    def __init__(self, agent, state, stochastic):
        """
        Monte Carlo Tree Search Algorithm 
        """
        self.root = Node(state = state)
        self.game_path: list[Edge] = []
        self.cur_board = chess.Board = None
        self.agent = agent
        self.stochastic = stochastic

    def stimulate(self, n):
        """
        Steps in simulation:
        1. Select child
        2. Expand from child and evaluate
        3. Backpropagation
        """

        for _ in range(n):
            self.game_path = []
            #Select the child node with the maximum UCT value for expansion
            leaf = self.select_child(self.root)

            #Expand selected leaf node
            leaf.N +=1
            leaf = self.expand(leaf)
            
            #Backpropagate on the result
            leaf = self.backpropagation(leaf, leaf.value)

    def select_child(self, node):
        """
        Traverse node with the highest UCT value.

        Returns: The node for expansion.
        """
        while not node.is_leaf():
            if not len(node.edges):
                #Return the node if the node is a terminal node.
                return node

            noise = [1 for _ in range(len(node.edges))]

            if self.stochastic and node == self.root:
                noise = np.random.dirichlet(constant.DIRICHLET * node.edges)
            
            best_edge = None
            best_score = -np.inf
            for i, edge in enumerate(node.edges):
                if edge.get_uct(noise[i]) > best_score:
                    best_score = edge.get_uct(noise[i])
                    best_edge = edge

            if best_edge == None:
                raise Exception('No edge found')

            node = best_edge.output_node
            self.game_path.append(best_edge)
