import chess
import numpy as np
import torch
from MCTS.edge import Edge
from MCTS.node import Node
from MCTS import constant
from Chess.board import Board
from Chess.action_encoding import move_to_index

class MCTS:
    def __init__(self, agent, state, stochastic=False):
        """
        Monte Carlo Tree Search Algorithm 
        """
        self.root = Node(state=state)
        self.game_path = []
        self.agent = agent
        self.stochastic = stochastic

    def simulate(self, n): # Fixed typo from stimulate
        for _ in range(n):
            self.game_path = []
            leaf = self.select_child(self.root)
            self.expand(leaf)
            self.backpropagation(leaf.value)

    def select_child(self, node):
        while not node.is_leaf():
            if not len(node.edges):
                return node

            noise = [1 for _ in range(len(node.edges))]
            if self.stochastic and node == self.root:
                noise = np.random.dirichlet([constant.DIRICHLET] * len(node.edges))
            
            best_edge = None
            best_score = -np.inf
            for i, edge in enumerate(node.edges):
                score = edge.get_uct(noise[i] if self.stochastic and node == self.root else 1.0)
                if score > best_score:
                    best_score = score
                    best_edge = edge

            if best_edge == None:
                return node

            node = best_edge.output_node
            self.game_path.append(best_edge)
        return node

    def expand(self, leaf):
        if leaf.game_over():
            result = leaf.state.board.result(claim_draw=True)
            if result == '1-0':
                # White won. leaf.turn is the side to move (Black, who just got checkmated).
                # Value is from the perspective of the player to move at this node.
                # Black is to move and lost → value = -1.0 for them.
                leaf.value = -1.0 if leaf.turn == chess.BLACK else 1.0
            elif result == '0-1':
                # Black won. White is to move and lost.
                leaf.value = -1.0 if leaf.turn == chess.WHITE else 1.0
            else:
                leaf.value = 0.0 # Draw
            return

        device = next(self.agent.parameters()).device
        input_tensor = torch.tensor(leaf.state.board_to_input(), dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            policy_logits, value = self.agent(input_tensor)
        
        leaf.value = value.item()
        
        # Apply softmax to convert logits to probabilities for MCTS priors
        policy = torch.softmax(policy_logits, dim=-1).squeeze().cpu().numpy()
        legal_moves = leaf.state.get_legal_moves()
        
        if not legal_moves:
            return

        # Use shared action encoding for consistent mapping
        for move in legal_moves:
            idx = move_to_index(move)
            prob = policy[idx % len(policy)]
            child_state = leaf.step(move)
            child_node = Node(child_state)
            leaf.add_child(child_node, action=move, priority=prob)

        # Normalize prior probabilities
        sum_p = sum(edge.priority for edge in leaf.edges)
        if sum_p > 0:
            for edge in leaf.edges:
                edge.priority /= sum_p

        leaf.N += 1

    def backpropagation(self, value):
        # Value is from the perspective of the leaf node.
        v = value
        for edge in reversed(self.game_path):
            v = -v # Zero sum
            edge.W += v
            edge.N += 1
            edge.input_node.N += 1
