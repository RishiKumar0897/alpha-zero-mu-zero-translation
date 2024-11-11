import logging
import math
import numpy as np
import torch

EPS = 1e-8
log = logging.getLogger(__name__)

class MCTS:
    """Monte Carlo Tree Search for MuZero."""

    def __init__(self, game, network, args):
        self.game = game
        self.network = network
        self.args = args
        self.Qsa = {}  # Q values for state-action pairs
        self.Nsa = {}  # Number of visits for state-action pairs
        self.Ns = {}   # Number of visits for state
        self.Ps = {}   # Policy prior for state
        self.Es = {}   # End status of state
        self.Vs = {}   # Valid actions for state
        # MuZero specific: Store hidden states for learned representations
        self.hidden_states = {}  # Hidden states for each visited node

    def getActionProb(self, canonicalBoard, temp=1):
        """Returns action probabilities for the given state."""
        for _ in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        probs = [x / float(sum(counts)) for x in counts]
        return probs

    def search(self, canonicalBoard):
        """Performs one iteration of MCTS."""


        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            return -self.Es[s]

        if s not in self.Ps:
                
            # Leaf node: use the network
            # MuZero: Use initial inference to get hidden state and predictions
            # Different from AlphaZero which would just do policy, value = network.predict(board)
            policy, value, hidden_state = self.network.initial_inference(torch.FloatTensor(canonicalBoard))
            valids = self.game.getValidMoves(canonicalBoard, 1)
            
            # Ensure policy is properly shaped and on CPU
            if isinstance(policy, torch.Tensor):
                # Apply softmax to convert logits to probabilities
                policy = torch.nn.functional.softmax(policy, dim=-1)
                policy = policy.detach().cpu().squeeze()
            policy = policy.numpy() if isinstance(policy, torch.Tensor) else policy
            
            # Convert valids to numpy if it's a tensor
            if isinstance(valids, torch.Tensor):
                valids = valids.cpu().numpy()
            
            # Mask invalid moves and renormalize
            masked_policy = policy * valids
            sum_Ps_s = np.sum(masked_policy)
            
            if sum_Ps_s > 0:
                self.Ps[s] = masked_policy / sum_Ps_s
            else:
                log.warning(f"Policy values too low or zero after softmax. Raw policy: {policy}, Valid moves: {valids}")
                self.Ps[s] = valids / np.sum(valids)
            
            # Store as tensor for later use
            self.Ps[s] = torch.FloatTensor(self.Ps[s])
            self.Vs[s] = torch.FloatTensor(valids)
            self.Ns[s] = 0

            # MuZero: Store hidden state for future recurrent inference
            self.hidden_states[s] = hidden_state
            return -value
    

        valids = self.Vs[s]
        best_ucb = -float('inf')
        best_action = -1

        # Select action with highest UCB
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    ucb = self.Qsa[(s, a)] + self.args.cpuct * float(self.Ps[s][a]) * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    ucb = self.args.cpuct * float(self.Ps[s][a]) * math.sqrt(self.Ns[s] + EPS)
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_action = a

        a = best_action
        next_s, _ = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, -1)

        # MuZero: Use recurrent inference for state transition
        # Different from AlphaZero which would use game rules
        _, value, _, next_hidden_state = self.network.recurrent_inference(self.hidden_states[s], torch.tensor(a))
        v = self.search(next_s)
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
