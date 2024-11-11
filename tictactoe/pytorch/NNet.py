import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim

from muzeroimplementation import TicTacToeNNet as MuZeroNet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.game = game
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.device = torch.device("cuda" if args.cuda else "cpu")
        
        # MuZero: Uses MuZeroNet architecture instead of traditional CNN
        self.nnet = MuZeroNet(
            observation_size=(1, self.board_x, self.board_y),
            action_size=self.action_size,
            hidden_size=args.num_channels,
            device=self.device
        ).to(self.device)
        
        if args.cuda:
            self.nnet.cuda()

    # MuZero: New method - Replaces simple forward pass in AlphaZero
    def initial_inference(self, board):
        """
        board: np array with board
        Returns: (policy, value, hidden_state)
        """
        # Preparing input
        if not isinstance(board, torch.Tensor):
            board = torch.FloatTensor(board.astype(np.float64))
        if len(board.shape) == 2:
            board = board.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            
        board = board.to(self.device)

        self.nnet.eval()
        with torch.no_grad():
            hidden_state, policy_logits, value = self.nnet.initial_inference(board)
            
            # Apply softmax to policy logits
            policy = torch.softmax(policy_logits, dim=1)
            
        return (
            policy.cpu().numpy()[0],
            value.item(),
            hidden_state
        )

    # MuZero: New method - Replaces recurrent pass in AlphaZero
    def recurrent_inference(self, hidden_state, action):
        """
        hidden_state: tensor from initial_inference
        action: integer action index
        Returns: (value, reward, policy, new_hidden_state)
        """
        if not isinstance(action, torch.Tensor):
            action = torch.tensor([[action]], device=self.device)
        
        self.nnet.eval()
        with torch.no_grad():
            new_hidden_state, reward, policy_logits, value = self.nnet.recurrent_inference(
                hidden_state, 
                action
            )
            
            # Apply softmax to policy logits
            policy = torch.softmax(policy_logits, dim=1)
            
        return (
            value.item(),
            reward.item(),
            policy.cpu().numpy()[0],
            new_hidden_state
        )

    def train(self, examples):
        """
        examples: list of tuples of (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr)

        self.nnet.train()
        
        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            
            batch_count = int(len(examples) / args.batch_size)
            t = tqdm(range(batch_count), desc='Training Net')
            
            # MuZero specific: Additional loss components
            representation_losses = AverageMeter()
            dynamics_losses = AverageMeter()
            policy_losses = AverageMeter()
            value_losses = AverageMeter()
            reward_losses = AverageMeter()

            # AlphaZero only needs policy_losses and value_losses
        
            # Current implementation is simplified
            # Full MuZero would include:
            # 1. Representation loss
            # 2. Dynamics loss
            # 3. Reward prediction loss
            # 4. Multiple step unrolling

            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                
                # Prepare input
                boards = torch.FloatTensor(np.array(boards)).to(self.device)
                target_pis = torch.FloatTensor(np.array(pis)).to(self.device)
                target_vs = torch.FloatTensor(np.array(vs)).to(self.device)
                
                # MuZero forward pass
                hidden_states, policy_logits, values = self.nnet.initial_inference(boards)
                
                # For simplicity, we'll just use the initial outputs for loss
                # In a full implementation, you'd want to unroll the model for multiple steps
                
                # Calculate losses
                policy_loss = self.loss_pi(target_pis, policy_logits)
                value_loss = self.loss_v(target_vs, values)
                
                # Total loss
                total_loss = policy_loss + value_loss
                
                # Optimize
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Record losses
                policy_losses.update(policy_loss.item(), boards.size(0))
                value_losses.update(value_loss.item(), boards.size(0))
                
                # Update progress bar
                t.set_postfix(Loss_pi=policy_losses, Loss_v=value_losses)

    def predict(self, board):
        """
        board: np array with board
        """
        # This method should use initial_inference instead of direct network call
        policy, value, _ = self.initial_inference(board)
        return policy, value

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])