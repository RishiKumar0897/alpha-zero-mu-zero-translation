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

from .TicTacToeNNet import TicTacToeNNet as tnnet
# Keras handles GPU allocation automatically, while PyTorch needs explicit CUDA management
args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})


class NNetWrapper(NeuralNet):
    """
    Neural Network wrapper class similar to Keras version but with PyTorch-specific implementations.
    Main differences are in tensor handling, GPU management, and training loop structure.
    """
    def __init__(self, game):
        self.nnet = tnnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        # PyTorch-specific: Explicit GPU assignment
        if args.cuda:
            self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        # PyTorch requires explicit optimizer initialization
        optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            # PyTorch requires explicit model training mode setting
            self.nnet.train()
            # PyTorch requires explicit loss tracking
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                # PyTorch requires explicit batch sampling
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                # manual tensor conversion
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # Prepare input
                # keras handles device placement automatically
                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), \
                        target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # Compute output
                out_pi, out_v = self.nnet(boards)

                # keras handles loss computation automatically via model.compile()
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # Record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # Compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board):
        """
        board: np array with board
        """
        # Preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if args.cuda:
            board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)

        # Set model to evaluation mode
        # PyTorch requires explicit model evaluation mode setting
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)
        # PyTorch requires explicit tensor conversion for CPU access
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        """
        Checkpoint saving differs:
        - Keras: Saves complete model or weights using model.save() or model.save_weights()
        - PyTorch: Saves state dictionary containing model parameters
        """
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
        """
        Checkpoint loading differs:
        - Keras: Loads complete model or weights using load_model() or load_weights()
        - PyTorch: Loads state dictionary and manually assigns to model
        """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])