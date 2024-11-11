import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys
sys.path.append('..')
import torch
from utils import *
from NeuralNet import NeuralNet

import argparse
from .TicTacToeNNet import TicTacToeNNet as onnet

"""
NeuralNet wrapper class for the TicTacToeNNet.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on (copy-pasted from) the NNet by SourKream and Surag Nair.
"""

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 512,
})

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs, _= list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.nnet.model.fit(x = input_boards, y = [target_pis, target_vs], batch_size = args.batch_size, epochs = args.epochs)

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = board[np.newaxis, :, :]

        # run
        pi, v = self.nnet.model.predict(board, verbose=False)

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # change extension
        filename = filename.split(".")[0] + ".h5"

        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # change extension
        filename = filename.split(".")[0] + ".h5"

        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ValueError("No model in path '{}'".format(filepath))
        self.nnet.model.load_weights(filepath)
    def initial_inference(self, board):
        """
        MuZero initial inference - converts AlphaZero's predict() to MuZero format
        """
        if isinstance(board, torch.Tensor):
            board = board.numpy()
        
        pi, v = self.predict(board)
        # For TicTacToe, we don't need a hidden state, so return None
        hidden_state = None
        return pi, v, hidden_state

    def recurrent_inference(self, hidden_state, action):
        """
        MuZero recurrent inference - for TicTacToe, we can just use the same prediction
        Since hidden_state is None, we need to handle this case differently
        """
        # Return zeros for pi (policy) and v (value) when hidden_state is None
        if hidden_state is None:
            pi = np.zeros(self.action_size)
            v = 0
            return pi, v, hidden_state
        
        # Only proceed with prediction if we have a valid board state
        pi, v = self.predict(hidden_state)
        return pi, v, hidden_state
