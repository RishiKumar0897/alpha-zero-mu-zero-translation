import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim
from muzeroimplementation.TicTacToeNNet import TicTacToeNNet as tnnet

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
        self.nnet = tnnet(game)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        if args.cuda:
            self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v, hidden_state)
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs, _ = list(zip(*[examples[i] for i in sample_ids]))
                
                # Convert to tensors and ensure they require gradients
                boards = torch.FloatTensor(np.array(boards).astype(np.float64)).requires_grad_(True)
                target_pis = torch.FloatTensor(np.array(pis)).requires_grad_(True)
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64)).requires_grad_(True)

                if args.cuda:
                    boards, target_pis, target_vs = boards.cuda(), target_pis.cuda(), target_vs.cuda()

                # Rest of the training loop remains the same
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            
    def predict(self, board):
        board = torch.FloatTensor(board.astype(np.float64))
        if args.cuda:
            board = board.cuda()
        board = board.view(1, self.board_x, self.board_y)

        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)
        return torch.exp(pi).cpu().numpy()[0], v.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save({'state_dict': self.nnet.state_dict()}, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model in path {filepath}")
        checkpoint = torch.load(filepath, map_location='cuda' if args.cuda else 'cpu')
        self.nnet.load_state_dict(checkpoint['state_dict'])

    def initial_inference(self, board):
        """Initial inference for root node."""
        board = torch.FloatTensor(board).unsqueeze(0)
        if args.cuda:
            board = board.cuda()
        self.nnet.eval()
        with torch.no_grad():
            policy, value, hidden_state = self.nnet.initial_inference(board)
        return policy, value.item(), hidden_state

    def recurrent_inference(self, hidden_state, action):
        """Recurrent inference for subsequent nodes."""
        action = torch.LongTensor([action])
        if args.cuda:
            hidden_state, action = hidden_state.cuda(), action.cuda()
        self.nnet.eval()
        with torch.no_grad():
            policy, value, reward, new_hidden_state = self.nnet.recurrent_inference(hidden_state, action)
        return policy.cpu().numpy(), value.item(), reward.item(), new_hidden_state
