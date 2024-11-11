import torch
import torch.nn as nn
import torch.nn.functional as F


class TicTacToeNNet(nn.Module):
    def __init__(self, game, args):
        super(TicTacToeNNet, self).__init__()
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Representation (Initial Inference)
        self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(args.num_channels)

        # Dynamics (Recurrent Inference)
        self.conv2 = nn.Conv2d(args.num_channels + 1, args.num_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(args.num_channels)

        # Fully connected layers for hidden state -> Policy and Value
        self.fc1 = nn.Linear(self.board_x * self.board_y * args.num_channels, 512)
        self.fc_policy = nn.Linear(512, self.action_size)
        self.fc_value = nn.Linear(512, 1)

        # Fully connected for hidden state -> Next hidden state and reward
        self.fc_hidden = nn.Linear(self.board_x * self.board_y * args.num_channels, 512)
        self.fc_reward = nn.Linear(512, 1)

    def initial_inference(self, board):
        """Initial inference step."""
        s = board.view(-1, 1, self.board_x, self.board_y)
        s = F.relu(self.bn1(self.conv1(s)))
        hidden_state = s.view(s.size(0), -1)
        policy = self.fc_policy(F.relu(self.fc1(hidden_state)))
        value = self.fc_value(F.relu(self.fc1(hidden_state)))
        return policy, value, hidden_state

    def recurrent_inference(self, hidden_state, action):
        """Recurrent inference step."""
        action_one_hot = F.one_hot(action, num_classes=self.action_size).float()
        action_one_hot = action_one_hot.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.board_x, self.board_y)
        x = torch.cat((hidden_state.view(-1, self.args.num_channels, self.board_x, self.board_y), action_one_hot), dim=1)
        x = F.relu(self.bn2(self.conv2(x)))

        new_hidden_state = self.fc_hidden(x.view(x.size(0), -1))
        reward = self.fc_reward(F.relu(new_hidden_state))
        policy = self.fc_policy(F.relu(self.fc1(new_hidden_state)))
        value = self.fc_value(F.relu(self.fc1(new_hidden_state)))
        return policy, value, reward, new_hidden_state
