import torch
import torch.nn as nn
import torch.nn.functional as F

class Args:
    def __init__(self):
        self.num_channels = 128  # Number of channels for convolutional layers
        self.dropout = 0.3       # Dropout rate for fully connected layers
        self.lr = 0.001          # Learning rate for optimizer
        self.epochs = 10         # Number of training epochs
        self.batch_size = 64     # Training batch size
        self.cuda = torch.cuda.is_available()  # Use GPU if available


args = Args()


class TicTacToeNNet(nn.Module):
    def __init__(self, game):
        super(TicTacToeNNet, self).__init__()
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Representation (Initial Inference)
        # In AlphaZero, this would be the only network component
        self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(args.num_channels)

        # Dynamics (Recurrent Inference)
        # MuZero specific: Predicts next state and reward
        self.conv2 = nn.Conv2d(args.num_channels + 1, args.num_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(args.num_channels)

        # Fully connected layers for hidden state -> Policy and Value
        # Prediction Network
        # Similar to AlphaZero's heads but operates on hidden state
        self.fc1 = nn.Linear(self.board_x * self.board_y * args.num_channels, 512)
        self.fc_policy = nn.Linear(512, self.action_size)
        self.fc_value = nn.Linear(512, 1)

        # Fully connected for hidden state -> Next hidden state and reward
        # MuZero specific: Additional networks for state transitions
        self.fc_hidden = nn.Linear(self.board_x * self.board_y * args.num_channels, 512)
        self.fc_reward = nn.Linear(512, 1)

    def initial_inference(self, board):
        """Initial inference step."""
        with torch.no_grad():
            # Convert board to hidden state (representation network)
            s = board.view(-1, 1, self.board_x, self.board_y)
            s = F.relu(self.bn1(self.conv1(s)))
            hidden_state = s.view(s.size(0), -1)

            # Predict policy and value from hidden state (prediction network)
            policy = self.fc_policy(F.relu(self.fc1(hidden_state)))
            value = self.fc_value(F.relu(self.fc1(hidden_state)))
            return policy, value, hidden_state

    def recurrent_inference(self, hidden_state, action):
        """
        MuZero's dynamics + prediction networks
        Input: (hidden_state, action)
        Output: (next_policy, next_value, reward, next_hidden_state)
        AlphaZero has no equivalent - it uses actual game rules
        """
        
        # Reshape action to be a single value if it's not already
        action = action.view(-1)
        
        # Create one-hot encoding with correct shape
        action_one_hot = F.one_hot(action, num_classes=self.action_size).float()
        # Reshape to match the hidden state's spatial dimensions
        action_one_hot = action_one_hot.view(-1, self.action_size, 1, 1)
        # Expand to match spatial dimensions
        action_one_hot = action_one_hot.expand(-1, -1, self.board_x, self.board_y)
        
        # Ensure hidden state has the correct shape
        hidden_state = hidden_state.view(-1, self.args.num_channels, self.board_x, self.board_y)
        
        # Take only one channel from action_one_hot when concatenating
        x = torch.cat((hidden_state, action_one_hot[:, :1, :, :]), dim=1)

        x = F.relu(self.bn2(self.conv2(x)))
        x_flat = x.view(x.size(0), -1)  # Flatten the tensor properly
        
        new_hidden_state = self.fc_hidden(x_flat)
        reward = self.fc_reward(F.relu(new_hidden_state))
        policy = self.fc_policy(F.relu(self.fc1(x_flat)))  # Use x_flat instead of new_hidden_state
        value = self.fc_value(F.relu(self.fc1(x_flat)))    # Use x_flat instead of new_hidden_state
        return policy, value, reward, new_hidden_state
        
    def forward(self, board):
        policy, value, _ = self.initial_inference(board)
        return policy, value
