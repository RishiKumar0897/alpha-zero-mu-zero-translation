from abc import ABC, abstractmethod

class NeuralNet(ABC):
    @abstractmethod
    def initial_inference(self, board):
        pass

    @abstractmethod
    def recurrent_inference(self, hidden_state, action):
        pass 