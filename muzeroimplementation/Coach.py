import logging
import os
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
import numpy as np
from tqdm import tqdm
from muzeroimplementation import MCTS

log = logging.getLogger(__name__)

class Coach:
    """Handles self-play, training, and evaluation for MuZero."""

    def __init__(self, game, network, args):
        self.game = game
        self.network = network
        self.args = args
        self.mcts = MCTS(game, network, args)
        self.trainExamplesHistory = []  # History of training examples
        self.skipFirstSelfPlay = False  # Override in loadTrainExamples()

    def executeEpisode(self):
        """Runs one episode of self-play, storing training examples."""
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0
        hidden_state = None

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None, hidden_state])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)
            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer)), x[4]) for x in trainExamples]

    def learn(self):
        """Runs the self-play and training process."""
        for i in range(1, self.args.numIters + 1):
            log.info(f"Starting Iter #{i} ...")
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.network, self.args)
                    iterationTrainExamples += self.executeEpisode()

                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning("Removing oldest entry from trainExamplesHistory.")
                self.trainExamplesHistory.pop(0)

            self.saveTrainExamples(i - 1)
            trainExamples = [e for ex in self.trainExamplesHistory for e in ex]
            shuffle(trainExamples)

            self.network.train(trainExamples)
            self.network.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))

    def getCheckpointFile(self, iteration):
        return f"checkpoint_{iteration}.pth.tar"

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
        else:
            log.info("Loading trainExamples from file...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info("Loading done!")
            self.skipFirstSelfPlay = True
