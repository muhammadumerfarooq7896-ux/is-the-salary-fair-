from agents.agent import Agent
from agents.deep_neural_network import DeepNeuralNetworkInference


class NeuralNetworkAgent(Agent):
    name = "Neural Network Agent"
    color = Agent.MAGENTA

    def __init__(self):
        """
        Initialize this object by loading in the saved model weights
        and the SentenceTransformer vector encoding model
        """
        self.log("Neural Network Agent is initializing")
        self.neural_network = DeepNeuralNetworkInference()
        self.neural_network.setup()
        self.neural_network.load("deep_neural_network.pth")
        self.log("Neural Network Agent is ready and weights are loaded")

    def salary(self, description):
        """Use the deep neural network to make a prediction of salary"""
        self.log("Neural Network Agent is making a prediction")
        result = self.neural_network.inference(description)
        self.log(f"Neural Network Agent completed - predicting ${result:.2f}")
        return result
