from feature_engineer import FeatureEngineer
from network import Network
import tensorflow as tf


class Agent:
    def __init__(self):
        self.N = Network("model.hdf5")
        self.CM = tf.keras.models.load_model("chat_model.hdf5")
        self.FE = FeatureEngineer(self.CM)
        # my message , teammates message
        self.MSGS = [(0, 0), (0, 0)]

    def init_agent(self, id_, game_type):
        pass

    def act(self, observation, action_space):
        self.MSGS[1] = observation["message"] if len(observation["message"]) > 1 else observation["message"][0]
        features = self.FE.get_features(observation, self.MSGS)
        a, m = self.N.predict(features, observation["position"])
        self.MSGS[0] = m
        return int(a), int(m[0]), int(m[1])

    def episode_end(self, reward):
        self.FE = FeatureEngineer(self.CM)

    def shutdown(self):
        pass
