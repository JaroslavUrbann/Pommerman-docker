from feature_engineer import FeatureEngineer
from network import Network
import tensorflow as tf
import numpy as np
import time
from threading import Thread


class Agent:

    def _load_models(self):
        self.N = [Network("n_60.h5"), Network("n_180.h5"), Network("n_800.h5")]
        self.CM = tf.keras.models.load_model("cn.h5", compile=False)
        self.CM._make_predict_function()
        self.FE = FeatureEngineer(self.CM)
        # my message , teammates message
        self.MSGS = [(0, 0), (0, 0)]

    def __init__(self):
        self.t = Thread(target=self._load_models)
        self.t.start()

    def init_agent(self, id_, game_type):
        if self.t.isAlive():
            time.sleep(0.1)

    def act(self, observation, action_space):
        if self.t.isAlive():
            time.sleep(0.05)
            return 0, 0, 0
        tim = time.time()
        self.MSGS[1] = observation["message"] if len(observation["message"]) > 1 else observation["message"][0]
        features = self.FE.get_features(observation, self.MSGS)
        n = 0 if observation["step_count"] < 60 else 1 if observation["step_count"] < 180 else 2
        a, m = self.N[n].predict(features, observation["position"])
        self.MSGS[0] = m
        return int(a), int(m[0]), int(m[1])

    def episode_end(self, reward):
        self.FE = FeatureEngineer(self.CM)

    def shutdown(self):
        pass
