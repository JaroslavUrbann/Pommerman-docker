import tensorflow as tf
import numpy as np
import action_filter as AF


class Network:

    def __init__(self, file):
        self.model = tf.keras.models.load_model(file, compile=False)
        self.model._make_predict_function()

    def predict(self, features, position):
        actions, message = self.model.predict(features)

        action_filter = AF.get_action_filter(position[0], position[1], features)
        a = AF.apply_action_filter(action_filter, actions[0])
        if a is None:
            a = 0

        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        message = sigmoid(message[0])
        binary = "".join(str(min(1, int(m // 0.5))) for m in message)
        dec = int(binary, 2)
        msg = (dec // 8 + 1, dec % 8 + 1)
        return a, msg
