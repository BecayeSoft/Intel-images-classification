import numpy as np

from sklearn.metrics import cohen_kappa_score
from keras.callbacks import Callback

class KappaScore(Callback):
    def __init__(self, val_data):
        super(KappaScore, self).__init__()
        self.x_val = val_data[0]
        self.y_val = val_data[1]
        self.kappa_scores = []

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x_val)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(self.y_val, axis=1)
        score = cohen_kappa_score(y_true, y_pred)
        self.kappa_scores.append(score)
        print(f"Kappa Score: {score}")
        return
