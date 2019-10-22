import keras
from keras.metrics import top_k_categorical_accuracy

from deepprofiler.learning.metric import Metric


class MetricClass(Metric):

    def create_metric(self):
        def metric_func(y_true, y_pred):
            return self.metric(y_true, y_pred)

        metric_func.__name__ = "top_" + str(self.config["train"]["validation"]["top_k"])
        self.f = metric_func

    def metric(self, y_true, y_pred):
        top_k = top_k_categorical_accuracy(y_true, y_pred, k=self.config["train"]["validation"]["top_k"])
        return keras.backend.mean(top_k)
