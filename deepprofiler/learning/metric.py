import abc


class Metric(abc.ABC):

    def __init__(self, config, name):
        self.config = config
        self.name = name
        self.create_metric()

    def create_metric(self):
        def metric_func(y_true, y_pred):
            return self.metric(y_true, y_pred)

        metric_func.__name__ = self.name
        self.f = metric_func

    @abc.abstractmethod
    def metric(self, y_true, y_pred):
        pass
