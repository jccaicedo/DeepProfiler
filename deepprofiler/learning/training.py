import importlib


# MAIN TRAINING ROUTINE


def learn_model(config, dset, epoch=1, seed=None, verbose=1):
    model_module = importlib.import_module("plugins.models.{}".format(config["train"]["model"]["name"]))
    crop_module = importlib.import_module(
        "plugins.crop_generators.{}".format(config["train"]["model"]["crop_generator"]))
    if "metrics" in config["train"]["model"].keys():
        if type(config["train"]["model"]["metrics"]) not in [list, dict]:
            raise ValueError("Metrics should be a list or dictionary.")
        keras_metrics = [
            "accuracy",
            "binary_accuracy",
            "categorical_accuracy",
            "sparse_categorical_accuracy",
            "top_k_categorical_accuracy",
            "sparse_top_k_categorical_accuracy"
        ]
        if type(config["train"]["model"]["metrics"] is list):
            metrics = list(map(lambda metric: importlib.import_module(
                "plugins.metrics.{}".format(metric)).MetricClass(config,
                                                                 metric).f if metric not in keras_metrics else metric,
                               config["train"]["model"]["metrics"]))
        elif type(config["train"]["model"]["metrics"] is dict):
            metrics = {k: lambda metric: importlib.import_module(
                "plugins.metrics.{}".format(metric)).MetricClass(config,
                                                                 metric).f if metric not in keras_metrics else metric
                       for k, v in config["train"]["model"]["metrics"].items()}
        else:
            metrics = None
    else:
        metrics = ["accuracy"]
    importlib.invalidate_caches()

    crop_generator = crop_module.GeneratorClass
    val_crop_generator = crop_module.SingleImageGeneratorClass
    model = model_module.ModelClass(config, dset, crop_generator, val_crop_generator)

    if seed is not None:
        model.seed(seed)
    if verbose == 0:
        if metrics:
            trained_model, x_validation, y_validation = model.train(epoch, metrics, verbose=verbose)

            evaluation = trained_model.evaluate(x_validation, y_validation, batch_size=config["train"]["validation"]["batch_size"], verbose=verbose)

            return evaluation
    else:
        model.train(epoch, metrics, verbose=verbose)
        return None
