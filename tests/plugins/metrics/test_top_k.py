import inspect
import json
import os

import numpy as np
import pytest
import tensorflow as tf

import plugins.metrics.top_k

cpu_config = tf.ConfigProto(
    device_count={'GPU': 0}
)


def is_method(obj, name):
    return hasattr(obj, name) and inspect.ismethod(getattr(obj, name))


@pytest.fixture(scope="function")
def out_dir(tmpdir):
    return os.path.abspath(tmpdir.mkdir("test"))


@pytest.fixture(scope="function")
def config(out_dir):
    with open("tests/files/config/test.json", "r") as f:
        config = json.load(f)
    for path in config["paths"]:
        config["paths"][path] = out_dir + config["paths"].get(path)
    config["paths"]["root_dir"] = out_dir
    return config


@pytest.fixture(scope="function")
def make_struct(config):
    for key, path in config["paths"].items():
        if key not in ["index", "config_file", "root_dir"]:
            os.makedirs(path + "/")
    return


def test_create_metric(config, make_struct):
    name = "Dog"
    metric = plugins.metrics.top_k.MetricClass(config, name)
    expected_name = "top_" + str(config["train"]["validation"]["top_k"])
    assert is_method(metric, "create_metric")
    assert metric.f.__name__ == expected_name


def test_metric(config, make_struct):
    with tf.Session(config=cpu_config) as sess:
        name = "Dog"
        metric = plugins.metrics.top_k.MetricClass(config, name)
        y_true = np.array([[0, 1], [1, 0]])
        y_pred = np.array([[0, 1], [0, 1]])
        expected_output = 0.5
        output = metric.metric(y_true, y_pred).eval()
        print(output)
        assert output == expected_output
        assert is_method(metric, "metric")
