import inspect
import json
import os

import pytest

import deepprofiler.learning.metric


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


class MetricClass(deepprofiler.learning.metric.Metric):
    def metric(self, y_true, y_pred):
        pass


def test_init(config, out_dir, make_struct):
    test_name = "Dog"
    test_metric = MetricClass(config, test_name)
    assert is_method(test_metric, "__init__")
    assert test_metric.config == config
    assert test_metric.name == test_name


def test_create_metric(config, out_dir, make_struct):
    test_name = "Dog"
    test_metric = MetricClass(config, test_name)
    assert is_method(test_metric, "create_metric")
    assert test_metric.f.__name__ == test_name


def test_metric(config, out_dir, make_struct):
    test_name = "Dog"
    test_metric = MetricClass(config, test_name)
    assert is_method(test_metric, "metric")
