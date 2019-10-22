import json
import os
import random

import numpy as np
import pandas as pd
import pytest

import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.indexing
import deepprofiler.dataset.metadata


# import shutil


def __rand_array():
    return np.array(random.sample(range(100), 12))


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


@pytest.fixture(scope="function")
def metadata(config, make_struct):
    filename = os.path.join(config["paths"]["metadata"], "index.csv")
    df = pd.DataFrame({
        "Metadata_Plate": __rand_array(),
        "Metadata_Well": __rand_array(),
        "Metadata_Site": __rand_array(),
        "R": [str(x) + ".png" for x in __rand_array()],
        "G": [str(x) + ".png" for x in __rand_array()],
        "B": [str(x) + ".png" for x in __rand_array()],
        "Sampling": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "Split": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    }, dtype=int)
    df.to_csv(filename, index=False)
    meta = deepprofiler.dataset.metadata.Metadata(filename)
    train_rule = lambda data: data["Split"].astype(int) == 0
    val_rule = lambda data: data["Split"].astype(int) == 1
    meta.split_metadata(train_rule, val_rule)
    return meta


@pytest.fixture(scope="function")
def dataset(metadata, config, make_struct):
    keygen = lambda r: "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
    return deepprofiler.dataset.image_dataset.ImageDataset(metadata, "Sampling", ["R", "G", "B"],
                                                           config["paths"]["images"], keygen)


def test_write_compression_index(config, metadata, dataset, make_struct):
    deepprofiler.dataset.indexing.write_compression_index(config)
    test_output = pd.read_csv(config["paths"]["compressed_metadata"] + "/compressed.csv", index_col=0)
    assert test_output.shape == (12, 8)


def test_split_index(config, metadata, dataset):
    test_parts = 3
    test_paths = [config["paths"]["metadata"] + "/index-000.csv",
                  config["paths"]["metadata"] + "/index-001.csv",
                  config["paths"]["metadata"] + "/index-002.csv"]
    deepprofiler.dataset.indexing.write_compression_index(config)
    deepprofiler.dataset.indexing.split_index(config, test_parts)
    assert os.path.exists(test_paths[0]) == True
    assert os.path.exists(test_paths[1]) == True
    assert os.path.exists(test_paths[2]) == True
    test_outputs = [pd.read_csv(config["paths"]["metadata"] + "/index-000.csv"),
                    pd.read_csv(config["paths"]["metadata"] + "/index-001.csv"),
                    pd.read_csv(config["paths"]["metadata"] + "/index-002.csv")]
    print(test_outputs[0])
    print(test_outputs[1])
    print(test_outputs[2])
    assert test_outputs[0].shape == (4, 8)
    assert test_outputs[1].shape == (4, 8)
    assert test_outputs[2].shape == (4, 8)
