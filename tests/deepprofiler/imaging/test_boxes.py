import json
import os
import random

import numpy as np
import pandas as pd
import pytest
import skimage.io

import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.metadata
import deepprofiler.imaging.boxes


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
        "Split": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        "Target": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
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
    dset = deepprofiler.dataset.image_dataset.ImageDataset(metadata, "Sampling", ["R", "G", "B"],
                                                           config["paths"]["root_dir"], keygen)
    target = deepprofiler.dataset.target.MetadataColumnTarget("Target", metadata.data["Target"].unique())
    dset.add_target(target)
    return dset


@pytest.fixture(scope="function")
def loadbatch(dataset, metadata, out_dir, config, make_struct):
    images = np.random.randint(0, 256, (128, 128, 36), dtype=np.uint8)
    for i in range(0, 36, 3):
        skimage.io.imsave(os.path.join(out_dir, dataset.meta.data["R"][i // 3]), images[:, :, i])
        skimage.io.imsave(os.path.join(out_dir, dataset.meta.data["G"][i // 3]), images[:, :, i + 1])
        skimage.io.imsave(os.path.join(out_dir, dataset.meta.data["B"][i // 3]), images[:, :, i + 2])
    result = deepprofiler.imaging.boxes.load_batch(dataset, config)
    return result


def test_get_locations(config, make_struct):
    test_image_key = "dog/cat"
    test_output = deepprofiler.imaging.boxes.get_locations(test_image_key, config)
    expected_output = pd.DataFrame(columns=["R_Location_Center_X", "R_Location_Center_Y"])
    assert test_output.equals(expected_output)

    test_locations_path = os.path.abspath(os.path.join(config["paths"]["locations"], "dog"))
    os.makedirs(test_locations_path)
    test_file_name = "cat-R.csv"
    test_locations_path = os.path.join(test_locations_path, test_file_name)
    expected_output = pd.DataFrame(columns=["R_Location_Center_X", "R_Location_Center_Y"])
    expected_output.to_csv(test_locations_path)
    expected_output = pd.read_csv(test_locations_path)
    assert os.path.exists(test_locations_path) == True
    test_output = deepprofiler.imaging.boxes.get_locations(test_image_key, config)
    assert test_output.equals(expected_output)

    expected_output = pd.DataFrame(index=range(10), columns=["R_Location_Center_X", "R_Location_Center_Y"])
    expected_output.to_csv(test_locations_path, mode="w")
    expected_output = pd.read_csv(test_locations_path)
    test_output = deepprofiler.imaging.boxes.get_locations(test_image_key, config)
    assert test_output.equals(expected_output)

    expected_output = pd.DataFrame(index=range(60), columns=["R_Location_Center_X", "R_Location_Center_Y"])
    expected_output.to_csv(test_locations_path, mode="w")
    expected_output = pd.read_csv(test_locations_path)
    expected_output = expected_output.sample(n=10, random_state=1414)
    test_output = deepprofiler.imaging.boxes.get_locations(test_image_key, config, randomize=True, seed=1414)
    assert test_output.equals(expected_output)


def test_load_batch(loadbatch):
    test_batch = loadbatch
    expected_batch_locations = 12 * [pd.DataFrame(columns=["R_Location_Center_X", "R_Location_Center_Y"])]
    for i in range(12):
        assert test_batch["locations"][i].equals(expected_batch_locations[i])


def test_prepare_boxes(config):
    test_batch = {"images": [np.random.randint(256, size=(64, 64), dtype=np.uint16)], "targets": [[1]],
                  "locations": [pd.DataFrame(data=[[32, 32]], columns=["R_Location_Center_X", "R_Location_Center_Y"])]}
    test_result = deepprofiler.imaging.boxes.prepare_boxes(test_batch, config)
    assert np.array(test_result[0]).shape == (1, 4)
    assert np.array(test_result[1]).shape == (1,)
    assert np.array(test_result[2]).shape == (1, 1)
    # ignores masks for testing
