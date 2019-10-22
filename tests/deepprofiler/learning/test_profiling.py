import importlib
import json
import os
import random

import keras
import numpy as np
import pandas as pd
import pytest
import skimage.io
import tensorflow as tf

import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.metadata
import deepprofiler.dataset.target
import deepprofiler.learning.profiling


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
def metadata(out_dir, make_struct):
    filename = os.path.join(out_dir, "index.csv")
    df = pd.DataFrame({
        "Metadata_Plate": __rand_array(),
        "Metadata_Well": __rand_array(),
        "Metadata_Site": __rand_array(),
        "R": [str(x) + ".png" for x in __rand_array()],
        "G": [str(x) + ".png" for x in __rand_array()],
        "B": [str(x) + ".png" for x in __rand_array()],
        "Class": ["0", "1", "2", "3", "0", "1", "2", "3", "0", "1", "2", "3"],
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
def dataset(metadata, out_dir, config, make_struct):
    keygen = lambda r: "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
    dset = deepprofiler.dataset.image_dataset.ImageDataset(metadata, "Sampling", ["R", "G", "B"],
                                                           config["paths"]["root_dir"], keygen)
    target = deepprofiler.dataset.target.MetadataColumnTarget("Class", metadata.data["Class"].unique())
    dset.add_target(target)
    return dset


@pytest.fixture(scope="function")
def data(metadata, out_dir, config, make_struct):
    images = np.random.randint(0, 256, (128, 128, 36), dtype=np.uint8)
    for i in range(0, 36, 3):
        skimage.io.imsave(os.path.join(config["paths"]["root_dir"], metadata.data["R"][i // 3]), images[:, :, i])
        skimage.io.imsave(os.path.join(config["paths"]["root_dir"], metadata.data["G"][i // 3]), images[:, :, i + 1])
        skimage.io.imsave(os.path.join(config["paths"]["root_dir"], metadata.data["B"][i // 3]), images[:, :, i + 2])


@pytest.fixture(scope="function")
def locations(out_dir, metadata, config, make_struct):
    for i in range(len(metadata.data.index)):
        meta = metadata.data.iloc[i]
        path = os.path.abspath(os.path.join(config["paths"]["locations"], meta["Metadata_Plate"]))
        os.makedirs(path, exist_ok=True)
        path = os.path.abspath(os.path.join(path, "{}-{}-{}.csv".format(meta["Metadata_Well"],
                                                                        meta["Metadata_Site"],
                                                                        config["train"]["sampling"][
                                                                            "locations_field"])))
        locs = pd.DataFrame({
            "R_Location_Center_X": np.random.randint(0, 128, (config["train"]["sampling"]["locations"])),
            "R_Location_Center_Y": np.random.randint(0, 128, (config["train"]["sampling"]["locations"]))
        })
        locs.to_csv(path, index=False)


@pytest.fixture(scope="function")
def checkpoint(config, dataset):
    crop_generator = importlib.import_module(
        "plugins.crop_generators.{}".format(config["train"]["model"]["crop_generator"])) \
        .GeneratorClass
    profile_crop_generator = importlib.import_module(
        "plugins.crop_generators.{}".format(config["train"]["model"]["crop_generator"])) \
        .SingleImageGeneratorClass
    dpmodel = importlib.import_module("plugins.models.{}".format(config["train"]["model"]["name"])) \
        .ModelClass(config, dataset, crop_generator, profile_crop_generator)
    dpmodel.feature_model.compile(dpmodel.optimizer, dpmodel.loss)
    filename = os.path.join(config["paths"]["checkpoints"], config["profile"]["checkpoint"])
    dpmodel.feature_model.save_weights(filename)
    return filename


@pytest.fixture(scope="function")
def profile(config, dataset):
    return deepprofiler.learning.profiling.Profile(config, dataset)


def test_init(config, dataset):
    prof = deepprofiler.learning.profiling.Profile(config, dataset)
    test_num_channels = len(config["dataset"]["images"]["channels"])
    assert prof.config == config
    assert prof.dset == dataset
    assert prof.num_channels == test_num_channels
    assert prof.crop_generator == importlib.import_module(
        "plugins.crop_generators.{}".format(config["train"]["model"]["crop_generator"])).GeneratorClass
    assert isinstance(prof.profile_crop_generator, importlib.import_module(
        "plugins.crop_generators.{}".format(config["train"]["model"]["crop_generator"])).SingleImageGeneratorClass)
    assert isinstance(prof.dpmodel,
                      importlib.import_module("plugins.models.{}".format(config["train"]["model"]["name"])).ModelClass)


def test_configure(profile, checkpoint):
    profile.configure()
    assert isinstance(profile.feat_extractor, keras.Model)
    assert isinstance(profile.sess, tf.Session)


def test_check(profile, metadata):
    assert profile.check(metadata.data)  # TODO: test false positive


def test_extract_features(profile, metadata, locations, checkpoint):
    meta = metadata.data.iloc[0]
    image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    profile.configure()
    profile.extract_features(None, image, meta)
    output_file = profile.config["paths"]["features"] + "/{}_{}_{}.npz" \
        .format(meta["Metadata_Plate"], meta["Metadata_Well"], meta["Metadata_Site"])
    assert os.path.isfile(output_file)


def test_profile(config, dataset, data, locations, checkpoint):
    deepprofiler.learning.profiling.profile(config, dataset)
    for index, row in dataset.meta.data.iterrows():
        output_file = config["paths"]["features"] + "/{}_{}_{}.npz" \
            .format(row["Metadata_Plate"], row["Metadata_Well"], row["Metadata_Site"])
        assert os.path.isfile(output_file)
