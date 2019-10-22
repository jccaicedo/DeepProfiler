import json
import os
import random

import keras
import numpy as np
import pandas as pd
import pytest
import skimage.io

import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.metadata
import deepprofiler.dataset.target
import deepprofiler.imaging.cropping
import plugins.models.gan


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
def generator():
    return deepprofiler.imaging.cropping.CropGenerator


@pytest.fixture(scope="function")
def val_generator():
    return deepprofiler.imaging.cropping.SingleImageCropGenerator


@pytest.fixture(scope="function")
def model(config, dataset, generator, val_generator):
    def create():
        return plugins.models.gan.ModelClass(config, dataset, generator, val_generator)

    return create


def test_gan(config, generator, val_generator):
    gan = plugins.models.gan.GAN(config, generator, val_generator)
    assert gan.config == config
    assert gan.crop_generator == generator
    assert gan.val_crop_generator == val_generator
    assert gan.img_cols == config["train"]["sampling"]["box_size"]
    assert gan.img_rows == config["train"]["sampling"]["box_size"]
    assert gan.channels == len(config["dataset"]["images"]["channels"])
    assert gan.img_shape == (
        config["train"]["sampling"]["box_size"],
        config["train"]["sampling"]["box_size"],
        len(config["dataset"]["images"]["channels"])
    )
    assert gan.latent_dim == config["train"]["model"]["params"]["latent_dim"]
    assert isinstance(gan.generator, keras.Model)
    assert isinstance(gan.discriminator, keras.Model)
    assert isinstance(gan.discriminator_fixed, keras.Model)
    assert isinstance(gan.combined, keras.Model)
    assert gan.generator in gan.combined.layers
    assert gan.discriminator not in gan.combined.layers
    assert gan.discriminator_fixed in gan.combined.layers
    assert gan.generator.trainable
    assert gan.discriminator.trainable
    assert not gan.discriminator_fixed.trainable


def test_init(config, dataset, generator, val_generator):
    dpmodel = plugins.models.gan.ModelClass(config, dataset, generator, val_generator)
    gan = plugins.models.gan.GAN(config, generator, val_generator)
    assert dpmodel.gan.__eq__(gan)
    assert isinstance(dpmodel.feature_model, keras.Model)


def test_train(model, out_dir, data, locations, config, make_struct):
    model1 = model()
    model1.train()
    assert os.path.exists(os.path.join(config["paths"]["checkpoints"], "discriminator_epoch_0001.hdf5"))
    assert os.path.exists(os.path.join(config["paths"]["checkpoints"], "generator_epoch_0001.hdf5"))
    assert os.path.exists(os.path.join(config["paths"]["checkpoints"], "discriminator_epoch_0002.hdf5"))
    assert os.path.exists(os.path.join(config["paths"]["checkpoints"], "generator_epoch_0002.hdf5"))
    model2 = model()
    epoch = 3
    model2.config["train"]["model"]["epochs"] = 4
    model2.train(epoch)
    assert os.path.exists(os.path.join(config["paths"]["checkpoints"], "discriminator_epoch_0003.hdf5"))
    assert os.path.exists(os.path.join(config["paths"]["checkpoints"], "generator_epoch_0003.hdf5"))
    assert os.path.exists(os.path.join(config["paths"]["checkpoints"], "discriminator_epoch_0004.hdf5"))
    assert os.path.exists(os.path.join(config["paths"]["checkpoints"], "generator_epoch_0004.hdf5"))
