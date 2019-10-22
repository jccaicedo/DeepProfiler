import json
import os
import random

import numpy as np
import pandas as pd
import pytest
import skimage.io
import tensorflow as tf

import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.metadata
import deepprofiler.dataset.target
import deepprofiler.imaging.cropping
import plugins.crop_generators.autoencoder_crop_generator


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


def test_autoencoder_crop_generator():
    assert issubclass(plugins.crop_generators.autoencoder_crop_generator.GeneratorClass,
                      deepprofiler.imaging.cropping.CropGenerator)
    assert issubclass(plugins.crop_generators.autoencoder_crop_generator.SingleImageGeneratorClass,
                      deepprofiler.imaging.cropping.SingleImageCropGenerator)


def test_generator_class_generate(config, dataset, out_dir):
    crop_generator = plugins.crop_generators.autoencoder_crop_generator.GeneratorClass(config, dataset)
    images = np.random.randint(0, 256, (128, 128, 36), dtype=np.uint8)
    for i in range(0, 36, 3):
        skimage.io.imsave(os.path.join(out_dir, crop_generator.dset.meta.data["R"][i // 3]), images[:, :, i])
        skimage.io.imsave(os.path.join(out_dir, crop_generator.dset.meta.data["G"][i // 3]), images[:, :, i + 1])
        skimage.io.imsave(os.path.join(out_dir, crop_generator.dset.meta.data["B"][i // 3]), images[:, :, i + 2])
    crop_generator.build_input_graph()
    crop_generator.build_augmentation_graph()
    sess = tf.Session()
    crop_generator.start(sess)
    generator = crop_generator.generate(sess)
    crop_generator.ready_to_sample = True
    test_steps = 3
    for i in range(test_steps):
        data = next(generator)
        np.testing.assert_array_equal(data[0], data[1])
    crop_generator.stop(sess)


def test_single_image_generator_class_generate(config, dataset, tmpdir):
    crop_generator = plugins.crop_generators.autoencoder_crop_generator.SingleImageGeneratorClass(config, dataset)
    image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    meta = crop_generator.dset.meta.data.iloc[0]
    tmpdir.mkdir(os.path.join("test", meta["Metadata_Plate"]))
    path = os.path.abspath(tmpdir.mkdir(os.path.join("test", meta["Metadata_Plate"], "locations")))
    path = os.path.join(path,
                        "{}-{}-{}.csv".format(meta["Metadata_Well"],
                                              meta["Metadata_Site"],
                                              crop_generator.config["train"]["sampling"]["locations_field"]))
    locations = pd.DataFrame({
        "R_Location_Center_X": np.random.randint(0, 128, (crop_generator.config["train"]["sampling"]["locations"])),
        "R_Location_Center_Y": np.random.randint(0, 128, (crop_generator.config["train"]["sampling"]["locations"]))
    })
    locations.to_csv(path, index=False)
    assert os.path.exists(path)
    sess = tf.Session()
    crop_generator.start(sess)
    num_crops = crop_generator.prepare_image(sess, image, meta)
    for i, item in enumerate(crop_generator.generate(sess)):
        np.testing.assert_array_equal(item[0], item[1])
        assert i == 0
