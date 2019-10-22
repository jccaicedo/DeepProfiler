import json
import os
import random

import numpy as np
import pandas as pd
import pytest
import skimage.io

import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.metadata
import deepprofiler.dataset.target


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
def metadata(out_dir, make_struct, config):
    filename = os.path.join(config["paths"]["metadata"], "index.csv")
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
def dataset(metadata, config, make_struct):
    keygen = lambda r: "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
    return deepprofiler.dataset.image_dataset.ImageDataset(metadata, "Sampling", ["R", "G", "B"],
                                                           config["paths"]["root_dir"], keygen)


def test_init(metadata, out_dir, dataset, config, make_struct):
    sampling_field = config["train"]["sampling"]["field"]
    channels = config["dataset"]["images"]["channels"]
    keygen = lambda r: "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
    dset = deepprofiler.dataset.image_dataset.ImageDataset(metadata, sampling_field, channels, out_dir, keygen)
    assert dset.meta == metadata
    assert dset.sampling_field == sampling_field
    np.testing.assert_array_equal(dset.sampling_values, metadata.data["Sampling"].unique())
    assert dset.channels == channels
    assert dset.root == out_dir
    assert dset.keyGen == keygen


def test_get_image_paths(metadata, out_dir, dataset, config, make_struct):
    for idx, row in dataset.meta.data.iterrows():
        key, image, outlines = dataset.get_image_paths(row)
        testKey = dataset.keyGen(row)
        testImage = [dataset.root + "/" + row[ch] for ch in dataset.channels]
        testOutlines = dataset.outlines
        assert key == testKey
        assert image == testImage
        assert outlines == testOutlines


def test_sample_images(metadata, out_dir, dataset, config, make_struct):
    n = 3
    keys, images, targets, outlines = dataset.sample_images(dataset.sampling_values, n)
    print(keys, images, targets, outlines)
    assert len(keys) == 2 * n
    assert len(images) == 2 * n
    assert len(targets) == 2 * n
    assert len(outlines) == 2 * n


def test_get_train_batch(metadata, out_dir, dataset, config, make_struct):
    images = np.random.randint(0, 256, (128, 128, 36), dtype=np.uint8)
    for i in range(0, 36, 3):
        skimage.io.imsave(os.path.join(out_dir, dataset.meta.data["R"][i // 3]), images[:, :, i])
        skimage.io.imsave(os.path.join(out_dir, dataset.meta.data["G"][i // 3]), images[:, :, i + 1])
        skimage.io.imsave(os.path.join(out_dir, dataset.meta.data["B"][i // 3]), images[:, :, i + 2])
    batch_size = 3
    batch = dataset.get_train_batch(batch_size)
    assert len(batch) == batch_size
    for image in batch["images"]:
        assert image.shape == (128, 128, 3)
        for i in range(3):
            assert image[:, :, i] in np.rollaxis(images, -1)


def test_scan(metadata, out_dir, dataset, config, make_struct):
    images = np.random.randint(0, 256, (128, 128, 36), dtype=np.uint8)
    for i in range(0, 36, 3):
        skimage.io.imsave(os.path.join(out_dir, dataset.meta.data["R"][i // 3]), images[:, :, i])
        skimage.io.imsave(os.path.join(out_dir, dataset.meta.data["G"][i // 3]), images[:, :, i + 1])
        skimage.io.imsave(os.path.join(out_dir, dataset.meta.data["B"][i // 3]), images[:, :, i + 2])
    data = {"index": [], "image": [], "meta": []}

    def func(index, image, meta):
        data["index"].append(index)
        data["image"].append(image)
        data["meta"].append(meta)

    dataset.scan(func, frame="all")
    for index in data["index"]:
        assert index in range(12)
    for image in data["image"]:
        assert image.shape == (128, 128, 3)
        for i in range(3):
            assert image[:, :, i] in np.rollaxis(images, -1)
    for meta in data["meta"]:
        assert (dataset.meta.data == meta).all(1).any()


def test_number_of_records(metadata, out_dir, dataset, config, make_struct):
    assert dataset.number_of_records("all") == len(dataset.meta.data)
    assert dataset.number_of_records("val") == len(dataset.meta.val)
    assert dataset.number_of_records("train") == len(dataset.meta.train)
    assert dataset.number_of_records("other") == 0


def test_add_target(metadata, out_dir, dataset, config, make_struct):
    target = deepprofiler.dataset.target.MetadataColumnTarget("Target", random.sample(range(100), 12))
    dataset.add_target(target)
    assert target in dataset.targets


def test_read_dataset(metadata, out_dir, dataset, config, make_struct):
    dset = deepprofiler.dataset.image_dataset.read_dataset(config)
    pd.testing.assert_frame_equal(dset.meta.data,
                                  deepprofiler.dataset.metadata.Metadata(config["paths"]["index"], dtype=None).data)
    assert dset.channels == config["dataset"]["images"]["channels"]
    assert dset.root == config["paths"]["images"]
    assert dset.sampling_field == config["train"]["sampling"]["field"]
    np.testing.assert_array_equal(dset.sampling_values, dset.meta.data[dset.sampling_field].unique())
