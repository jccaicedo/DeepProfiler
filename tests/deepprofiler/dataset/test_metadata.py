import os
import random

import numpy as np
import pandas as pd
import pytest

import deepprofiler.dataset.metadata


def __rand_array():
    return np.array(random.sample(range(100), 10))


@pytest.fixture(scope="function")
def out_dir(tmpdir):
    return os.path.abspath(tmpdir.mkdir("metadata"))


@pytest.fixture(scope="function")
def dataframe(out_dir):
    df = pd.DataFrame({
        "Metadata_Plate": __rand_array(),
        "Metadata_Well": __rand_array(),
        "Metadata_Site": __rand_array()
    }, dtype=int)
    df.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    return df


@pytest.fixture(scope="function")
def metadata():
    return deepprofiler.dataset.metadata.Metadata()


def test_parse_delimiter():
    assert deepprofiler.dataset.metadata.parse_delimiter("blanks") == "\s+"
    assert deepprofiler.dataset.metadata.parse_delimiter("tabs") == "\t"
    assert deepprofiler.dataset.metadata.parse_delimiter("default") == ","


def test_read_plates(out_dir):
    filename = os.path.join(out_dir, "test.csv")
    df = pd.DataFrame({
        "Metadata_Plate": [0, 1, 1, 2, 2, 2, 3, 3, 3, 3],
        "Metadata_Well": __rand_array(),
        "Metadata_Site": __rand_array()
    }, dtype=int)
    df.to_csv(filename, index=False)
    generator = deepprofiler.dataset.metadata.read_plates(filename)
    plates = 0
    total = 0
    for item in generator:
        assert len(set(item.data["Metadata_Plate"])) == 1
        plates += 1
        total += len(item.data)
    assert plates == 4
    assert total == len(df)


def test_load_single(metadata, dataframe, out_dir):
    filename = os.path.join(out_dir, "test.csv")
    assert os.path.exists(filename)
    metadata.load_single(filename, "default", int)
    pd.testing.assert_frame_equal(metadata.data, dataframe)


def test_load_multiple(metadata, dataframe, out_dir):
    csv1 = os.path.join(out_dir, "test.csv")
    csv2 = os.path.join(out_dir, "test2.csv")
    dataframe2 = pd.DataFrame({
        "Metadata_Plate": __rand_array(),
        "Metadata_Well": __rand_array(),
        "Metadata_Site": __rand_array()
    }, dtype=int)
    dataframe2.to_csv(os.path.join(out_dir, "test2.csv"), index=False)
    with open(os.path.join(out_dir, "filelist.txt"), "w") as filelist:
        filelist.write(csv1 + "\n")
        filelist.write(csv2 + "\n")
    filelist = os.path.join(out_dir, "filelist.txt")
    assert os.path.exists(csv1)
    assert os.path.exists(csv2)
    assert os.path.exists(filelist)
    metadata.load_multiple(filelist, "default", int)
    concat = pd.concat([dataframe, dataframe2])
    pd.testing.assert_frame_equal(metadata.data, concat)


def test_filter_records(metadata, dataframe, out_dir):
    metadata.load_single(os.path.join(out_dir, "test.csv"), "default", int)
    rule = lambda data: map(lambda row: any(row % 2 == 0), data.values)
    metadata.filter_records(rule)
    filtered = dataframe.loc[rule(dataframe), :]
    pd.testing.assert_frame_equal(metadata.data, filtered)


def test_split_metadata(metadata, dataframe, out_dir):
    metadata.load_single(os.path.join(out_dir, "test.csv"), "default", int)
    train_rule = lambda data: data["Metadata_Plate"] < 50
    val_rule = lambda data: data["Metadata_Plate"] >= 50
    metadata.split_metadata(train_rule, val_rule)
    assert len(metadata.train) + len(metadata.val) == len(metadata.data)


def test_merge_outlines(metadata, dataframe, out_dir):
    metadata.load_single(os.path.join(out_dir, "test.csv"), "default", int)
    outlines = pd.DataFrame({
        "Metadata_Plate": __rand_array(),
        "Metadata_Well": __rand_array(),
        "Metadata_Site": __rand_array()
    }, dtype=int)
    metadata.merge_outlines(outlines)
    merged = pd.merge(metadata.data, outlines, on=["Metadata_Plate", "Metadata_Well", "Metadata_Site"])
    pd.testing.assert_frame_equal(metadata.data, merged)
