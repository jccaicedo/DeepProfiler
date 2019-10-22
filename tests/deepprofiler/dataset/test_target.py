from random import sample, shuffle

import pytest

import deepprofiler.dataset.target


@pytest.fixture(scope="function")
def values():
    return sample(range(100), 10)


@pytest.fixture(scope="function")
def target(values):
    field_name = "test"
    shuffle(values)
    return deepprofiler.dataset.target.MetadataColumnTarget(field_name, values)


def test_init(target, values):
    field_name = "test"
    shuffle(values)
    assert target.field_name == field_name
    assert len(target.index) == len(values)
    assert list(target.index) == sorted(values)


def test_get_values(target, values):
    record = {"test": values[0]}
    assert target.get_values(record) == 0


def test_shape(target, values):
    assert len(target.shape) == 2
    assert target.shape[1] == len(values)
