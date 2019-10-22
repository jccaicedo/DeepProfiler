import glob
import os.path

import numpy
import numpy.random
import numpy.testing
import pytest
import skimage.io

import deepprofiler.dataset.compression


@pytest.fixture(scope="function")
def out_dir(tmpdir):
    return os.path.abspath(tmpdir.mkdir("images"))


@pytest.fixture(scope="function")
def compress(out_dir):
    stats = {"original_size": [16, 16]}
    channels = ["DNA", "ER", "Mito"]
    return deepprofiler.dataset.compression.Compress(stats, channels, out_dir)


def test_init(compress, out_dir):
    stats = {"original_size": [16, 16]}
    channels = ["DNA", "ER", "Mito"]
    control_distribution = numpy.zeros((3, 2 ** 8), dtype=numpy.float64)

    assert compress.stats == stats
    assert compress.channels == channels
    assert compress.out_dir == out_dir
    assert compress.count == 0
    assert compress.expected == 1
    assert not compress.metadata_control_filter("x")
    numpy.testing.assert_array_equal(compress.controls_distribution, control_distribution)
    assert compress.source_format == "tiff"
    assert compress.target_format == "png"
    assert compress.output_shape == [16, 16]


def test_recompute_percentile(compress):
    compress.stats["histogram"] = numpy.asarray([[1] * 100] * 3)

    compress.recompute_percentile(0.9, "upper_percentile")
    numpy.testing.assert_array_equal(compress.stats["upper_percentile"], numpy.asarray([89.0] * 3))
    compress.recompute_percentile(0.1, "lower_percentile")
    numpy.testing.assert_array_equal(compress.stats["lower_percentile"], numpy.asarray([10.0] * 3))


def test_set_control_samples_filter(compress):
    test_filter = lambda x: True
    control_distribution = numpy.zeros((3, 2 ** 8), dtype=numpy.float64)

    compress.set_control_samples_filter(test_filter)

    assert compress.metadata_control_filter(1)
    numpy.testing.assert_array_equal(compress.controls_distribution, control_distribution)


def test_set_formats(compress):
    compress.set_formats()
    assert compress.source_format == "tiff"
    assert compress.target_format == "png"
    compress.set_formats(source_format="tif")
    assert compress.source_format == "tif"
    assert compress.target_format == "png"


def test_set_scaling_factor(compress):
    compress.set_scaling_factor(1.0)
    assert compress.output_shape[0] == 16
    assert compress.output_shape[1] == 16
    compress.set_scaling_factor(0.5)
    assert compress.output_shape[0] == 8
    assert compress.output_shape[1] == 8


def test_target_path(compress, out_dir):
    new_path = compress.target_path("/tmp/image.tiff")
    assert new_path == out_dir + "/image.png"


def test_process_image(compress, out_dir):
    numpy.random.seed(8)
    image = numpy.random.randint(256, size=(16, 16, 3), dtype=numpy.uint16)

    meta = {
        "DNA": "/tmp/dna.tiff",
        "ER": "/tmp/er.tiff",
        "Mito": "/tmp/mito.tiff"
    }
    compress.stats["illum_correction_function"] = numpy.ones((16, 16, 3))
    compress.stats["upper_percentiles"] = [255, 255, 255]
    compress.stats["lower_percentiles"] = [0, 0, 0]
    compress.process_image(0, image, meta)
    filenames = glob.glob(os.path.join(out_dir, "*"))
    real_filenames = [os.path.join(out_dir, x) for x in ["dna.png", "er.png", "mito.png"]]
    filenames.sort()

    assert real_filenames == filenames

    for i in range(3):
        assert os.path.exists(filenames[i])
        data = skimage.io.imread(filenames[i])
        numpy.testing.assert_allclose(image[:, :, i], data, rtol=0.04, atol=0)
        geq = (data >= image[:, :, i])
        assert numpy.count_nonzero(geq) == geq.size
