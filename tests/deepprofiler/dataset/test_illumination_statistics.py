import numpy
import numpy.random
import numpy.testing
import pytest

import deepprofiler.dataset.illumination_statistics


@pytest.fixture(scope="function")
def illumination_stats():
    istats = deepprofiler.dataset.illumination_statistics.IlluminationStatistics(
        16,
        ["DNA", "ER", "Mito"],
        2,
        3
    )
    return istats


def test_init(illumination_stats):
    histogram = numpy.zeros((3, 2 ** 16), dtype=numpy.float64)

    assert illumination_stats.depth == 2 ** 16
    assert illumination_stats.channels == ["DNA", "ER", "Mito"]
    assert illumination_stats.name == ""
    assert illumination_stats.down_scale_factor == 2
    assert illumination_stats.median_filter_size == 3
    numpy.testing.assert_array_equal(illumination_stats.hist, histogram)
    assert illumination_stats.count == 0
    assert illumination_stats.expected == 1
    assert illumination_stats.mean_image is None
    assert illumination_stats.original_image_size is None


def test_add_to_mean_no_scaling(illumination_stats):
    numpy.random.seed(8)
    image = numpy.random.randint(256, size=(16, 16, 3), dtype=numpy.uint16)

    illumination_stats.down_scale_factor = 1
    illumination_stats.add_to_mean(image)

    assert illumination_stats.mean_image.shape == (16, 16, 3)
    # This method rescales the input image and normalizes pixels according to
    # the data type. We restore the values in this test to match the input for comparison.
    result_mean = illumination_stats.mean_image  # * (2 ** 16)
    numpy.testing.assert_array_equal(numpy.round(result_mean).astype(numpy.uint16), image)


def test_add_to_mean_with_scaling(illumination_stats):
    numpy.random.seed(8)
    image = numpy.random.randint(256, size=(16, 16, 3), dtype=numpy.uint16)

    illumination_stats.add_to_mean(image)

    assert illumination_stats.mean_image.shape == (8, 8, 3)
    result_mean = illumination_stats.mean_image
    assert result_mean.sum() > 0
    # numpy.testing.assert_array_equal(result_mean.astype(numpy.uint16), image)


def test_process_image(illumination_stats):
    numpy.random.seed(8)
    image = numpy.random.randint(256, size=(16, 16, 3), dtype=numpy.uint16)

    illumination_stats.process_image(0, image, None)

    histogram1 = numpy.histogram(image[:, :, 0], bins=2 ** 16, range=(0, 2 ** 16))[0]
    histogram2 = numpy.histogram(image[:, :, 1], bins=2 ** 16, range=(0, 2 ** 16))[0]
    histogram3 = numpy.histogram(image[:, :, 2], bins=2 ** 16, range=(0, 2 ** 16))[0]

    assert illumination_stats.count == 1
    numpy.testing.assert_array_equal(illumination_stats.hist[0], histogram1)
    numpy.testing.assert_array_equal(illumination_stats.hist[1], histogram2)
    numpy.testing.assert_array_equal(illumination_stats.hist[2], histogram3)


def test_compute_stats(illumination_stats):
    numpy.random.seed(8)
    image1 = numpy.random.randint(256, size=(16, 16, 3), dtype=numpy.uint16)
    image2 = numpy.random.randint(256, size=(16, 16, 3), dtype=numpy.uint16)

    illumination_stats.process_image(0, image1, None)
    illumination_stats.process_image(1, image2, None)

    stats = illumination_stats.compute_stats()

    keys = {"mean_values", "upper_percentiles", "lower_percentiles", "histogram", "mean_image", "channels",
            "original_size", "illum_correction_function"}
    result = set(stats.keys())

    assert len(result.difference(keys)) == 0
