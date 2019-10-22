import numpy
import numpy.random
import numpy.testing
import pytest

import deepprofiler.dataset.illumination_correction


@pytest.fixture(scope="function")
def mean_image():
    numpy.random.seed(81)

    mean_image = numpy.random.randint(256, size=(16, 16, 3), dtype=numpy.uint16)

    return mean_image


@pytest.fixture(scope="function")
def corrector(mean_image):
    stats = {"mean_image": mean_image}
    channels = ["DNA", "ER", "Mito"]
    target_dim = (24, 24, 3)
    return deepprofiler.dataset.illumination_correction.IlluminationCorrection(
        stats,
        channels,
        target_dim
    )


def test_init(corrector, mean_image):
    channels = ["DNA", "ER", "Mito"]
    target_dim = (24, 24, 3)

    numpy.testing.assert_array_equal(corrector.stats["mean_image"], mean_image)
    assert corrector.channels == channels
    assert corrector.target_dim[0] == target_dim[0]
    assert corrector.target_dim[1] == target_dim[1]


# With "enough" pixels we'd expect scipy.stats.scoreatpercentile(X, per=2) to return a value at 0.02 <- robust minimum
# 
def test_channel_function(corrector):
    numpy.random.seed(8)

    mean_image = numpy.random.uniform(low=0.0, high=255.0, size=(16, 16))

    result = corrector.channel_function(mean_image, 3)

    assert result.shape == (24, 24)

    assert numpy.all(result >= 1)


# def test_compute_all(corrector, mocker):  # Juan can remove mocker, it is a fun experiment
#     mocker.spy(corrector, "channel_function")
#
#     corrector.compute_all(3)
#
#     assert corrector.illum_corr_func.shape == (24, 24, 3)
#
#     assert corrector.channel_function.call_count == 3
#
#     corrector.channel_function.assert_called_with(mocker.ANY, 1.5)  # TODO: Claire feels bad about mocker.ANY
#
#     assert not numpy.all(corrector.illum_corr_func == 0)


def test_apply(corrector):
    image = numpy.random.randint(256, size=(24, 24, 3), dtype=numpy.uint16)

    illum_corr_func = numpy.random.rand(24, 24, 3)

    illum_corr_func /= illum_corr_func.min()

    corrector.illum_corr_func = illum_corr_func

    corrected = corrector.apply(image)

    expected = image / illum_corr_func

    assert corrected.shape == (24, 24, 3)

    numpy.testing.assert_array_equal(corrected, expected)
