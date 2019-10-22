import tempfile

import numpy.random
import skimage.io

import deepprofiler.dataset.pixels


def test_openImage():
    numpy.random.seed(11)

    # Creates a temporary directory and returns a string containing the path
    tmp_path = tempfile.mkdtemp()

    # Creates random images and outlines files and saves them to the termporary directory
    skimage.io.imsave(tmp_path + "rand_img_1.jpg", numpy.random.randint(256, size=(16, 16), dtype=numpy.uint16))
    skimage.io.imsave(tmp_path + "rand_img_2.jpg", numpy.random.randint(256, size=(16, 16), dtype=numpy.uint16))
    skimage.io.imsave(tmp_path + "rand_img_3.jpg", numpy.random.randint(256, size=(16, 16), dtype=numpy.uint16))
    skimage.io.imsave(tmp_path + "rand_outlines.jpg", numpy.random.randint(256, size=(16, 16), dtype=numpy.uint16))

    # Stores the paths to the random images and outlines files in a list and a string
    test_paths = [tmp_path + "rand_img_1.jpg", tmp_path + "rand_img_2.jpg", tmp_path + "rand_img_3.jpg"]
    test_outlines = tmp_path + "rand_outlines.jpg"

    # Creates an image from the random paths and tests the output dimensions
    new_img = deepprofiler.dataset.pixels.open_image(test_paths, None)
    assert new_img.shape == (16, 16, 3)

    # Add random the outlines and tests the dimensions again
    new_img = deepprofiler.dataset.pixels.open_image(test_paths, test_outlines)
    assert new_img.shape == (16, 16, 4)
