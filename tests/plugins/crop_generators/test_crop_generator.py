import deepprofiler.imaging.cropping
import plugins.crop_generators.crop_generator


def test_crop_generator():
    assert plugins.crop_generators.crop_generator.GeneratorClass == deepprofiler.imaging.cropping.CropGenerator
    assert plugins.crop_generators.crop_generator.SingleImageGeneratorClass == deepprofiler.imaging.cropping.SingleImageCropGenerator
