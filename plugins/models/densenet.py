'''DenseNet models for Keras.
# Reference
- [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)
- [The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/pdf/1611.09326.pdf)
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
import keras.applications

from deepprofiler.learning.model import DeepProfilerModel


def define_model(config, dset):
    # 1. Create ResNet architecture to extract features
    input_shape = (
        config["train"]["sampling"]["box_size"],  # height
        config["train"]["sampling"]["box_size"],  # width
        len(config["dataset"]["images"]["channels"])  # channels
    )
    input_image = keras.layers.Input(input_shape)

    model = keras.applications.DenseNet121(
        input_shape=input_shape,
        classes=dset.targets[0].shape[1],
        input_tensor=input_image,
        weights=None
    )

    # TODO: factorize the multi-target output model

    # 3. Define the loss function
    loss_func = "categorical_crossentropy"

    # 4. Create and compile model
    optimizer = keras.optimizers.Adam(lr=config["train"]["model"]["params"]["learning_rate"])

    return model, optimizer, loss_func


class ModelClass(DeepProfilerModel):
    def __init__(self, config, dset, generator, val_generator):
        super(ModelClass, self).__init__(config, dset, generator, val_generator)
        self.feature_model, self.optimizer, self.loss = define_model(config, dset)
