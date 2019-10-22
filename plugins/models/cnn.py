import keras
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam

from deepprofiler.learning.model import DeepProfilerModel


##################################################
# Basic convolutional network with alternating
# convolutions and max pooling
##################################################


def define_model(config, dset):
    # Define input layer
    input_shape = (
        config["train"]["sampling"]["box_size"],  # height
        config["train"]["sampling"]["box_size"],  # width
        len(config["dataset"]["images"]["channels"])  # channels
    )
    input_image = keras.layers.Input(input_shape)

    if config["train"]["model"]["params"]["conv_blocks"] < 1:
        raise ValueError("At least 1 convolutional block is required.")

    # Add convolutional blocks based on number specified in config, with increasing number of filters
    x = input_image
    for i in range(config["train"]["model"]["params"]["conv_blocks"]):
        x = Conv2D(8 * 2 ** i, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D((2, 2), padding="same")(x)
    x = Flatten()(x)
    features = Dense(config["train"]["model"]["params"]["feature_dim"], activation="relu", name="features")(x)

    # Create an output embedding for each target
    class_outputs = []
    i = 0
    for t in dset.targets:
        y = keras.layers.Dense(t.shape[1], activation="softmax", name=t.field_name)(features)
        class_outputs.append(y)
        i += 1

    # Define model
    model = Model(input_image, class_outputs)
    optimizer = Adam(lr=config["train"]["model"]["params"]["learning_rate"])
    loss = "categorical_crossentropy"
    return model, optimizer, loss


class ModelClass(DeepProfilerModel):
    def __init__(self, config, dset, generator, val_generator):
        super(ModelClass, self).__init__(config, dset, generator, val_generator)
        self.feature_model, self.optimizer, self.loss = define_model(config, dset)
