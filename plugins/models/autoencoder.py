import keras
from keras.layers import *
from keras.models import Model, Sequential
from keras.optimizers import Adam

from deepprofiler.learning.model import DeepProfilerModel


##################################################
# Convolutional autoencoder with alternating
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

    # Add convolutional blocks to encoder based on number specified in config, with increasing number of filters
    x = input_image
    for i in range(config["train"]["model"]["params"]["conv_blocks"]):
        x = Conv2D(8 * 2 ** i, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D((2, 2))(x)
    conv_shape = x._keras_shape[1:]
    x = Flatten()(x)
    flattened_shape = x._keras_shape[1:]
    encoded = Dense(config["train"]["model"]["params"]["feature_dim"], name="encoded")(x)
    encoded_shape = encoded._keras_shape[1:]
    encoder = Model(input_image, encoded)

    # Build decoder
    decoder_input = Input(encoded_shape)
    decoder_layers = []
    decoder_layers.append(Dense(flattened_shape[0], input_shape=encoded_shape))
    decoder_layers.append(Reshape(conv_shape))
    for i in reversed(range(config["train"]["model"]["params"]["conv_blocks"])):
        decoder_layers.extend([
            Conv2DTranspose(8 * 2 ** i, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            UpSampling2D((2, 2))
        ])
    decoder_layers.append(
        Conv2DTranspose(len(config["dataset"]["images"]["channels"]), (3, 3), activation="sigmoid", padding="same"))
    decoder = Sequential(decoder_layers, name="decoded")
    decoded = decoder(encoded)
    decoder = Model(decoder_input, decoder(decoder_input))

    # Define autoencoder
    autoencoder = Model(input_image, decoded)
    optimizer = Adam(lr=config["train"]["model"]["params"]["learning_rate"])
    loss = "mse"

    return autoencoder, encoder, decoder, optimizer, loss


class ModelClass(DeepProfilerModel):
    def __init__(self, config, dset, generator, val_generator):
        super(ModelClass, self).__init__(config, dset, generator, val_generator)
        self.feature_model, self.encoder, self.decoder, self.optimizer, self.loss = define_model(config, dset)
