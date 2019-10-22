import keras
from keras import backend as K
from keras import objectives
from keras.layers import *
from keras.models import Model, Sequential
from keras.optimizers import Adam

from deepprofiler.learning.model import DeepProfilerModel


##################################################
# Convolutional variational autoencoder with
# alternating convolutions and max pooling
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
    encoded_shape = x._keras_shape[1:]
    x = Flatten()(x)
    flattened_shape = x._keras_shape[1:]

    # Define mean and log variance layers
    z_mean = Dense(config["train"]["model"]["params"]["latent_dim"], name="z_mean")(x)
    z_log_sigma = Dense(config["train"]["model"]["params"]["latent_dim"], name="z_log_sigma")(x)

    # Sampling function for latent variable
    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(
            shape=(config["train"]["model"]["params"]["batch_size"], config["train"]["model"]["params"]["latent_dim"]),
            mean=0., stddev=config["train"]["model"]["params"]["epsilon_std"])
        return z_mean + K.exp(z_log_sigma) * epsilon

    z = Lambda(sampling, output_shape=(config["train"]["model"]["params"]["latent_dim"],), name="z")(
        [z_mean, z_log_sigma])
    encoder = Model(input_image, z_mean)

    # Define decoder
    decoder_input = Input((config["train"]["model"]["params"]["latent_dim"],))
    decoder_layers = []
    decoder_layers.append(
        Dense(flattened_shape[0], activation="relu", input_shape=(config["train"]["model"]["params"]["latent_dim"],)))
    decoder_layers.append(Reshape(encoded_shape))
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
    decoded = decoder(z)
    generator = Model(decoder_input, decoder(decoder_input))

    # Define VAE
    vae = Model(input_image, decoded)

    # Define variational loss function
    def vae_loss(x, x_decoded_mean):
        x = K.flatten(x)
        x_decoded_mean = K.flatten(x_decoded_mean)
        xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
        return xent_loss + kl_loss

    optimizer = Adam(lr=config["train"]["model"]["params"]["learning_rate"])

    return vae, encoder, generator, optimizer, vae_loss


class ModelClass(DeepProfilerModel):
    def __init__(self, config, dset, generator, val_generator):
        super(ModelClass, self).__init__(config, dset, generator, val_generator)
        self.feature_model, self.encoder, self.generator, self.optimizer, self.loss = define_model(config, dset)
