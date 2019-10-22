import os

import keras
import numpy as np
import tensorflow as tf
from keras.layers import BatchNormalization
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.models import Model
from keras.optimizers import Adam

from deepprofiler.learning import model
from deepprofiler.learning.model import DeepProfilerModel


#######################################################
# Based on https://github.com/eriklindernoren/Keras-GAN
#######################################################


class GAN(object):
    def __init__(self, config, crop_generator, val_crop_generator):

        if config["train"]["model"]["params"]["conv_blocks"] < 1:
            raise ValueError("At least 1 convolutional block is required.")

        self.config = config
        self.crop_generator = crop_generator
        self.val_crop_generator = val_crop_generator
        self.img_rows = config["train"]["sampling"]["box_size"]
        self.img_cols = config["train"]["sampling"]["box_size"]
        self.channels = len(config["dataset"]["images"]["channels"])
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = config["train"]["model"]["params"]["latent_dim"]  # TODO: move to params

        optimizer = Adam(config["train"]["model"]["params"]["learning_rate"], 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss="binary_crossentropy",
                                   optimizer=optimizer,
                                   metrics=["accuracy"])  # TODO

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator_fixed = Model(inputs=self.discriminator.inputs, outputs=self.discriminator.outputs,
                                         name="discriminator")
        self.discriminator_fixed.trainable = False
        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator_fixed(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss="binary_crossentropy", optimizer=optimizer)  # TODO

    def build_generator(self):

        s = self.config["train"]["sampling"]["box_size"] // 2 ** self.config["train"]["model"]["params"]["conv_blocks"]
        if s < 1:
            raise ValueError("Too many convolutional blocks for the specified crop size!")
        noise = Input(shape=(self.latent_dim,))
        x = Dense(s * s, input_dim=self.latent_dim)(noise)
        x = LeakyReLU(alpha=0.2)(x)
        x = Reshape((s, s, 1))(x)
        for i in reversed(range(self.config["train"]["model"]["params"]["conv_blocks"])):
            x = Conv2DTranspose(8 * 2 ** i, (3, 3), padding="same")(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = UpSampling2D((2, 2))(x)
        img = Conv2DTranspose(self.channels, (3, 3), padding="same", activation="sigmoid")(x)

        return Model(noise, img, name="generator")

    def build_discriminator(self):

        img = Input(shape=self.img_shape)
        x = img
        for i in range(self.config["train"]["model"]["params"]["conv_blocks"]):
            x = Conv2D(8 * 2 ** i, (3, 3), padding="same")(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(self.config["train"]["model"]["params"]["feature_dim"], name="features")(x)
        x = LeakyReLU(alpha=0.2)(x)
        validity = Dense(1, activation="sigmoid")(x)

        return Model(img, validity, name="discriminator")

    def train(self, epochs, steps_per_epoch, init_epoch):
        sess = tf.Session()
        crop_generator = self.crop_generator.generate(sess)
        for epoch in range(init_epoch, epochs + 1):
            for step in range(steps_per_epoch):
                crops = next(crop_generator)[0]
                batch_size = crops.shape[0]

                valid = np.ones((batch_size, 1))
                fake = np.zeros((batch_size, 1))

                # ---------------------
                #  Train Discriminator
                # ---------------------

                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_crops = self.generator.predict(noise)

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(crops, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_crops, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Train the generator (to have the discriminator label samples as valid)
                g_loss = self.combined.train_on_batch(noise, valid)

                # Plot the progress
                print("Epoch %d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            filename_d = os.path.join(self.config["paths"]["checkpoints"],
                                      "{}_epoch_{:04d}.hdf5".format("discriminator", epoch))
            filename_g = os.path.join(self.config["paths"]["checkpoints"],
                                      "{}_epoch_{:04d}.hdf5".format("generator", epoch))
            self.discriminator.save_weights(filename_d)
            self.generator.save_weights(filename_g)


class ModelClass(DeepProfilerModel):
    def __init__(self, config, dset, generator, val_generator):
        super(ModelClass, self).__init__(config, dset, generator, val_generator)
        self.gan = GAN(config, self.train_crop_generator, self.val_crop_generator)
        self.feature_model = self.gan.discriminator

    def train(self, epoch=1, metrics=["accuracy"], verbose=1):
        model.check_feature_model(self)
        self.gan.combined.summary()
        experiment = model.setup_comet_ml(self)  # TODO: comet ml doesn't currently work with this model
        configuration = model.tf_configure()
        crop_session = model.start_crop_session(self, configuration)
        # TODO: no validation
        main_session = model.start_main_session(configuration)
        discriminator_file = os.path.join(self.config["paths"]["checkpoints"],
                                          "{}_epoch_{:04d}.hdf5".format("discriminator", epoch - 1))
        generator_file = os.path.join(self.config["paths"]["checkpoints"],
                                      "{}_epoch_{:04d}.hdf5".format("generator", epoch - 1))
        if epoch >= 1 and os.path.isfile(discriminator_file) and os.path.isfile(generator_file):
            self.gan.discriminator.load_weights(discriminator_file)
            self.gan.generator.load_weights(generator_file)
            print("Weights from previous models loaded:", discriminator_file, generator_file)
        else:
            keras.backend.get_session().run(tf.global_variables_initializer())  # workaround for tf bug
        # TODO: no callbacks
        epochs, steps, lr_schedule_epochs, lr_schedule_lr = model.setup_params(self, experiment)
        self.gan.train(epochs, steps, epoch)
        model.close(self, crop_session)
        # TODO: no return values
