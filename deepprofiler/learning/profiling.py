import importlib
import os

import keras
import numpy
import tensorflow
import keras.backend
import keras.models

from deepprofiler.dataset.utils import tic, toc


class Profile(object):

    def __init__(self, config, dset):
        self.config = config
        self.dset = dset
        self.num_channels = len(self.config["dataset"]["images"]["channels"])
        self.crop_generator = importlib.import_module(
            "plugins.crop_generators.{}".format(config["train"]["model"]["crop_generator"])) \
            .GeneratorClass
        self.profile_crop_generator = importlib.import_module(
            "plugins.crop_generators.{}".format(config["train"]["model"]["crop_generator"])) \
            .SingleImageGeneratorClass
        self.dpmodel = importlib.import_module("plugins.models.{}".format(config["train"]["model"]["name"])) \
            .ModelClass(config, dset, self.crop_generator, self.profile_crop_generator)
        self.profile_crop_generator = self.profile_crop_generator(config, dset)

    def configure(self):
        # Main session configuration
        configuration = tensorflow.ConfigProto()
        configuration.gpu_options.allow_growth = True
        self.sess = tensorflow.Session(config=configuration)
        self.profile_crop_generator.start(self.sess)
        keras.backend.set_session(self.sess)

        # Create feature extractor
        if self.config["profile"]["pretrained"]:
            checkpoint = self.config["paths"]["pretrained"] + "/" + self.config["profile"]["checkpoint"]
        else:
            checkpoint = self.config["paths"]["checkpoints"] + "/" + self.config["profile"]["checkpoint"]
        self.dpmodel.feature_model.load_weights(checkpoint)
        self.dpmodel.feature_model.summary()
        self.feat_extractor = keras.Model(self.dpmodel.feature_model.inputs, self.dpmodel.feature_model.get_layer(
            self.config["profile"]["feature_layer"]).output)
        self.feat_extractor.summary()

    def check(self, meta):
        output_file = self.config["paths"]["features"] + "/{}_{}_{}.npz"
        output_file = output_file.format(meta["Metadata_Plate"], meta["Metadata_Well"], meta["Metadata_Site"])

        # Check if features were computed before
        if os.path.isfile(output_file):
            print("Already done:", output_file)
            return False
        else:
            return True

    # Function to process a single image
    def extract_features(self, key, image_array, meta):  # key is a placeholder
        start = tic()
        output_file = self.config["paths"]["features"] + "/{}_{}_{}.npz"
        output_file = output_file.format(meta["Metadata_Plate"], meta["Metadata_Well"], meta["Metadata_Site"])

        batch_size = self.config["profile"]["batch_size"]
        image_key, image_names, outlines = self.dset.get_image_paths(meta)
        total_crops = self.profile_crop_generator.prepare_image(
            self.sess,
            image_array,
            meta,
            False
        )
        if total_crops == 0:
            print("No cells to profile:", output_file)
            return
        num_features = self.config["train"]["model"]["params"]["feature_dim"]
        repeats = "channel_repeats" in self.config["dataset"]["images"].keys()

        # Extract features
        crops = next(self.profile_crop_generator.generate(self.sess))[0]  # single image crop generator yields one batch
        feats = self.feat_extractor.predict(crops, batch_size=batch_size)
        if repeats:
            feats = numpy.reshape(feats, (self.num_channels, total_crops, num_features))
            feats = numpy.concatenate(feats, axis=-1)

        # Save features
        while len(feats.shape) > 2:  # 2D mean spatial pooling
            feats = numpy.mean(feats, axis=1)

        numpy.savez_compressed(output_file, f=feats)
        toc(image_key + " (" + str(total_crops) + " cells)", start)


def profile(config, dset):
    profile = Profile(config, dset)
    profile.configure()
    dset.scan(profile.extract_features, frame="all", check=profile.check)
    print("Profiling: done")
