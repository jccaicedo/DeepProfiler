import tensorflow as tf
from keras import backend as K
from keras.applications import inception_resnet_v2
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

from deepprofiler.learning.model import DeepProfilerModel


def define_model(config, dset):
    # Set session
    configuration = tf.ConfigProto()
    configuration.gpu_options.allow_growth = True
    sess = tf.Session(config=configuration)
    K.set_session(sess)

    # Load InceptionResnetV2 base architecture
    if config["profile"]["pretrained"]:
        weights = None
        input_tensor = Input((299, 299, 3), name="input")
        base = inception_resnet_v2.InceptionResNetV2(
            include_top=True,
            input_tensor=input_tensor
        )
        base.get_layer(index=-2).name = "global_{}_pool".format(config["train"]["model"]["params"]["pooling"])
        # Define model
        model = base
    else:
        weights = None
        input_tensor = Input((
            config["train"]["sampling"]["box_size"],  # height
            config["train"]["sampling"]["box_size"],  # width
            len(config["dataset"]["images"]["channels"])  # channels
        ), name="input")
        base = inception_resnet_v2.InceptionResNetV2(
            include_top=False,
            weights=weights,
            input_tensor=input_tensor,
            pooling=config["train"]["model"]["params"]["pooling"],
            classes=dset.targets[0].shape[1]
        )
        base.get_layer(index=-1).name = "global_{}_pool".format(config["train"]["model"]["params"]["pooling"])
        # Create output embedding for each target
        class_outputs = []
        i = 0
        for t in dset.targets:
            y = Dense(t.shape[1], activation="softmax", name=t.field_name)(base.output)
            class_outputs.append(y)
            i += 1
        # Define model
        model = Model(input_tensor, class_outputs)

    # Define optimizer and loss
    optimizer = Adam(lr=config["train"]["model"]["params"]["learning_rate"])
    loss = "categorical_crossentropy"

    return model, optimizer, loss


class ModelClass(DeepProfilerModel):
    def __init__(self, config, dset, generator, val_generator):
        super(ModelClass, self).__init__(config, dset, generator, val_generator)
        self.feature_model, self.optimizer, self.loss = define_model(config, dset)
