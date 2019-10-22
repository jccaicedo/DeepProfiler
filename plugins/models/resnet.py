import keras
import keras_resnet.models

from deepprofiler.learning.model import DeepProfilerModel

##################################################
# ResNet architecture as defined in "Deep Residual
# Learning for Image Recognition" by Kaiming He,
# Xiangyu Zhang, Shaoqing Ren, Jian Sun
# https://arxiv.org/abs/1512.03385
##################################################

supported_models = {
    18: keras_resnet.models.ResNet18,
    50: keras_resnet.models.ResNet2D50,
    101: keras_resnet.models.ResNet2D101,
    152: keras_resnet.models.ResNet2D152,
    200: keras_resnet.models.ResNet2D200
}
SM = "ResNet supported models: " + ",".join([str(x) for x in supported_models.keys()])


def define_model(config, dset):
    # 1. Create ResNet architecture to extract features
    input_shape = (
        config["train"]["sampling"]["box_size"],  # height
        config["train"]["sampling"]["box_size"],  # width
        len(config["dataset"]["images"]["channels"])  # channels
    )
    input_image = keras.layers.Input(input_shape)

    num_layers = config["train"]["model"]["params"]["conv_blocks"]
    error_msg = str(num_layers) + " conv_blocks not in " + SM
    assert num_layers in supported_models.keys(), error_msg

    model = supported_models[num_layers](input_image, include_top=False)
    features = keras.layers.GlobalAveragePooling2D(name="pool5")(model.layers[-1].output)

    # TODO: factorize the multi-target output model

    # 2. Create an output embedding for each target
    class_outputs = []

    i = 0
    for t in dset.targets:
        y = keras.layers.Dense(t.shape[1], activation="softmax", name=t.field_name)(features)
        class_outputs.append(y)
        i += 1

    # 3. Define the loss function
    loss_func = "categorical_crossentropy"

    # 4. Create and compile model
    model = keras.models.Model(inputs=input_image, outputs=class_outputs)
    optimizer = keras.optimizers.SGD(lr=config["train"]["model"]["params"]["learning_rate"], momentum=0.9,
                                     nesterov=True)

    return model, optimizer, loss_func


class ModelClass(DeepProfilerModel):
    def __init__(self, config, dset, generator, val_generator):
        super(ModelClass, self).__init__(config, dset, generator, val_generator)
        self.feature_model, self.optimizer, self.loss = define_model(config, dset)
