import numpy as np
import pandas as pd

import deepprofiler.imaging.cropping


class Mixup(object):

    def __init__(self, table_size, crop_shape, target_size, alpha):
        self.alpha = alpha
        self.table_size = table_size
        self.target_size = target_size
        self.crops = np.zeros((table_size, crop_shape[0][0], crop_shape[0][1], crop_shape[0][2]))
        # TODO: Support for multiple targets
        self.labels = pd.DataFrame(data=np.zeros((table_size), dtype=np.int32), columns=["target"])
        self.pointer = 0
        self.ready = False

    def add_crops(self, crops, labels):
        left_space = self.table_size - self.pointer
        load = crops.shape[0]
        if left_space >= load:
            self.crops[self.pointer:self.pointer + load, ...] = crops
            index = [i for i in range(self.pointer, self.pointer + load)]
            self.labels.loc[index, "target"] = np.argmax(labels, axis=1)
            self.pointer += load
        else:
            self.crops[self.pointer:self.pointer + left_space, ...] = crops[0:left_space, ...]
            index = [i for i in range(self.pointer, self.pointer + left_space)]
            self.labels.loc[index, "target"] = np.argmax(labels[0:left_space], axis=1)
            self.pointer = 0
            self.ready = True
            self.add_crops(crops[left_space:, ...], labels[left_space:, ...])

    def batch(self, batch_size, seed=None):
        np.random.seed(seed)
        targets = self.labels["target"].unique()
        s, w, h, c = self.crops.shape
        data = np.zeros((batch_size, w, h, c))
        labels = np.zeros((batch_size, self.target_size))

        for i in range(batch_size):
            lam = np.random.beta(self.alpha, self.alpha)
            sample = self.labels.sample(n=2, random_state=seed)
            idx = sample.index.tolist()
            data[i, :, :, :] = lam * self.crops[idx[0], ...] + (1. - lam) * self.crops[idx[1], ...]
            labels[i, sample.loc[idx[0], "target"]] += lam
            labels[i, sample.loc[idx[1], "target"]] += 1. - lam
        return data, labels


class GeneratorClass(deepprofiler.imaging.cropping.CropGenerator):

    def start(self, session):
        super().start(session)

        self.batch_size = self.config["train"]["model"]["params"]["batch_size"]
        self.target_sizes = []
        targets = [t for t in self.train_variables.keys() if t.startswith("target_")]
        targets.sort()
        for t in targets:
            self.target_sizes.append(self.train_variables[t].shape[1])
        self.mixer = Mixup(
            self.config["train"]["queueing"]["queue_size"],
            self.input_variables["shapes"]["crops"],
            self.target_sizes[0],
            self.config["train"]["sampling"]["alpha"]
        )

    def generate(self, sess, global_step=0):
        pool_index = np.arange(self.image_pool.shape[0])
        while True:
            if self.coord.should_stop():
                break
            data = self.sample_batch(pool_index)
            # Indices of data => [0] images, [1:-1] targets, [-1] summary
            self.mixer.add_crops(data[0], data[1])  # TODO: support for multiple targets
            while not self.mixer.ready:
                data = self.sample_batch(pool_index)
                self.mixer.add_crops(data[0], data[1])

            global_step += 1
            batch = self.mixer.batch(self.batch_size)

            yield (batch[0], batch[1])  # TODO: support for multiple targets


SingleImageGeneratorClass = deepprofiler.imaging.cropping.SingleImageCropGenerator
