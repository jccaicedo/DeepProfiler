import gc
import threading
import time

import keras
import numpy
import tensorflow

import deepprofiler.dataset.utils
import deepprofiler.imaging.augmentations
import deepprofiler.imaging.boxes


def crop_graph(image_ph, boxes_ph, box_ind_ph, mask_ind_ph, box_size, mask_boxes=False):
    with tensorflow.variable_scope("cropping"):
        crop_size_ph = tensorflow.constant([box_size, box_size], name="crop_size")
        crops = tensorflow.image.crop_and_resize(image_ph, boxes_ph, box_ind_ph, crop_size_ph)
        if mask_boxes:
            mask_ind = tensorflow.expand_dims(tensorflow.expand_dims(mask_ind_ph, -1), -1)
            mask_values = tensorflow.ones_like(crops[:, :, :, -1], dtype=tensorflow.float32) * tensorflow.cast(mask_ind, dtype=tensorflow.float32)
            masks = tensorflow.to_float(tensorflow.equal(crops[:, :, :, -1], mask_values))
            crops = crops[:, :, :, 0:-1] * tensorflow.expand_dims(masks, -1)
        max_intensities = tensorflow.reduce_max(tensorflow.reduce_max(crops, axis=1, keepdims=True), axis=2, keepdims=True)
        crops = crops / (max_intensities + 1e-6)
    return crops


# TODO: implement abstract crop generator
class CropGenerator(object):

    def __init__(self, config, dset):  # TODO: add mode="train"
        self.config = config
        self.dset = dset
        # TODO: add self.mode = mode

    # INPUT GRAPH DEFINITION

    def build_input_graph(self):
        # Identify number of channels
        mask_objects = self.config["train"]["sampling"]["mask_objects"]
        if mask_objects:
            img_channels = len(self.config["dataset"]["images"]["channels"]) + 1
        else:
            img_channels = len(self.config["dataset"]["images"]["channels"])
        crop_channels = len(self.config["dataset"]["images"]["channels"])

        # Identify image and box sizes
        box_size = self.config["train"]["sampling"]["box_size"]
        img_width = self.config["dataset"]["images"]["width"]
        img_height = self.config["dataset"]["images"]["height"]

        # Data shapes
        num_targets = len(self.dset.targets)
        crop_shape = [(box_size, box_size, crop_channels)] + [()] * num_targets
        imgs_shape = [None, img_height, img_width, img_channels]
        batch_shape = (-1, img_height, img_width, img_channels)

        # Inputs to the load data queue
        image_ph = tensorflow.placeholder(tensorflow.float32, shape=imgs_shape, name="raw_images")
        boxes_ph = tensorflow.placeholder(tensorflow.float32, shape=[None, 4], name="cell_boxes")
        box_ind_ph = tensorflow.placeholder(tensorflow.int32, shape=[None], name="box_indicators")
        mask_ind_ph = tensorflow.placeholder(tensorflow.int32, shape=[None], name="mask_indicators")
        targets_phs = {}
        for i in range(num_targets):
            tname = "target_" + str(i)
            tgt = self.dset.targets[i]
            targets_phs[tname] = tensorflow.placeholder(tensorflow.int32, shape=[None], name=tname)

        # Outputs and queue of the cropping graph
        crop_op = crop_graph(
            image_ph,
            boxes_ph,
            box_ind_ph,
            mask_ind_ph,
            box_size,
            mask_objects
        )
        labeled_crops = tensorflow.tuple([crop_op] + [targets_phs[t] for t in targets_phs.keys()])

        self.input_variables = {
            "image_ph": image_ph,
            "boxes_ph": boxes_ph,
            "box_ind_ph": box_ind_ph,
            "targets_phs": targets_phs,
            "mask_ind_ph": mask_ind_ph,
            "labeled_crops": labeled_crops,
            "shapes": {
                "crops": crop_shape,
                "images": imgs_shape,
                "batch": batch_shape
            },
        }

    # AUGMENTATION GRAPH DEFINITION

    def build_augmentation_graph(self):
        num_targets = len(self.dset.targets)

        # Outputs and queue of the data augmentation graph
        augmented_op = deepprofiler.imaging.augmentations.augment_multiple(
            tensorflow.cast(self.input_variables["labeled_crops"][0], tensorflow.float32),
            self.config["train"]["model"]["params"]["batch_size"]
        )
        train_inputs = tensorflow.tuple([augmented_op] + self.input_variables["labeled_crops"][1:])

        self.train_variables = {
            "image_batch": train_inputs[0],
        }

        for i in range(num_targets):
            tname = "target_" + str(i)
            tgt = self.dset.targets[i]
            self.train_variables[tname] = tensorflow.one_hot(train_inputs[i + 1], tgt.shape[1])

    # START TRAINING QUEUES

    def training_queues(self, sess):
        coord = tensorflow.train.Coordinator()
        lock = threading.Lock()
        self.exception_occurred = False

        # Enqueueing threads for raw images
        # vvv Start of thread function vvv
        def data_loading_thread():
            while not coord.should_stop():
                try:
                    # Load images and cell boxes
                    batch = deepprofiler.imaging.boxes.load_batch(self.dset, self.config)  # TODO
                    images = numpy.reshape(batch["images"], self.input_variables["shapes"]["batch"])
                    boxes, box_ind, targets, masks = deepprofiler.imaging.boxes.prepare_boxes(batch, self.config)
                    feed_dict = {
                        self.input_variables["image_ph"]: images,
                        self.input_variables["boxes_ph"]: boxes,
                        self.input_variables["box_ind_ph"]: box_ind,
                        self.input_variables["mask_ind_ph"]: masks
                    }
                    for target_index in range(len(targets)):
                        tname = "target_" + str(target_index)
                        feed_dict[self.input_variables["targets_phs"][tname]] = targets[target_index]

                    output = sess.run(self.train_variables, feed_dict)

                    # Remove crops without any content TODO: enable multiple targets
                    valid = numpy.sum(output["image_batch"], axis=(1, 2, 3)) != 0
                    output["image_batch"] = output["image_batch"][valid, ...]
                    output["target_0"] = output["target_0"][valid, ...]

                    # Find block of the pool to store data
                    lock.acquire()
                    first = self.pool_pointer
                    records = output["image_batch"].shape[0]

                    if self.pool_pointer + records < self.image_pool.shape[0]:
                        last = self.pool_pointer + records
                        self.pool_pointer += records
                    else:
                        last = self.image_pool.shape[0]
                        records = last - first
                        self.pool_pointer = 0
                        self.ready_to_sample = True

                    # Replace block (TODO:order of targets and keys may be wrong)
                    self.image_pool[first:last, ...] = output["image_batch"][0:records, ...]
                    k = 0
                    for t in output.keys():
                        if t.startswith("target_"):
                            self.label_pool[k][first:last, :] = output[t][0:records, :]
                            k += 1
                    lock.release()

                except Exception:
                    import traceback
                    traceback.print_exc()
                    print(".", end="", flush=True)
                    self.exception_occurred = True
                    return

        # ^^^ End of thread function ^^^

        load_threads = []
        for i in range(self.config["train"]["queueing"]["loading_workers"]):
            lt = threading.Thread(target=data_loading_thread)
            load_threads.append(lt)
            lt.isDaemon()
            lt.start()

        return coord, load_threads

    def start(self, session):
        # Define input data batches
        with tensorflow.variable_scope("train_inputs"):
            self.build_input_graph()
            self.build_augmentation_graph()
            targets = [self.train_variables[t] for t in self.train_variables.keys() if t.startswith("target_")]

            self.image_pool = numpy.zeros(
                [self.config["train"]["queueing"]["queue_size"]] + list(self.input_variables["shapes"]["crops"][0]))
            self.label_pool = [numpy.zeros([self.config["train"]["queueing"]["queue_size"], t.shape[1]]) for t in targets]
            self.pool_pointer = 0
            self.ready_to_sample = False
            print("Waiting for data", self.image_pool.shape, [l.shape for l in self.label_pool])

        self.merged_summary = tensorflow.summary.merge_all()
        self.summary_writer = tensorflow.summary.FileWriter(self.config["paths"]["summaries"], session.graph)

        # Start data threads
        self.coord, self.queue_threads = self.training_queues(session)
        tensorflow.train.start_queue_runners(coord=self.coord, sess=session)

    def sample_batch(self, pool_index):
        while not self.ready_to_sample:
            time.sleep(2)
        numpy.random.shuffle(pool_index)  # TODO
        idx = pool_index[0:self.config["train"]["model"]["params"]["batch_size"]]
        # TODO: make outputs for all targets
        data = [self.image_pool[idx, ...], self.label_pool[0][idx, :], 0]
        return data

    def generate(self, sess, global_step=0):
        pool_index = numpy.arange(self.image_pool.shape[0])
        while True:
            if self.coord.should_stop():
                break
            data = self.sample_batch(pool_index)
            # Indices of data => [0] images, [1:-1] targets, [-1] summary
            # ms = data[-1]

            global_step += 1
            # if global_step % 10 == 0:
            #    self.summary_writer.add_summary(ms, global_step)

            yield (data[0], data[1:-1])

    def stop(self, session):
        self.coord.request_stop()
        self.coord.join(self.queue_threads)
        session.close()
        gc.collect()


# SUB CLASS TO GENERATE ALL CROPS IN A SINGLE IMAGE

# Useful for validation, predictions and profiling.
# Important differences to the above class:
# * No randomization is performed for crop generation
# * A batch of crops is padded with zeros at the end
#   for avoiding mixing of crops from different images.
# * Only one queue is used for loading a single image
#   and creating all crops in that image
# * No need to stop threads.
# * The generate method yields crops for a single image
# * The generator needs to be restarted for each image.
#########################################################

class SingleImageCropGenerator(CropGenerator):

    def start(self, session):
        # Define input data batches
        with tensorflow.variable_scope("train_inputs"):
            self.config["train"]["model"]["params"]["batch_size"] = self.config["train"]["validation"]["batch_size"]
            self.build_input_graph()
            # Align cells by rotating nuclei
            # self.angles = tf.placeholder(tf.float32, shape=[None], name="nuclei_angles")
            # rotated_imgs = tf.contrib.image.rotate(self.input_variables["labeled_crops"][0], self.angles, interpolation="BILINEAR")
            # self.aligned_labeled = [rotated_imgs, self.input_variables["labeled_crops"][1]]

    def prepare_image(self, session, image_array, meta, sample_first_crops=False):

        num_targets = len(self.dset.targets)
        self.batch_size = self.config["train"]["validation"]["batch_size"]
        image_key, image_names, outlines = self.dset.get_image_paths(meta)

        batch = {"images": [], "locations": [], "targets": [[] for i in range(num_targets)]}
        batch["images"].append(image_array)
        batch["locations"].append(deepprofiler.imaging.boxes.get_locations(image_key, self.config, randomize=False))
        for i in range(num_targets):
            tgt = self.dset.targets[i]
            batch["targets"][i].append(tgt.get_values(meta))

        if sample_first_crops and self.batch_size < len(batch["locations"][0]):
            batch["locations"][0] = batch["locations"][0].head(self.batch_size)

        has_orientation = "Orientation" in batch["locations"][0].columns
        boxes, box_ind, targets, mask_ind = deepprofiler.imaging.boxes.prepare_boxes(batch, self.config)
        batch["images"] = numpy.reshape(image_array, self.input_variables["shapes"]["batch"])
        feed_dict = {
            self.input_variables["image_ph"]: batch["images"],
            self.input_variables["boxes_ph"]: boxes,
            self.input_variables["box_ind_ph"]: box_ind,
            self.input_variables["mask_ind_ph"]: mask_ind
        }

        for i in range(num_targets):
            tname = "target_" + str(i)
            feed_dict[self.input_variables["targets_phs"][tname]] = targets[i]

        total_crops = len(batch["locations"][0])

        # if has_orientation:
        #    # Align cells by rotating to the major axis of nuclei
        #    feed_dict[self.angles] = (batch["locations"][0]["Orientation"]*deepprofiler.dataset.utils.PI)/180.
        #    output = session.run(self.aligned_labeled, feed_dict)
        # else:
        output = session.run(self.input_variables["labeled_crops"], feed_dict)

        output = {"image_batch": output[0], "target_0": output[1]}
        # Remove crops without any content TODO: enable multiple targets
        valid = numpy.sum(output["image_batch"], axis=(1, 2, 3)) != 0
        self.image_pool = output["image_batch"][valid, ...]
        self.label_pool = output["target_0"][valid, ...]
        num_classes = self.dset.targets[0].shape[1]
        self.label_pool = keras.utils.to_categorical(self.label_pool, num_classes=num_classes)

        return total_crops

    def generate(self, session, global_step=0):
        yield [self.image_pool, self.label_pool]
