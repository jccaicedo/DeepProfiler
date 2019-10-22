import numpy as np


class Validation(object):

    def __init__(self, config, dset, crop_generator, session):
        self.config = config
        self.dset = dset
        self.crop_generator = crop_generator
        self.session = session
        self.batch_inputs = []
        self.batch_outputs = []
        self.count = 0

    def process_batches(self, key, image_array, meta):
        # Prepare image for cropping
        total_crops = self.crop_generator.prepare_image(
            self.session,
            image_array,
            meta,
            self.config["train"]["validation"]["sample_first_crops"]
        )
        self.count += 1
        if total_crops > 0:
            # We expect all crops in a single batch
            batches = [b for b in self.crop_generator.generate(self.session)]
            self.batch_inputs.append(batches[0][0])
            self.batch_outputs.append(batches[0][1])
        print("Loading validation data:", self.count, "records of", self.dset.number_of_records("val"), end="\r")


def load_validation_data(config, dset, crop_generator, session):
    validation = Validation(config, dset, crop_generator, session)
    dset.scan(validation.process_batches, frame="val")
    print("Validation data loaded")
    return np.concatenate(validation.batch_inputs), np.concatenate(validation.batch_outputs)
