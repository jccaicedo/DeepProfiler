import os
import pickle as pickle

import numpy as np
import skimage.transform

import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.utils as utils
from .illumination_correction import IlluminationCorrection


def illum_stats_filename(output_dir, plate_name):
    return "{}/{}/{}.pkl".format(output_dir, plate_name, plate_name)


def percentile(prob, p):
    cum = np.cumsum(prob)
    pos = cum > p
    return np.argmax(pos)


# COMPUTATION OF ILLUMINATION STATISTICS

# Build pixel histogram for each channel
class IlluminationStatistics:
    def __init__(self, bits, channels, down_scale_factor, median_filter_size, name=""):
        self.depth = 2 ** bits
        self.channels = channels
        self.name = name
        self.down_scale_factor = down_scale_factor
        self.median_filter_size = median_filter_size
        self.hist = np.zeros((len(channels), self.depth), dtype=np.float64)
        self.count = 0
        self.expected = 1
        self.mean_image = None
        self.original_image_size = None

    def process_image(self, index, img, meta):
        self.add_to_mean(img)
        self.count += 1
        utils.logger.info("Plate {} Image {} of {} ({:4.2f}%)".format(self.name,
                                                                      self.count, self.expected,
                                                                      100 * float(self.count) / self.expected))
        for i in range(len(self.channels)):
            counts = np.histogram(img[:, :, i], bins=self.depth, range=(0, self.depth))[0]
            self.hist[i] += counts.astype(np.float64)

    # Accumulate the mean image. Useful for illumination correction purposes
    def add_to_mean(self, img):
        # Check image size (we assume all images have the same size)
        if self.original_image_size is None:
            self.original_image_size = img.shape
            self.scale = (img.shape[0] / self.down_scale_factor, img.shape[1] / self.down_scale_factor)
        else:
            if img.shape != self.original_image_size:
                raise ValueError("Images in this plate don't match: required=",
                                 self.original_image_size, " found=", img.shape)
        # Rescale original image to half
        thumb = skimage.transform.resize(img, self.scale, mode="reflect", anti_aliasing=True, preserve_range=True)
        if self.mean_image is None:
            self.mean_image = np.zeros_like(thumb, dtype=np.float64)
        # Add image to current mean values
        self.mean_image += thumb
        return

    # Compute global statistics on pixels. 
    def compute_stats(self):
        # Initialize counters
        bins = np.linspace(0, self.depth - 1, self.depth)
        mean = np.zeros((len(self.channels)))
        lower = np.zeros((len(self.channels)))
        upper = np.zeros((len(self.channels)))
        self.mean_image /= self.count

        # Compute percentiles and histogram
        for i in range(len(self.channels)):
            probs = self.hist[i] / self.hist[i].sum()
            mean[i] = (bins * probs).sum()
            lower[i] = percentile(probs, 0.0001)
            upper[i] = percentile(probs, 0.9999)
        stats = {"mean_values": mean, "upper_percentiles": upper, "lower_percentiles": lower, "histogram": self.hist,
                 "mean_image": self.mean_image, "channels": self.channels, "original_size": self.original_image_size}

        # Compute illumination correction function and add it to the dictionary
        correct = IlluminationCorrection(stats, self.channels, self.original_image_size)
        correct.compute_all(self.median_filter_size)
        stats["illum_correction_function"] = correct.illum_corr_func

        # Plate ready
        utils.logger.info("Plate " + self.name + " done")
        return stats


# COMPUTE INTENSITY STATISTICS IN A SINGLE PLATE

# TODO: try def calculate_stats(plate, config) DOES IT WORK???? I DUNNO
def calculate_statistics(args):
    # Load input parameters
    plate, config = args

    plate_name = plate.data["Metadata_Plate"].iloc[0]

    outfile = illum_stats_filename(config["paths"]["intensities"], plate_name)

    if os.path.isfile(outfile):
        print(outfile, "exists")
        return

    # Create Dataset object
    dset = deepprofiler.dataset.image_dataset.ImageDataset(
        plate,
        config["dataset"]["metadata"]["label_field"],
        config["dataset"]["images"]["channels"],
        config["paths"]["images"],
        lambda r: "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
    )

    # Prepare ImageStatistics object
    hist = IlluminationStatistics(
        config["dataset"]["images"]["bits"],
        config["dataset"]["images"]["channels"],
        config["prepare"]["illumination_correction"]["down_scale_factor"],
        config["prepare"]["illumination_correction"]["median_filter_size"],
        name=plate_name
    )

    hist.expected = dset.number_of_records("all")

    # Run the intensity computation
    dset.scan(hist.process_image, frame="all")

    # Retrieve and store results
    stats = hist.compute_stats()

    utils.check_path(outfile)

    with open(outfile, "wb") as output:
        pickle.dump(stats, output)
