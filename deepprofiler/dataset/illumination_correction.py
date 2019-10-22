import numpy as np
import scipy.stats
import skimage.filters
import skimage.morphology
import skimage.transform


# ILLUMINATION CORRECTION FUNCTION


class IlluminationCorrection(object):
    def __init__(self, stats, channels, target_dim):
        self.stats = stats
        self.channels = channels
        self.target_dim = (target_dim[0], target_dim[1])
        self.illum_corr_func = np.zeros((self.target_dim[0], self.target_dim[1], len(self.channels)))

    # Based on Sing et al. 2014 paper
    def channel_function(self, mean_channel, disk_size):
        # TODO: get np.type from other source or parameterize or compute :/
        # We currently assume 16 bit images
        operator = skimage.morphology.disk(disk_size)
        filtered_channel = skimage.filters.median(mean_channel.astype(np.uint16), operator)
        filtered_channel = skimage.transform.resize(filtered_channel, self.target_dim, mode="reflect",
                                                    anti_aliasing=True, preserve_range=True)
        robust_minimum = scipy.stats.scoreatpercentile(filtered_channel, 2)
        filtered_channel = np.maximum(filtered_channel, robust_minimum)
        illum_corr_func = filtered_channel / robust_minimum
        return illum_corr_func

    def compute_all(self, median_filter_size):
        disk_size = median_filter_size / 2  # From diameter to radius
        for ch in range(len(self.channels)):
            self.illum_corr_func[:, :, ch] = self.channel_function(self.stats["mean_image"][:, :, ch], disk_size)

    # TODO: Is image a uint16 or float32/64? What is its data type?
    # TODO: Update the test as appropriate.
    # Not being used? Not needed?
    def apply(self, image):
        return image / self.illum_corr_func
