import numpy as np
import skimage.io
import skimage.measure


# COMMON IMAGE HANDLING OPPERATIONS

# Main image reading function. Images are treated as 3D tensors: (height, width, channels)
def open_image(paths, outlines):
    channels = [skimage.io.imread(p) for p in paths]
    img = np.zeros((channels[0].shape[0], channels[0].shape[1], len(channels)))
    for c in range(len(channels)):
        # TODO: dividing here is a terrible idea for illumination correction!
        # img[:,:,c] = (channels[c] -128.0) / 128.0
        # max_value =  max(np.max(channels[c]), 1.0)/2
        # img[:,:,c] = (channels[c] - max_value) / max_value
        img[:, :, c] = channels[c]
    if outlines is not None:
        boundaries = skimage.io.imread(outlines)
        labels = skimage.measure.label(boundaries, background=1)
        img = np.concatenate((img, labels[:, :, np.newaxis]), axis=2)
    return img
