import os

import numpy as np
import pandas as pd


# BOUNDING BOX HANDLING

def get_locations(image_key, config, randomize=True, seed=None):
    keys = image_key.split("/")
    locations_file = "{}/{}-{}.csv".format(
        keys[0],
        keys[1],
        config["train"]["sampling"]["locations_field"]
    )
    locations_path = os.path.join(config["paths"]["locations"], locations_file)
    if os.path.exists(locations_path):
        locations = pd.read_csv(locations_path)
        random_sample = config["train"]["sampling"]["locations"]
        if randomize and random_sample is not None and random_sample < len(locations):
            return locations.sample(random_sample, random_state=seed)
        else:
            return locations
    else:
        y_key = config["train"]["sampling"]["locations_field"] + "_Location_Center_Y"
        x_key = config["train"]["sampling"]["locations_field"] + "_Location_Center_X"
        return pd.DataFrame(columns=[x_key, y_key])


def load_batch(dataset, config):
    batch = dataset.get_train_batch(config["train"]["sampling"]["images"])
    batch["locations"] = [get_locations(x, config) for x in batch["keys"]]
    return batch


def prepare_boxes(batch, config):
    locations_batch = batch["locations"]
    image_targets = batch["targets"]
    images = batch["images"]
    all_boxes = []
    all_indices = []
    all_targets = [[] for _ in range(len(image_targets[0]))]
    all_masks = []
    index = 0
    y_key = config["train"]["sampling"]["locations_field"] + "_Location_Center_Y"
    x_key = config["train"]["sampling"]["locations_field"] + "_Location_Center_X"
    for locations in locations_batch:
        # Collect and normalize boxes between 0 and 1
        boxes = np.zeros((len(locations), 4), np.float32)
        boxes[:, 0] = locations[y_key] - config["train"]["sampling"]["box_size"] / 2
        boxes[:, 1] = locations[x_key] - config["train"]["sampling"]["box_size"] / 2
        boxes[:, 2] = locations[y_key] + config["train"]["sampling"]["box_size"] / 2
        boxes[:, 3] = locations[x_key] + config["train"]["sampling"]["box_size"] / 2
        boxes[:, [0, 2]] /= config["dataset"]["images"]["height"]
        boxes[:, [1, 3]] /= config["dataset"]["images"]["width"]
        # Create indicators for this set of boxes, belonging to the same image
        box_ind = index * np.ones((len(locations)), np.int32)
        # Propage the same labels to all crops
        for i in range(len(image_targets[index])):
            all_targets[i].append(image_targets[index][i] * np.ones((len(locations)), np.int32))
        # Identify object mask for each crop
        masks = np.zeros(len(locations), np.int32)
        if config["train"]["sampling"]["mask_objects"]:
            i = 0
            for lkey in locations.index:
                y = int(locations.loc[lkey, y_key])
                x = int(locations.loc[lkey, x_key])
                patch = images[index][max(y - 5, 0):y + 5, max(x - 5, 0):x + 5, -1]
                if np.size(patch) > 0:
                    masks[i] = int(np.median(patch))
                i += 1
        # Pile up the resulting variables
        all_boxes.append(boxes)
        all_indices.append(box_ind)
        all_masks.append(masks)
        index += 1

    result = (np.concatenate(all_boxes),
              np.concatenate(all_indices),
              [np.concatenate(t) for t in all_targets],
              np.concatenate(all_masks)
              )
    return result
