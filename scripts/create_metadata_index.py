import argparse
import json
import os
import sys

import pandas as pd


def relative_paths(df, target, path, filename, root):
    df[target] = df[path].str.replace(root, "") + "/" + df[filename]
    return df.drop([path, filename], axis=1)


def print_progress(iteration, total, prefix="Progress", suffix="Complete", decimals=1, barLength=50):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    if all(t >= 0 for t in [iteration, total, barLength]) and iteration <= total:
        formatStr = "{0:." + str(decimals) + "f}"
        percents = formatStr.format(100 * (iteration / float(total)))
        filledLength = int(round(barLength * iteration / float(total)))
        bar = "#" * filledLength + "-" * (barLength - filledLength)
        sys.stdout.write("\r%s |%s| %s%s %s" % (prefix, bar, percents, "%", suffix)),
        sys.stdout.flush()
        if iteration == total:
            sys.stdout.write("\n")
            sys.stdout.flush()
    elif sum([iteration < 0, total < 0, barLength < 0]) > 1:
        sys.stdout.write("\rError: print_progress() function received multiple negative values.")
        sys.stdout.flush()
    elif iteration < 0:
        sys.stdout.write("\rError: print_progress() function received a negative 'iteration' value.")
        sys.stdout.flush()
    elif total < 0:
        sys.stdout.write("\rError: print_progress() function received a negative 'total' value.")
        sys.stdout.flush()
    elif barLength < 0:
        sys.stdout.write("\rError: print_progress() function received a negative 'barLength' value.")
        sys.stdout.flush()
    elif iteration > total:
        sys.stdout.write(
            "\rError: print_progress() function received an 'iteration' value greater than the 'total' value.")
        sys.stdout.flush()


def parse_delimiter(delimiter):
    if delimiter == "blanks":
        return "\s+"
    elif delimiter == "tabs":
        return "\t"
    else:
        return ","


class Metadata():

    # The dtype argument indicates whether the dataset should be read as strings (object)
    # or according to the dataset type (None)
    def __init__(self, filename=None, csvMode="single", delimiter="default", dtype=object):
        if filename is not None:
            if csvMode == "single":
                self.loadSingle(filename, delimiter, dtype)
            elif csvMode == "multi":
                self.loadMultiple(filename, delimiter, dtype)

    def loadSingle(self, filename, delim, dtype):
        print("Reading metadata form", filename)
        delimiter = parse_delimiter(delim)
        # Read csv files as strings without dropping NA symbols
        self.data = pd.read_csv(filename, delimiter, dtype=dtype, keep_default_na=False)

    def loadMultiple(self, filename, delim, dtype):
        frames = []
        delimiter = parse_delimiter(delim)
        with open(filename, "r") as filelist:
            for line in filelist:
                csvPath = line.replace("\n", "")
                print("Reading from", csvPath)
                frames.append(pd.read_csv(csvPath, delimiter, dtype=dtype, keep_default_na=False))
        self.data = pd.concat(frames)
        print("Multiple CSV files loaded")


parser = argparse.ArgumentParser(description="Create metadata index")
parser.add_argument("config", help="The path to the configuration file")
options = parser.parse_args()

assert os.path.exists(options.config)

with open(options.config, "r") as f:
    config = json.load(f)

# Load plate maps dataset and create labels
plate_maps = Metadata(
    config["metadata"]["path"] + config["metadata"]["plate_maps"],
    "multi",
    config["metadata"]["platemap_separator"]
)

maps = plate_maps.data
treatment_name = config["metadata"]["treatment_name"]
field1 = config["metadata"]["treatment_columns"][0]
field2 = config["metadata"]["treatment_columns"][1]
maps[treatment_name] = maps[field1].astype(str) + "@" + maps[field2].astype(str)

label_values = maps[treatment_name].unique()

print("Unique {}: {}".format(treatment_name, len(label_values)))
for i in range(len(label_values)):
    maps.loc[lambda df: df[treatment_name] == label_values[i], treatment_name] = i
    print_progress(i + 1, len(label_values), prefix=treatment_name)

# Load barcodes and csv files
barcodes = Metadata(config["metadata"]["barcode_file"], "single")
load_data = Metadata(config["metadata"]["path"] + config["metadata"]["csv_list"], "multi")

# Merge two frames: csvs + barcodes to attach labels to each image
columns = list(load_data.data.columns.values)
metadata = pd.merge(
    load_data.data.drop(columns[13:], axis=1),
    barcodes.data,
    left_on=["Metadata_Plate"],
    right_on=["Assay_Plate_Barcode"],
    how="inner"
)
del load_data, barcodes

# Concatenate paths and filenames and make them relative
for ch in config["metadata"]["channels"]:
    metadata = relative_paths(
        metadata,
        ch,
        "PathName_Orig" + ch,
        "FileName_Orig" + ch,
        config["metadata"]["image_path"]
    )
print(metadata.info())

# Merge two frames: metadata + plate_maps to attach treatment info to each image
metadata = pd.merge(
    metadata,
    maps,
    left_on=["Plate_Map_Name", "Metadata_Well"],
    right_on=["plate_map_name", "well_position"],
    how="left"
)

# Remove unnecessary columns from the index
required_columns = ["Metadata_Plate", "Metadata_Well", "Metadata_Site", "Assay_Plate_Barcode", "Plate_Map_Name"]
required_columns += config["metadata"]["channels"] + [treatment_name]
available_columns = list(metadata.columns.values)
columns_to_remove = [c for c in available_columns if c not in required_columns]
metadata = metadata.drop(columns_to_remove, axis=1)

# Find replicates
replicate_field = treatment_name + "_Replicate"
metadata["plate_well"] = metadata["Metadata_Plate"].astype(str) + "::" + metadata["Metadata_Well"].astype(str)
metadata[replicate_field] = 0
replicate_distribution = {}
for i in range(len(label_values)):
    mask1 = metadata[treatment_name] == i
    wells = metadata[mask1]["plate_well"].unique()
    print_progress(i + 1, len(label_values), "Replicates")
    replicate = 1
    for j in range(len(wells)):
        mask2 = metadata["plate_well"] == wells[j]
        metadata.loc[mask1 & mask2, replicate_field] = replicate
        replicate += 1
    try:
        replicate_distribution[replicate - 1] += 1
    except:
        replicate_distribution[replicate - 1] = 1
metadata = metadata.drop(["plate_well"], axis=1)
print(replicate_distribution)

# Save resulting metadata
metadata.to_csv(config["metadata"]["path"] + "metadata.csv", index=False)
dframe = pd.DataFrame({"ID": pd.Series(range(len(label_values))), "Treatment": pd.Series(label_values)})
dframe.to_csv(config["metadata"]["path"] + "labels.csv", index=False)
