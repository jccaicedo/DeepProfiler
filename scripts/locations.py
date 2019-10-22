import argparse
import json
import logging
import multiprocessing
import os
import sqlite3
import sys

import pandas as pd


def check_path(filename):
    path = "/".join(filename.split("/")[0:-1])
    os.system("mkdir -p " + path)


class Logger():

    def __init__(self):
        self.root = logging.getLogger()
        self.root.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        self.root.addHandler(ch)

    def log(self, level, msg):
        self.root.log(level, msg)

    def info(self, msg):
        self.root.info(msg)


logger = Logger()


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

    def filterRecords(self, filteringRule, copy=False):
        if copy:
            newMeta = Metadata()
            newMeta.data = self.data.loc[filteringRule(self.data), :].copy()
            return newMeta
        else:
            self.data = self.data.loc[filteringRule(self.data), :]


## Generator of plates. Reads metadata and yields plates
def read_plates(metaFile):
    metadata = Metadata(metaFile)
    plates = metadata.data["Metadata_Plate"].unique()
    logger.info("Total plates: {}".format(len(plates)))
    for i in range(len(plates)):  # & (df.Metadata_Well == "a01")
        plate = metadata.filterRecords(lambda df: (df.Metadata_Plate == plates[i]), copy=True)
        yield plate
    return


def print_progress(iteration, total, prefix="Progress", suffix="Complete", decimals=1, barLength=100):
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
        sys.stdout.write("\rError: print_progress() function received a negative "
        iteration
        " value.")
        sys.stdout.flush()
    elif total < 0:
        sys.stdout.write("\rError: print_progress() function received a negative "
        total
        " value.")
        sys.stdout.flush()
    elif barLength < 0:
        sys.stdout.write("\rError: print_progress() function received a negative "
        barLength
        " value.")
        sys.stdout.flush()
    elif iteration > total:
        sys.stdout.write("\rError: print_progress() function received an "
        iteration
        " value greater than the "
        total
        " value.")
        sys.stdout.flush()


def write_locations(field, query_template, plate_name, row, conn, config):
    # Read cells file for each image
    query = query_template.replace("@@@", field).format(
        plate_name,
        row["Metadata_Well"],
        row["Metadata_Site"]
    )
    locations = pd.read_sql_query(query, conn)

    # Keep center coordinates only, remove NaNs, and transform to integers
    locations = locations.dropna(axis=0, how="any")
    locations[field + "_Location_Center_X"] = locations[field + "_Location_Center_X"] * config["compression"][
        "scaling_factor"]
    locations[field + "_Location_Center_Y"] = locations[field + "_Location_Center_Y"] * config["compression"][
        "scaling_factor"]
    locations[field + "_Location_Center_X"] = locations[field + "_Location_Center_X"].astype(int)
    locations[field + "_Location_Center_Y"] = locations[field + "_Location_Center_Y"].astype(int)

    # Save the resulting dataset frame in the output directory
    loc_file = "{}/{}/locations/{}-{}-{}.csv".format(
        config["compression"]["output_dir"],
        row["Metadata_Plate"],
        row["Metadata_Well"],
        row["Metadata_Site"],
        field
    )
    check_path(loc_file)
    locations.to_csv(loc_file, index=False)


def create_cell_indices(args):
    plate, config = args

    # Open database
    plate_name = plate.data.iloc[0]["Metadata_Plate"]
    database_file = "{}/{}/{}.sqlite".format(config["original_images"]["backend"], plate_name, plate_name)
    conn = sqlite3.connect(database_file)

    # Define query template: @@@ is either Cells or Nuclei
    query_template = "SELECT @@@_Location_Center_X, @@@_Location_Center_Y " + \
                     " FROM @@@ INNER JOIN Image " + \
                     "    ON Image.ImageNumber = @@@.ImageNumber " + \
                     "    AND Image.TableNumber = @@@.TableNumber " + \
                     " WHERE Image.Image_Metadata_Plate = '{}' " + \
                     "    AND Image.Image_Metadata_Well = '{}' " + \
                     "    AND Image.Image_Metadata_Site = '{}' " + \
                     "    AND @@@_Location_Center_X NOT LIKE 'NaN' " + \
                     "    AND @@@_Location_Center_Y NOT LIKE 'NaN' "

    # Extract cells and nuclei locations for each image
    iteration = 1
    for index, row in plate.data.iterrows():
        write_locations("Cells", query_template, plate_name, row, conn, config)
        write_locations("Nuclei", query_template, plate_name, row, conn, config)
        print_progress(iteration, len(plate.data), prefix="Locations in plate " + str(plate_name))
        iteration += 1
    print("")


class Parallel():

    def __init__(self, fixed_args, numProcs=None):
        self.fixed_args = fixed_args
        cpus = multiprocessing.cpu_count()
        if numProcs is None or numProcs > cpus or numProcs < 1:
            numProcs = cpus
        self.pool = multiprocessing.Pool(numProcs)

    def compute(self, operation, data):
        iterable = [[d, self.fixed_args] for d in data]
        self.pool.map(operation, iterable)
        return


parser = argparse.ArgumentParser(description="Find cell locations")
parser.add_argument("config", help="The path to the configuration file")
parser.add_argument("core", help="Number of CPU cores for parallel processing (all=0)")
options = parser.parse_args()

cores = int(options.core)
with open(options.config, "r") as f:
    config = json.load(f)
process = Parallel(config, numProcs=cores)
metadata = read_plates(config["metadata"]["path"] + config["metadata"]["filename"])
process.compute(create_cell_indices, metadata)
