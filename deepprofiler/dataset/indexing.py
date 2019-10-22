import pandas as pd

import deepprofiler.dataset.metadata


def write_compression_index(config):
    metadata = deepprofiler.dataset.metadata.Metadata(config["paths"]["index"], dtype=None)
    new_index = metadata.data
    for ch in config["dataset"]["images"]["channels"]:
        new_index[ch] = new_index[ch].str.split("/").str[-1]
        new_index[ch] = new_index["Metadata_Plate"].astype(str) + new_index[ch].map(lambda x: "/" + x.replace("." + config["dataset"]["images"]["file_format"], ".png"))
    new_index.to_csv(config["paths"]["compressed_metadata"] + "/compressed.csv")


# Split a metadata file in a number of parts
def split_index(config, parts):
    index = pd.read_csv(config["paths"]["metadata"] + "/index.csv")
    plate_wells = index.groupby(["Metadata_Plate", "Metadata_Well"]).count()["Metadata_Site"]
    plate_wells = plate_wells.reset_index().drop(["Metadata_Site"], axis=1)
    part_size = int(len(plate_wells) / parts)
    for i in range(parts):
        if i < parts - 1:
            df = plate_wells[i * part_size:(i + 1) * part_size]
        else:
            df = plate_wells[i * part_size:]
        df = pd.merge(index, df, on=["Metadata_Plate", "Metadata_Well"])
        df.to_csv(config["paths"]["metadata"] + "/index-{0:03d}.csv".format(i), index=False)
    print("All set")
