import glob
import os
import random
import pandas as pd

random.seed(42)

DATA = "sample_data"
folder = f"{DATA}/embryo_dataset"

subfolders = [f.name for f in os.scandir(folder) if f.is_dir()]
subfolders = random.sample(subfolders, len(subfolders))

train = subfolders[: int(0.7 * len(subfolders))]
val = subfolders[int(0.7 * len(subfolders)) : int(0.9 * len(subfolders))]
test = subfolders[int(0.9 * len(subfolders)) :]

with open(f"{DATA}/train.txt", "w") as f:
    for item in train:
        f.write("%s\n" % item)
with open(f"{DATA}/val.txt", "w") as f:
    for item in val:
        f.write("%s\n" % item)
with open(f"{DATA}/test.txt", "w") as f:
    for item in test:
        f.write("%s\n" % item)

from tqdm import tqdm


def build_folder(split, folders):
    print(f"Building {split} folders")
    if not os.path.exists(f"{DATA}/{split}"):
        os.makedirs(f"{DATA}/{split}")

    for embryo in tqdm(folders):
        annotations = pd.read_csv(
            f"{DATA}/embryo_dataset_annotations/{embryo}_phases.csv", header=None
        )
        one_jpeg = glob.glob(f"{DATA}/embryo_dataset/{embryo}/*.jpeg")[0]
        naming = one_jpeg.split("/")[-1].split("_")[:-1]
        naming = "_".join(naming)
        for row in annotations.iterrows():
            label = row[1][0]
            start_idx = row[1][1]
            end_idx = row[1][2]
            if not os.path.exists(f"{DATA}/{split}/{label}"):
                os.makedirs(f"{DATA}/{split}/{label}")
            for idx in range(start_idx, end_idx + 1):
                img_path = f"{DATA}/embryo_dataset/{embryo}/{naming}_RUN{idx}.jpeg"
                os.system(f"cp {img_path} {DATA}/{split}/{label}")


build_folder("train", train)
build_folder("val", val)
build_folder("test", test)
