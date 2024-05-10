import glob
import os
import random
import pandas as pd
from PIL import Image
from tqdm import tqdm
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

random.seed(21)

DATA = "data"
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


def fix_image(path):
    with open(path, "rb") as f:
        check_chars = f.read()[-2:]
    if check_chars != b"\xff\xd9":
        im = Image.open(path)
        im = im.convert("RGB")
        im.save(path, "JPEG")


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
                if os.path.exists(img_path):
                    fix_image(img_path)
                    os.system(f"cp {img_path} {DATA}/{split}/{label}")


if __name__ == "__main__":
    build_folder("test", test)
    build_folder("val", val)
    build_folder("train", train)
