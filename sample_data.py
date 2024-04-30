import glob
import os
import random

folders = glob.glob("data/embryo_dataset/*")
sample_folders = random.sample(folders, 10)

sample_images = []
for folder in sample_folders:
    images = glob.glob(os.path.join(folder, "*.jpeg"))
    sample = random.sample(images, 10)
    sample_images.extend(sample)

# print(sample_images)

sample_annotations = []
for folder in sample_folders:
    annotation_file = f"/users/rkrish16/data/rkrish16/other/BeforeTheyHatch/data/embryo_dataset_annotations/{folder.split('/')[-1]}_phases.csv"
    sample_annotations.append(annotation_file)

# print(sample_annotations)

if not os.path.exists("sample_data"):
    os.makedirs("sample_data")

if not os.path.exists("sample_data/embryo_dataset"):
    os.makedirs("sample_data/embryo_dataset")

for image in sample_images:
    folder = image.split("/")[-2]
    if not os.path.exists(f"sample_data/embryo_dataset/{folder}"):
        os.makedirs(f"sample_data/embryo_dataset/{folder}")
    os.system(f"cp {image} sample_data/embryo_dataset/{folder}")

if not os.path.exists("sample_data/embryo_dataset_annotations"):
    os.makedirs("sample_data/embryo_dataset_annotations")

for annotation in sample_annotations:
    os.system(f"cp {annotation} sample_data/embryo_dataset_annotations")
