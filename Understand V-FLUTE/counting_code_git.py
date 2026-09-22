import hashlib
import io
from functools import lru_cache

import pandas as pd
from datasets import load_dataset
from PIL import Image

# CHANGE THIS TO ADD YOUR PATHS
BASE_DIRS = {
    "train": "/home/andrea/Desktop/TRAIN_VFLUTE",
    "validation": "/home/andrea/Desktop/VALIDATION_VFLUTE",
    "test": "/home/andrea/Desktop/TEST_VFLUTE",
}
CSV_NAMES = {
    "train": "train_vflute.csv",
    "validation": "valid_vflute.csv",
    "test": "test_vflute.csv",
}


@lru_cache(maxsize=None)
def md5_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def csv_path(split):
    return f"{BASE_DIRS[split]}/{CSV_NAMES[split]}"


def image_path(split, image_name):
    return f"{BASE_DIRS[split]}/{image_name}"


def load_metaphor_split(ds, split):
    df = ds[split].to_pandas()
    print(f"Entire {split} set: {df.shape}")
    df = df[df["phenomenon"] == "metaphor"]
    print(f"{split} set with metaphors: {df.shape}")
    df = df[["image", "claim", "label"]]

    image_hash = df["image"].apply(lambda img: hashlib.md5(img["bytes"]).hexdigest())
    df = df.loc[~pd.concat([image_hash, df["claim"], df["label"]], axis=1).duplicated()]
    print(f"Metaphor {split} set without explanations: {df.shape}")
    return df


def extract_images_and_save_csv(df, split):
    rows = []
    for _, row in df.iterrows():
        image_name = row["image"]["path"]
        Image.open(io.BytesIO(row["image"]["bytes"])).save(image_path(split, image_name))
        rows.append((image_name, row["claim"], row["label"]))
    flat_df = pd.DataFrame(rows, columns=["image", "claim", "label"])
    flat_df.to_csv(csv_path(split), index=False)
    return flat_df


def load_split_csv(split):
    df = pd.read_csv(csv_path(split))
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


def group_images_by_hash(flat_df, split):
    images_by_hash = {}
    for image_name in flat_df["image"]:
        h = md5_hash(image_path(split, image_name))
        images_by_hash.setdefault(h, []).append(image_name)
    return images_by_hash


def count_i2t_instances(images_by_hash, flat_df):
    images_dic2 = {}
    for k in images_by_hash:
        # print(k)
        # print(images_dic[k])
        for i in images_by_hash[k]:
            i_i = flat_df[flat_df['image'] == i]
            # print(i_i)
            i_i_label = i_i['label'].item()
            if i_i_label == 'entailment':
                
                if k not in images_dic2:
                    images_dic2[k] = [i]
                else:
                    images_dic2[k].append(i)

    return len(images_dic2.keys())


def count_t2i_instances(df):
    return df[df["label"] == "entailment"]["claim"].nunique()


def count_visual_entailment(df, split):
    ve_dict = {}
    for _, row in df.iterrows():
        h = md5_hash(image_path(split, row["image"]["path"]))
        ve_dict.setdefault(h, set()).add((row["claim"], row["label"]))

    total = 0
    balanced = 0
    for pairs in ve_dict.values():
        entailment = {claim for claim, label in pairs if label == "entailment"}
        contradiction = {claim for claim, label in pairs if label == "contradiction"}
        total += len(entailment) + len(contradiction)
        balanced += len(entailment) * len(contradiction)
    return total, balanced


def process_split(ds, split):
    print()
    print("------------------------")
    print(split.upper())

    df = load_metaphor_split(ds, split)

    flat_df = load_split_csv(split)
    images_by_hash = group_images_by_hash(flat_df, split)
    print(f"Unique images: {len(images_by_hash)}")
    print(f"Unique metaphors: {df['claim'].nunique()}")

    print(f"I2T retrieval instances: {count_i2t_instances(images_by_hash, flat_df)}")
    print(f"T2I retrieval instances: {count_t2i_instances(df)}")

    total, balanced = count_visual_entailment(df, split)
    print(f"Visual entailment instances: {total}")
    print(f"Visual entailment balanced instances: {balanced}")


def main():
    ds = load_dataset("ColumbiaNLP/V-FLUTE")
    print(ds)
    for split in ("train", "validation", "test"):
        process_split(ds, split)


if __name__ == "__main__":
    main()
