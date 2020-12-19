import os
import shutil
import argparse
from glob import glob
import pandas as pd

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--librimix_dir", type=str, default=None, help="Path to librispeech root directory"
)
parser.add_argument(
    "--metadata_old_root", type=str, default=None, help="Old root in metadata, specified to change to new root"
)


def main(args):
    librimix_dir = args.librimix_dir
    metadata_old_root = args.metadata_old_root
    create_local_metadata(librimix_dir, metadata_old_root)


def create_local_metadata(librimix_dir, metadata_old_root):

    corpus = librimix_dir.split("/")[-1]
    md_dirs = [f for f in glob(os.path.join(librimix_dir, "*/*/*")) if f.endswith("metadata")]
    for md_dir in md_dirs:
        md_files = [f for f in os.listdir(md_dir) if f.startswith("mix")]
        for md_file in md_files:
            print(md_dir, md_file)
            subset = md_file.split("_")[1]
            local_path = os.path.join(
                "data/librimix", os.path.relpath(md_dir, librimix_dir), subset
            ).replace("/metadata", "")
            os.makedirs(local_path, exist_ok=True)
            if metadata_old_root is None:
                shutil.copy(os.path.join(md_dir, md_file), local_path)
            else:
                data = pd.read_csv(os.path.join(md_dir, md_file))
                for key in data.keys():
                    if "path" in key:
                        data[key] = data[key].str.replace(metadata_old_root, librimix_dir)
                data.to_csv(os.path.join(local_path, md_file), index=0)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
