import argparse

import numpy as np
import pandas as pd
from rich.console import Console

from dataset import DataLoader
from model import build_model


def parse_args():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Build a Keras model")

    # Add the arguments
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/base_model.h5",
        help="The path to the model checkpoint",
    )
    parser.add_argument(
        "--mls",
        type=bool,
        default=True,
    )

    # Parse the arguments
    args = parser.parse_args()

    return args


def mls():
    model = build_model(checkpoint_path="checkpoints/base_model.h5")
    data_loader = DataLoader(
        csv_path="datasets/nih/train.csv",
        image_dir="datasets/nih/images",
        batch_size=1,
    )
    dataset = data_loader.get_dataset()

    # Get logits for each image and calculate max logits
    logits_list = []
    for batch in dataset:
        images, image_indices, _ = batch
        logits = model.predict(x=images, verbose=0)[0]
        max_logits = np.max(logits)
        # Decode as string
        image_index_str = image_indices.numpy()[0].decode("utf-8")
        logits_list.append([image_index_str, max_logits])

    # Save max logits to CSV
    df = pd.DataFrame(logits_list, columns=["Image Index", "max_logit_score"])
    csv_path = "max_logits_output.csv"
    df.to_csv(csv_path, index=False)

    print(f"Max logits CSV saved to {csv_path}")


def main(args):
    console = Console()

    if args.mls:
        console.print("MLS Mode!", style="bold red")
        mls()

    else:
        raise NotImplementedError("Not implemented yet")


if __name__ == "__main__":
    args = parse_args()
    main(args)
