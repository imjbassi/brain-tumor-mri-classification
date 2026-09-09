# visualizer.py
"""Dataset-inspection figures: a sample grid and the class distribution
bar chart referenced in the paper's Dataset section."""
import argparse
import os

import matplotlib.pyplot as plt
from PIL import Image

from data_loader import BrainMRIDataset
from utils import plot_class_distribution


def visualize_samples(data_dir, class_names=None, num_samples=4, out_path=None):
    """Displays (or saves) sample images from each class in the dataset."""
    if class_names is None:
        class_names = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))

    fig = plt.figure(figsize=(3 * num_samples, 3 * len(class_names)))
    img_count = 1

    for cls in class_names:
        cls_dir = os.path.join(data_dir, cls)
        images = [img for img in os.listdir(cls_dir)
                  if img.lower().endswith((".png", ".jpg", ".jpeg"))][:num_samples]

        for img_name in images:
            img = Image.open(os.path.join(cls_dir, img_name)).convert("RGB")
            plt.subplot(len(class_names), num_samples, img_count)
            plt.imshow(img)
            plt.title(cls)
            plt.axis("off")
            img_count += 1

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved sample grid to {out_path}")
    else:
        plt.show()


def summarize_class_distribution(data_dir, out_path="class_distribution.png"):
    """Counts images per class across Training/Testing and plots them."""
    counts_by_split = {}
    for split in ("Training", "Testing"):
        split_dir = os.path.join(data_dir, split)
        if os.path.isdir(split_dir):
            dataset = BrainMRIDataset(split_dir)
            counts_by_split[split] = dataset.class_counts()

    if not counts_by_split:
        raise FileNotFoundError(f"No Training/ or Testing/ subfolder found in {data_dir}.")

    plot_class_distribution(counts_by_split, out_path)
    return counts_by_split


def main():
    parser = argparse.ArgumentParser(description="Generate dataset-inspection figures.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the dataset root (containing Training/ and Testing/).")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--save", action="store_true",
                        help="Save figures to disk instead of opening a window.")
    parser.add_argument("--fig_dir", type=str, default="paper/figures",
                        help="Directory to save sample_grid.png / class_distribution.png in.")
    args = parser.parse_args()

    if args.save:
        os.makedirs(args.fig_dir, exist_ok=True)

    train_dir = os.path.join(args.data_dir, "Training")
    sample_dir = train_dir if os.path.isdir(train_dir) else args.data_dir
    visualize_samples(sample_dir, num_samples=args.num_samples,
                       out_path=os.path.join(args.fig_dir, "sample_grid.png") if args.save else None)
    summarize_class_distribution(args.data_dir, out_path=os.path.join(args.fig_dir, "class_distribution.png"))


if __name__ == "__main__":
    main()
