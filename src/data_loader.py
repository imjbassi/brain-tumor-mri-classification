# data_loader.py
"""Dataset and DataLoader utilities for the brain MRI classification pipeline.

Expects the dataset laid out the way it ships from Mendeley/Kaggle:

    data_dir/
        Training/
            glioma/  meningioma/  notumor/  pituitary/  (folder names may vary)
        Testing/
            glioma/  meningioma/  notumor/  pituitary/

The Training split is further divided into train/validation with a
stratified split so every class is represented in both subsets. The Testing
split is kept fully held out for evaluate.py.
"""
import json
import os
from collections import Counter

from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(augment=False):
    """Eval transform is always deterministic; train transform can add
    light, label-preserving augmentation (a scan flipped or rotated a few
    degrees is still the same tumor class)."""
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    if not augment:
        return eval_transform, eval_transform

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, eval_transform


class BrainMRIDataset(Dataset):
    """Loads images from `root_dir/<class_name>/*.jpg` style folders."""

    def __init__(self, root_dir, class_to_idx=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        classes = sorted(d for d in os.listdir(root_dir)
                          if os.path.isdir(os.path.join(root_dir, d)))
        self.classes = classes
        self.class_to_idx = class_to_idx or {c: i for i, c in enumerate(classes)}

        self.image_paths = []
        self.labels = []
        for cls_name in classes:
            class_dir = os.path.join(root_dir, cls_name)
            for filename in os.listdir(class_dir):
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(class_dir, filename))
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def class_counts(self):
        idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        return {idx_to_class[i]: n for i, n in Counter(self.labels).items()}


def load_group_map(path):
    """Load a JSON mapping of image path -> group id.

    A "group" is whatever unit must not straddle the train/validation
    boundary: a patient ID for the tumor classes, and a near-duplicate
    cluster ID for `notumor`, which has no recoverable patient structure.

    Keys are re-indexed by basename. The map is written against the original
    `data/` tree, but derived splits (deduplicated, per-fold) copy images into
    other directories, so a path-relative key would silently fail to match
    there and every image would fall back to a singleton group, defeating the
    grouping without any error. Basenames are unique across all 7,023 images
    in this dataset (verified), so they key reliably across every derived
    layout.
    """
    with open(path) as f:
        raw = json.load(f)
    by_base = {}
    for k, v in raw.items():
        by_base[os.path.basename(k)] = str(v)
    if len(by_base) != len(raw):
        raise ValueError(
            f"group map basenames are not unique ({len(by_base)} of {len(raw)}); "
            "cannot key by basename for this dataset."
        )
    return by_base


def _group_for(path, group_map):
    """Group id for an image. Images absent from the map become their own
    singleton group, which is the conservative choice: an unmatched image
    can never pull a known group across the boundary."""
    key = os.path.basename(path)
    if key in group_map:
        return group_map[key]
    return f"__ungrouped__{key}"


def get_dataloaders(data_dir, batch_size=32, val_split=0.2, augment=False,
                     balance_classes=False, num_workers=4, seed=42,
                     group_map=None):
    """Build train/val/test DataLoaders from a Training/Testing directory pair.

    If `group_map` is given (path to JSON, or a dict), the train/validation
    split is made with GroupShuffleSplit so that no group straddles the
    boundary. Without it the split is stratified by class only, which means
    slices from one patient land on both sides. That is precisely the defect
    this project's own audit documents in the released dataset, and it
    silently corrupts model selection: the checkpoint, the LR schedule, early
    stopping, and any temperature fitted on the validation set are all chosen
    against data the model has effectively already seen. Pass a group map
    whenever one is available.

    Returns (train_loader, val_loader, test_loader, class_names).
    """
    train_dir = os.path.join(data_dir, "Training")
    test_dir = os.path.join(data_dir, "Testing")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Expected a 'Training' subfolder inside {data_dir}. "
            "See the README for the dataset layout."
        )

    train_transform, eval_transform = build_transforms(augment)

    full_train = BrainMRIDataset(train_dir, transform=train_transform)
    full_eval = BrainMRIDataset(train_dir, class_to_idx=full_train.class_to_idx,
                                 transform=eval_transform)
    class_names = full_train.classes

    if group_map is not None:
        if isinstance(group_map, str):
            group_map = load_group_map(group_map)
        groups = [_group_for(p, group_map) for p in full_train.image_paths]
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
        train_idx, val_idx = next(splitter.split(range(len(full_train)),
                                                  full_train.labels, groups))
        train_groups = {groups[i] for i in train_idx}
        val_groups = {groups[i] for i in val_idx}
        overlap = train_groups & val_groups
        assert not overlap, (
            f"{len(overlap)} group(s) appear in both train and validation; "
            "grouped splitting failed."
        )
    else:
        train_idx, val_idx = train_test_split(
            range(len(full_train)),
            test_size=val_split,
            stratify=full_train.labels,
            random_state=seed,
        )
    train_dataset = Subset(full_train, train_idx)
    val_dataset = Subset(full_eval, val_idx)

    if balance_classes:
        label_counts = Counter(full_train.labels[i] for i in train_idx)
        weights = [1.0 / label_counts[full_train.labels[i]] for i in train_idx]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                   sampler=sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=num_workers)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    test_loader = None
    if os.path.isdir(test_dir):
        test_dataset = BrainMRIDataset(test_dir, class_to_idx=full_train.class_to_idx,
                                        transform=eval_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_names
