# gradcam.py
"""Grad-CAM (Selvaraju et al., 2017) for the trained tumor classifier.

Produces one heatmap-overlaid example per class so a reader can check
whether the network is actually attending to the tumor region rather than
to scan borders, text overlays, or other spurious cues.
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data_loader import IMAGENET_MEAN, IMAGENET_STD, build_transforms
from model import get_model
from utils import load_checkpoint, plot_image_grid


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, image_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(image_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def overlay_heatmap(image, cam, alpha=0.4):
    """Blend a normalized [0,1] Grad-CAM map onto the original PIL image using
    a simple red-channel heatmap (keeps this dependency-free, no cv2/matplotlib colormap needed)."""
    image = np.array(image.resize((224, 224))).astype(float) / 255.0
    heatmap = np.stack([cam, np.zeros_like(cam), 1 - cam], axis=-1)
    blended = (1 - alpha) * image + alpha * heatmap
    return np.clip(blended, 0, 1)


def find_one_example_per_class(data_dir, class_names):
    test_dir = os.path.join(data_dir, "Testing")
    search_dir = test_dir if os.path.isdir(test_dir) else os.path.join(data_dir, "Training")
    examples = {}
    for cls in class_names:
        matches = glob.glob(os.path.join(search_dir, cls, "*"))
        if matches:
            examples[cls] = matches[0]
    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM overlays for one example per class.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--fig_dir", type=str, default="paper/figures",
                        help="Directory to save gradcam_examples.png in.")
    args = parser.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)
    output_path = os.path.join(args.fig_dir, "gradcam_examples.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.model_path, map_location=device)
    class_names = ckpt["class_names"]

    model = get_model(num_classes=len(class_names), backbone=ckpt["model_name"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    cam_extractor = GradCAM(model, model.last_conv_layer())
    _, eval_transform = build_transforms(augment=False)

    examples = find_one_example_per_class(args.data_dir, class_names)
    overlays, titles = [], []
    for cls, path in examples.items():
        image = Image.open(path).convert("RGB")
        tensor = eval_transform(image).unsqueeze(0).to(device)
        cam, pred_idx = cam_extractor(tensor, class_idx=None)
        overlays.append(overlay_heatmap(image, cam))
        titles.append(f"true: {cls}\npred: {class_names[pred_idx]}")

    plot_image_grid(overlays, titles, output_path, ncols=len(overlays) or 1,
                     suptitle="Grad-CAM: Regions Driving Each Prediction")


if __name__ == "__main__":
    main()
