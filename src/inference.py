# inference.py
import argparse

import torch
import torch.nn.functional as F

from data_loader import build_transforms
from model import get_model
from utils import load_checkpoint


def predict(model, device, image_path, transform, class_names, top_k=3):
    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1).squeeze(0)

    top_probs, top_idx = probs.topk(min(top_k, len(class_names)))
    ranked = [(class_names[i], p.item()) for p, i in zip(top_probs, top_idx)]
    return ranked


def main():
    parser = argparse.ArgumentParser(description="Classify a brain tumor MRI image.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the checkpoint saved by train.py.")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.model_path, map_location=device)
    class_names = ckpt["class_names"]

    model = get_model(num_classes=len(class_names), backbone=ckpt["model_name"]).to(device)
    model.load_state_dict(ckpt["model_state"])

    _, eval_transform = build_transforms(augment=False)
    ranked = predict(model, device, args.image_path, eval_transform, class_names, args.top_k)

    print(f"Predicted class: {ranked[0][0]} ({ranked[0][1]:.1%} confidence)")
    if len(ranked) > 1:
        print("Runner-up predictions:")
        for cls, prob in ranked[1:]:
            print(f"  {cls}: {prob:.1%}")


if __name__ == "__main__":
    main()
