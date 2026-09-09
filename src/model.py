# model.py
import torch.nn as nn
import torchvision.models as models

_BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
}


class TumorClassifier(nn.Module):
    """Wraps a torchvision backbone pretrained on ImageNet and swaps its
    classification head for `num_classes` outputs.

    freeze_backbone=True keeps the pretrained feature extractor fixed and
    only trains the new head, which is much faster and works well when the
    dataset is small; the default (False) fine-tunes the whole network,
    which tends to perform better once there is enough data, per
    Tajbakhsh et al. (2016).
    """

    def __init__(self, num_classes=4, backbone="resnet18", freeze_backbone=False):
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"Unknown backbone '{backbone}'. Choose from {list(_BACKBONES)}.")

        ctor, weights = _BACKBONES[backbone]
        self.backbone_name = backbone
        self.model = ctor(weights=weights)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        if backbone.startswith("resnet"):
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
            self._head = self.model.fc
        else:  # efficientnet
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
            self._head = self.model.classifier[-1]

        # The new head is always trainable, even when the backbone is frozen.
        for param in self._head.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)

    def last_conv_layer(self):
        """Returns the final convolutional layer, used as the Grad-CAM target."""
        if self.backbone_name.startswith("resnet"):
            return self.model.layer4[-1]
        return self.model.features[-1]


def get_model(num_classes=4, backbone="resnet18", freeze_backbone=False):
    return TumorClassifier(num_classes=num_classes, backbone=backbone,
                            freeze_backbone=freeze_backbone)
