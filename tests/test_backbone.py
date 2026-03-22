"""Test backbone feature extraction."""

import torch

from neurosymbolic_da.nn.backbone import ResNetBackbone


def test_resnet18_output_shape():
    backbone = ResNetBackbone("resnet18", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    features = backbone(x)
    assert features.shape == (2, 512, 7, 7)


def test_resnet50_output_shape():
    backbone = ResNetBackbone("resnet50", pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    features = backbone(x)
    assert features.shape == (1, 2048, 7, 7)


def test_backbone_gradient_flow():
    backbone = ResNetBackbone("resnet18", pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    features = backbone(x)
    loss = features.sum()
    loss.backward()
    # Check that backbone parameters received gradients
    for p in backbone.parameters():
        if p.requires_grad:
            assert p.grad is not None
            break
