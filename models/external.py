"""
Benchmark models that are not in timm, wrapped with the interface the pipeline uses for timm models:
forward / forward_features (feature map, NCHW) / forward_head / get_classifier / num_features / pretrained_cfg
(mean, std of the pretrained weights). Grad-CAM uses forward_features -> forward_head (utils/gradcam.py).

  SegFormer MiT-B2 (HF transformers, nvidia/mit-b2: ImageNet-1k classification weights of the SegFormer encoder);
      head as in SegformerForImageClassification: mean over the last feature map + linear layer.
  YOLO11s-cls (Ultralytics, yolo11s-cls.pt, ImageNet-1k): backbone + Classify head (1x1 conv to 1280 channels,
      average pooling, linear layer), trained in our loop like every other model. Ultralytics feeds images in [0, 1]
      without normalisation -> mean 0, std 1.

Weights are downloaded once (python tools/check_models.py; HF cache in $HF_HOME, YOLO file in $TORCH_HOME/ultralytics/);
the training jobs run offline.
"""
import os

import torch
import torch.nn as nn

IMAGENET = {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)}


class SegFormerClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True, drop_rate=0.0, hf_name='nvidia/mit-b2', **_):
        super().__init__()
        from transformers import SegformerConfig, SegformerModel
        self.encoder = (SegformerModel.from_pretrained(hf_name) if pretrained else
                        SegformerModel(SegformerConfig.from_pretrained(hf_name)))
        self.num_features = self.encoder.config.hidden_sizes[-1]
        self.drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.num_features, num_classes)
        self.pretrained_cfg = dict(IMAGENET, hf_name=hf_name)

    def forward_features(self, x):
        return self.encoder(pixel_values=x).last_hidden_state          # (B, C, H/32, W/32)

    def forward_head(self, f):
        return self.head(self.drop(f.mean(dim=(2, 3))))

    def forward(self, x):
        return self.forward_head(self.forward_features(x))

    def get_classifier(self):
        return self.head


def yolo_weights(name='yolo11s-cls.pt'):
    return os.path.join(os.environ.get('TORCH_HOME', os.path.expanduser('~/.cache/torch')), 'ultralytics', name)


class YOLOClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True, drop_rate=0.0, cfg='yolo11s-cls', **_):
        super().__init__()
        from ultralytics.nn.tasks import ClassificationModel
        if pretrained:
            path = yolo_weights(f'{cfg}.pt')
            if not os.path.exists(path):
                raise FileNotFoundError(f'{path} not found: download yolo11s-cls.pt first (see README)')
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            net = (ckpt['ema'] if ckpt.get('ema') is not None else ckpt['model']).float()
        else:
            net = ClassificationModel(f'{cfg}.yaml', nc=1000, verbose=False)
        layers = list(net.model)
        cls = layers[-1]                                    # Ultralytics Classify: conv -> pool -> drop -> linear
        assert all(getattr(m, 'f', -1) == -1 for m in layers), 'YOLO classification backbone must be sequential'
        self.body = nn.Sequential(*layers[:-1])
        self.conv, self.pool = cls.conv, cls.pool
        self.num_features = cls.linear.in_features
        self.drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.num_features, num_classes)
        for p in self.parameters():
            p.requires_grad_(True)                          # Ultralytics checkpoints are saved with requires_grad off
        self.pretrained_cfg = {'mean': (0.0, 0.0, 0.0), 'std': (1.0, 1.0, 1.0), 'file': f'{cfg}.pt'}

    def forward_features(self, x):
        return self.conv(self.body(x))                      # (B, 1280, H/32, W/32)

    def forward_head(self, f):
        return self.head(self.drop(self.pool(f).flatten(1)))

    def forward(self, x):
        return self.forward_head(self.forward_features(x))

    def get_classifier(self):
        return self.head
