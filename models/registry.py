"""
Model factory. Every model has pretrained weights and a new linear head; SegFormer and YOLO11s-cls come from
models/external.py, all others from timm.

  get_model('convnext_tiny', num_classes=8)        # short name from BASELINES
  get_model('timm:efficientnet_b4.ra2_in1k', 8)     # any other timm model, by its full name

A proposed model goes into models/<name>.py and is added to PROPOSED (name -> builder(num_classes, pretrained,
img_size, **kw)); the rest of the pipeline does not change.
"""
import timm

# short name -> (timm name, model category, display name). Categories group the models in the paper tables and figures.
BASELINES = {
    # classic CNNs (still the most common baselines in plant-disease papers)
    'vgg16': ('vgg16_bn.tv_in1k', 'cnn', 'VGG16-BN'),
    'resnet50': ('resnet50.a1_in1k', 'cnn', 'ResNet-50'),
    'densenet121': ('densenet121.tv_in1k', 'cnn', 'DenseNet-121'),
    'efficientnet_b0': ('efficientnet_b0.ra_in1k', 'cnn', 'EfficientNet-B0'),
    'efficientnet_b3': ('efficientnet_b3.ra2_in1k', 'cnn', 'EfficientNet-B3'),
    'efficientnetv2_s': ('tf_efficientnetv2_s.in21k_ft_in1k', 'cnn', 'EfficientNetV2-S'),
    'convnext_tiny': ('convnext_tiny.fb_in22k_ft_in1k', 'cnn', 'ConvNeXt-T'),
    'convnextv2_tiny': ('convnextv2_tiny.fcmae_ft_in22k_in1k', 'cnn', 'ConvNeXtV2-T'),
    # lightweight (field / mobile deployment)
    'mobilenetv2': ('mobilenetv2_100.ra_in1k', 'light', 'MobileNetV2'),
    'mobilenetv3_large': ('mobilenetv3_large_100.ra_in1k', 'light', 'MobileNetV3-L'),
    'mobilevit_s': ('mobilevit_s.cvnets_in1k', 'light', 'MobileViT-S'),
    'efficientvit_b1': ('efficientvit_b1.r224_in1k', 'light', 'EfficientViT-B1'),
    # transformers / hybrids
    'vit_b16': ('vit_base_patch16_224.augreg2_in21k_ft_in1k', 'transformer', 'ViT-B/16'),
    'deit_s': ('deit_small_patch16_224.fb_in1k', 'transformer', 'DeiT-S'),
    'swin_t': ('swin_tiny_patch4_window7_224.ms_in1k', 'transformer', 'Swin-T'),
    'maxvit_t': ('maxvit_tiny_tf_224.in1k', 'hybrid', 'MaxViT-T'),
    'caformer_s18': ('caformer_s18.sail_in1k', 'transformer', 'CAFormer-S18'),
    # further benchmark models
    'mobilenetv4_m': ('mobilenetv4_conv_medium.e500_r224_in1k', 'light', 'MobileNetV4-M'),
    'mambaout_t': ('mambaout_tiny.in1k', 'mamba', 'MambaOut-T'),
    'dinov2_s': ('vit_small_patch14_dinov2.lvd142m', 'foundation', 'DINOv2 ViT-S/14'),
}
# not in timm: short name -> (builder, model category, display name); see models/external.py
EXTERNAL = {
    'segformer_b2': ('segformer', 'segmentation', 'SegFormer MiT-B2'),
    'yolo11s_cls': ('yolo', 'yolo', 'YOLO11s-cls'),
}
# The 12 models of the paper: 1-2 per model category, pretrained weights, 224 px.
DEFAULT_BENCHMARK = ['resnet50', 'efficientnetv2_s', 'convnext_tiny', 'mobilenetv4_m', 'efficientvit_b1',
                     'vit_b16', 'swin_t', 'maxvit_t', 'mambaout_t', 'dinov2_s', 'segformer_b2', 'yolo11s_cls']
PROPOSED = {}


def normalize_name(name):
    return name.lower().replace('-', '_')


def display_name(name):
    n = normalize_name(name)
    if n in BASELINES:
        return BASELINES[n][2]
    if n in EXTERNAL:
        return EXTERNAL[n][2]
    return n[5:] if n.startswith('timm:') else n


def family(name):
    n = normalize_name(name)
    if n in BASELINES:
        return BASELINES[n][1]
    if n in EXTERNAL:
        return EXTERNAL[n][1]
    return 'proposed' if n in PROPOSED else 'other'


def get_model(name, num_classes, pretrained=True, img_size=224, drop_rate=0.0, drop_path_rate=0.0, **kw):
    n = normalize_name(name)
    if n in PROPOSED:
        return PROPOSED[n](num_classes=num_classes, pretrained=pretrained, img_size=img_size,
                           drop_rate=drop_rate, drop_path_rate=drop_path_rate, **kw)
    if n in EXTERNAL:
        from .external import SegFormerClassifier, YOLOClassifier
        builder = {'segformer': SegFormerClassifier, 'yolo': YOLOClassifier}[EXTERNAL[n][0]]
        return builder(num_classes=num_classes, pretrained=pretrained, drop_rate=drop_rate)
    if n in BASELINES:
        timm_name = BASELINES[n][0]
    elif n.startswith('timm:'):
        timm_name = name[5:]
    else:
        raise ValueError(f"unknown model {name}; options: {sorted(BASELINES) + sorted(EXTERNAL) + sorted(PROPOSED)} "
                         "or timm:<name>")
    args = dict(pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
    if drop_path_rate > 0:
        args['drop_path_rate'] = drop_path_rate
    # Transformers with fixed position embeddings / windows need the input size at build time (timm resamples
    # the pretrained embeddings). CNNs do not accept img_size.
    # Some timm models (e.g. EfficientViT-B1) implement no stochastic depth and reject drop_path_rate: built without it.
    try:
        model = timm.create_model(timm_name, img_size=img_size, **args)
    except TypeError:
        try:
            model = timm.create_model(timm_name, **args)
        except TypeError as e:
            if 'drop_path_rate' not in str(e):
                raise
            args.pop('drop_path_rate')
            print(f"[MODEL] {timm_name} has no stochastic depth -> built without drop_path_rate")
            model = timm.create_model(timm_name, **args)
    model.timm_name = timm_name
    return model


def data_config(model):
    """mean / std the pretrained weights expect (e.g. 0.5/0.5 for some ViTs instead of the ImageNet values)."""
    cfg = getattr(model, 'pretrained_cfg', None) or {}
    return tuple(cfg.get('mean', (0.485, 0.456, 0.406))), tuple(cfg.get('std', (0.229, 0.224, 0.225)))


def head_param_names(model):
    """Names of the classifier parameters (trained with the full LR; the pretrained rest gets enc_lr_mult)."""
    if hasattr(model, 'get_classifier'):
        head = model.get_classifier()
        ids = {id(p) for p in head.parameters()} if hasattr(head, 'parameters') else set()
        return {n for n, p in model.named_parameters() if id(p) in ids}
    return {n for n, _ in model.named_parameters() if n.startswith(('head', 'fc', 'classifier'))}
