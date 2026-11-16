"""
Grad-CAM for every model in the registry, without per-architecture target layers, plus the leaf-focus score.

Where the class activation map comes from
  * CNNs, hybrids, Swin, MambaOut, ViTs with average pooling: the output of forward_features (the last feature
    map before pooling + classifier). Layouts handled: NCHW (ResNet, ConvNeXt, MaxViT), NHWC (Swin, MambaOut),
    tokens (B, N, C).
  * ViTs that classify from the class token (ViT, DeiT, DINOv2): the patch tokens of the last block's output do
    not reach the class token any more (zero gradient), so the map is taken at the input of the last block's
    attention (blocks[-1].norm1), as is common practice for ViT Grad-CAM.
  * A model can name its own layer with a `cam_target` attribute (module whose output is the feature map).

Leaf-focus score (does the model look at the leaf or at the background?)
  The leaf mask is the same as in tools/prepare_data.py: pixels that differ from the median border colour.
  It is only defined for images with a uniform background (at least 60 % of the border has the paper colour); field photos
  are skipped. Per image: share of the CAM energy inside the leaf mask, and that share divided by the leaf's area
  share (focus ratio: 1 = no preference, > 1 = the model concentrates on the leaf).
  Computed three times: on the Grad-CAM map (keys leaf_*), the Grad-CAM++ map (keys pp_leaf_*, from the same
  forward/backward pass) and an occlusion map (keys occ_leaf_*). Grad-CAM(++) depends on
  the layer it is taken from (class-token ViTs give noisy maps with background artefacts), occlusion is the same
  procedure for every architecture -> the occlusion score is the one compared across models in the paper.
"""
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

LEAF_THR = 45.0          # colour distance to the background (same value as tools/prepare_data.py)
BORDER_UNIFORM_MIN = 0.6  # background counts as uniform if >= 60 % of the border pixels have the paper colour
MASK_AREA = (0.02, 0.90)


def _cls_token_vit(model):
    return hasattr(model, 'blocks') and getattr(model, 'num_prefix_tokens', 0) > 0 and \
        getattr(model, 'global_pool', '') in ('token', 'map') and hasattr(model.blocks[-1], 'norm1')


def _to_map(t, model):
    """(B, C, H, W) view of a feature tensor in any of the supported layouts."""
    C = getattr(model, 'num_features', None)
    if t.ndim == 4:
        if C is not None and t.shape[-1] == C and t.shape[1] != C:
            return t.permute(0, 3, 1, 2)                     # NHWC
        return t
    if t.ndim == 3:                                          # tokens
        n_prefix = getattr(model, 'num_prefix_tokens', 0)
        tokens = t[:, n_prefix:]
        s = int(round(math.sqrt(tokens.shape[1])))
        if s * s != tokens.shape[1]:
            raise ValueError(f'cannot reshape {tokens.shape[1]} tokens to a square map')
        return tokens.transpose(1, 2).reshape(t.shape[0], t.shape[2], s, s)
    raise ValueError(f'unsupported feature shape {tuple(t.shape)}')


class GradCAM:
    def __init__(self, model):
        self.model = model.eval()
        self.target = getattr(model, 'cam_target', None)
        if self.target is None and _cls_token_vit(model):
            self.target = model.blocks[-1].norm1
        self.method = 'hook:' + type(self.target).__name__ if self.target is not None else 'forward_features'
        # timm attention: plain matmul attention instead of the fused kernel while computing gradients (same values;
        # the backward pass through the fused kernel was much slower in a test run). Restored by close().
        self._fused = [(m, m.fused_attn) for m in model.modules() if isinstance(getattr(m, 'fused_attn', None), bool)]
        for m, _ in self._fused:
            m.fused_attn = False

    def close(self):
        for m, v in self._fused:
            m.fused_attn = v

    def __call__(self, x, classes=None):
        """x (B,3,H,W) normalised images -> Grad-CAM (B,H,W), Grad-CAM++ (B,H,W), both in [0,1], logits (B,K).
        classes=None: predicted class."""
        store = {}
        if self.target is not None:
            def hook(_, __, out):
                store['act'] = out
            h = self.target.register_forward_hook(hook)
            try:
                with torch.enable_grad():
                    logits = self.model(x)
                    act = store['act']
                    cls = logits.argmax(1) if classes is None else classes
                    grad = torch.autograd.grad(logits.gather(1, cls[:, None]).sum(), act)[0]
            finally:
                h.remove()
        else:
            with torch.enable_grad():
                act = self.model.forward_features(x)
                logits = self.model.forward_head(act)
                cls = logits.argmax(1) if classes is None else classes
                grad = torch.autograd.grad(logits.gather(1, cls[:, None]).sum(), act)[0]
        a, g = _to_map(act.detach().float(), self.model), _to_map(grad.float(), self.model)
        cam = F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1, keepdim=True))
        # Grad-CAM++ (Chattopadhay et al. 2018), same activations and gradients, closed-form alpha of the paper
        # (higher-order derivatives from the exponential-score assumption, as in the reference implementation)
        g2 = g.pow(2)
        denom = 2 * g2 + a.sum(dim=(2, 3), keepdim=True) * g.pow(3)
        alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
        campp = F.relu(((alpha * F.relu(g)).sum(dim=(2, 3), keepdim=True) * a).sum(1, keepdim=True))
        return self._norm(cam, x), self._norm(campp, x), logits.detach().float().cpu()

    @staticmethod
    def _norm(cam, x):
        cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
        return (cam / cam.flatten(1).max(1)[0].clamp(min=1e-12)[:, None, None]).cpu()


@torch.no_grad()
def occlusion_maps(model, x, cls, grid=8):
    """
    Model-agnostic saliency (Zeiler & Fergus 2014): drop of the probability of class `cls` when one cell of a
    grid x grid partition is replaced by the dataset mean (0 after normalisation). x (B,3,H,W) -> maps (B,H,W) in [0,1].
    """
    H, W = x.shape[-2:]
    ys, xs = np.linspace(0, H, grid + 1).astype(int), np.linspace(0, W, grid + 1).astype(int)

    def prob(inp):
        with torch.autocast('cuda', dtype=torch.float16, enabled=inp.is_cuda):
            return model(inp).float().softmax(1).gather(1, cls[:, None])[:, 0]

    base = prob(x)
    drop = torch.zeros(x.shape[0], 1, grid, grid, device=x.device)
    for r in range(grid):
        for c in range(grid):
            xo = x.clone()
            xo[:, :, ys[r]:ys[r + 1], xs[c]:xs[c + 1]] = 0
            drop[:, 0, r, c] = (base - prob(xo)).clamp(min=0)
    maps = F.interpolate(drop, size=(H, W), mode='nearest')[:, 0]
    return (maps / maps.flatten(1).max(1)[0].clamp(min=1e-12)[:, None, None]).cpu()


def _focus(cam, m):
    """(leaf energy share, focus ratio) of one map, or None if the map is empty."""
    if cam.sum() <= 0:
        return None
    share = float((cam * m).sum() / cam.sum())
    return share, share / float(m.mean())


def _stats(prefix, values):
    if not values:
        return {}
    s, r = np.array(values).T
    return {f'{prefix}leaf_energy_share': float(s.mean()), f'{prefix}leaf_energy_share_std': float(s.std()),
            f'{prefix}leaf_focus_ratio': float(r.mean()), f'{prefix}leaf_focus_ratio_median': float(np.median(r))}


def leaf_mask(img):
    """img (H,W,3) uint8/float -> boolean leaf mask, or None if the background is not uniform."""
    a = np.asarray(img, dtype=np.float32)
    border = np.concatenate([a[:4].reshape(-1, 3), a[-4:].reshape(-1, 3),
                             a[:, :4].reshape(-1, 3), a[:, -4:].reshape(-1, 3)])
    bg = np.median(border, 0)
    # uniform background = most of the border has the paper colour (a table edge in one corner is tolerated)
    if (np.linalg.norm(border - bg, axis=1) <= LEAF_THR).mean() < BORDER_UNIFORM_MIN:
        return None
    mask = np.linalg.norm(a - bg, axis=2) > LEAF_THR
    return mask if MASK_AREA[0] <= mask.mean() <= MASK_AREA[1] else None


def _raw_image(ds, i, size):
    path = os.path.join(ds.root, ds.samples[i][0])
    return np.asarray(Image.open(path).convert('RGB').resize((size, size), Image.BILINEAR))


def evaluate_cam(model, ds, device, img_size, batch_size=32, fig_idx=(), limit=0, leaf_focus=True, occ_grid=8):
    """
    Grad-CAM and Grad-CAM++ over the whole dataset (in order), occlusion on the images with a leaf mask and on the
    figure images.
    Returns
      summary: leaf-focus statistics (images with uniform background only) for Grad-CAM (leaf_*), Grad-CAM++ (pp_*)
               and occlusion (occ_*),
      figs: {'gradcam' | 'gradcampp' | 'occlusion': {index: (raw image, map, true, pred, prob)}} for the indices in fig_idx.
    """
    cam_fn = GradCAM(model)
    fig_idx = set(fig_idx)
    cam_focus, pp_focus, occ_focus = [], [], []
    figs = {'gradcam': {}, 'gradcampp': {}, 'occlusion': {}}
    t_cam = t_occ = 0.0
    n = len(ds) if not limit else min(len(ds), limit * batch_size)
    try:
        for s in range(0, n, batch_size):
            idx = list(range(s, min(n, s + batch_size)))
            x = torch.stack([ds[i][0] for i in idx]).to(device)
            t0 = time.time()
            cams, campps, logits = cam_fn(x)
            t_cam += time.time() - t0
            probs = logits.softmax(1)
            raws = [_raw_image(ds, i, img_size) for i in idx]
            masks = [leaf_mask(r) if leaf_focus else None for r in raws]
            need = [k for k, i in enumerate(idx) if masks[k] is not None or i in fig_idx]
            occ = {}
            if need and occ_grid:
                t0 = time.time()
                maps = occlusion_maps(model, x[need], probs[need].argmax(1).to(device), occ_grid)
                t_occ += time.time() - t0
                occ = {k: maps[j].numpy() for j, k in enumerate(need)}
            for k, i in enumerate(idx):
                kmaps = {'gradcam': cams[k].numpy(), 'gradcampp': campps[k].numpy(), 'occlusion': occ.get(k)}
                m = masks[k]
                if m is not None:
                    for store, key in ((cam_focus, 'gradcam'), (pp_focus, 'gradcampp'), (occ_focus, 'occlusion')):
                        f = _focus(kmaps[key], m) if kmaps[key] is not None else None
                        if f is not None:
                            store.append(f)
                if i in fig_idx:
                    p = int(probs[k].argmax())
                    for key, mp in kmaps.items():
                        if mp is not None:
                            figs[key][i] = (raws[k], mp, ds.targets[i], p, float(probs[k, p]))
    finally:
        cam_fn.close()
    summary = {'cam_method': cam_fn.method, 'cam_n_images': n, 'cam_leaf_focus': bool(leaf_focus),
               'cam_n_uniform_background': len(cam_focus), 'occ_grid': occ_grid, 'occ_n_images': len(occ_focus),
               'cam_seconds': round(t_cam, 1), 'occ_seconds': round(t_occ, 1)}
    summary.update(_stats('', cam_focus))
    summary.update(_stats('pp_', pp_focus))
    summary.update(_stats('occ_', occ_focus))
    return summary, figs


def select_figure_images(ds, probs, per_class=2, n_errors=6, groups=None, cols=4):
    """Fixed images (first `per_class` test images of each class, the same in every run) + most confident errors.
    groups: same-leaf group per image (data.leaf_datasets.leaf_groups) -> each leaf appears only once.
    The total is rounded up to a multiple of `cols` (pairs per figure row) so that the last row is full: more errors,
    or, if the model makes too few, the next test images of each class."""
    groups = groups or list(range(len(ds)))
    sel, seen, used = [], {}, set()

    def add(i):
        sel.append(int(i))
        used.add(groups[i])

    for i, t in enumerate(ds.targets):
        if seen.get(t, 0) < per_class and groups[i] not in used:
            add(i)
            seen[t] = seen.get(t, 0) + 1
    total = min(len(set(groups)), int(np.ceil((len(sel) + n_errors) / cols)) * cols)
    if probs is not None:
        pred = probs.argmax(1)
        for i in np.argsort(-probs.max(1)):
            if len(sel) >= total:
                break
            if pred[i] != ds.targets[i] and groups[i] not in used:
                add(i)
    rank, order = {}, []                         # fill the last row with further fixed images, one class after another
    for i, t in enumerate(ds.targets):
        order.append((rank.get(t, 0), t, i))
        rank[t] = rank.get(t, 0) + 1
    for _, _, i in sorted(order):
        if len(sel) >= total:
            break
        if groups[i] not in used:
            add(i)
    return sel
