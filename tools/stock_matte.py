"""Removes the baked-in transparency checkerboard from Rawpixel CC0 preview PNGs.

The preview grid alternates 238/255 grey squares on a 55 px period; pixels matching that
model (border-connected, or enclosed regions that show both checker tones) are background.
"""
import os
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

SRC = 'tools/stock'
OUT = 'assets/media/stock'
os.makedirs(OUT, exist_ok=True)
JOBS = {'ext-a': 'extinguisher', 'smoke-a': 'smoke-detector'}
SQ = 27.5


def matte(src, name):
    rgb = np.asarray(Image.open(os.path.join(SRC, src + '.png')).convert('RGB')).astype(np.float32)
    h, w, _ = rgb.shape
    yy, xx = np.mgrid[0:h, 0:w]
    model = np.where(((xx // SQ) + (yy // SQ)) % 2 == 0, 238.0, 255.0)
    near_edge = (np.minimum(xx % SQ, SQ - xx % SQ) < 1.5) | (np.minimum(yy % SQ, SQ - yy % SQ) < 1.5)
    lo, hi = rgb.min(2), rgb.max(2)
    lum = rgb.mean(2)
    neutral = (hi - lo) <= 4
    cand = neutral & ((np.abs(lum - model) <= 4) | (near_edge & (lum >= 234)))

    lab, n = ndi.label(cand)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(border))
    for i in range(1, n + 1):
        if i in border:
            continue
        comp = lab == i
        if comp.sum() < 300:
            continue
        vals = lum[comp]
        if (vals < 245).mean() > 0.15 and (vals > 250).mean() > 0.15:
            bg |= comp
    bg = ndi.binary_opening(bg, iterations=1)

    B = np.repeat(model[..., None], 3, axis=2)
    dist = ndi.distance_transform_edt(~bg)
    up = np.where(rgb > B, (rgb - B) / np.maximum(255 - B, 1), (B - rgb) / np.maximum(B, 1))
    a_edge = np.clip((up.max(2) - 0.04) / 0.96, 0, 1)
    band = (~bg) & (dist <= 3)
    alpha = np.where(bg, 0.0, 1.0)
    alpha[band] = a_edge[band]
    fg = np.where(band[..., None], np.clip(B + (rgb - B) / np.maximum(alpha, 1e-3)[..., None], 0, 255), rgb)

    solid = alpha > 0.5
    lab2, n2 = ndi.label(solid)
    if n2 > 1:
        sizes = ndi.sum(solid, lab2, index=np.arange(1, n2 + 1))
        keep = ndi.binary_dilation(np.isin(lab2, np.where(sizes >= sizes.max() * 0.01)[0] + 1), iterations=2)
        alpha *= keep

    im = Image.fromarray(np.dstack([fg, alpha * 255]).astype(np.uint8), 'RGBA')
    im = im.crop(im.getchannel('A').point(lambda v: 255 if v > 12 else 0).getbbox())
    if im.height > 1000:
        im = im.resize((round(im.width * 1000 / im.height), 1000), Image.LANCZOS)
    im.save(os.path.join(OUT, name + '.webp'), 'WEBP', quality=88, method=6)
    for col, tag in (((18, 19, 22), 'dark'), ((217, 35, 50), 'red'), ((245, 244, 240), 'light')):
        prev = Image.new('RGBA', im.size, col + (255,))
        prev.alpha_composite(im)
        prev.convert('RGB').save(f'{SRC}/_{name}_{tag}.jpg', quality=88)
    print(name, im.size)


for s, n in JOBS.items():
    matte(s, n)
