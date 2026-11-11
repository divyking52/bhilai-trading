"""Regenerates transparent product cut-outs from the raw catalogue photos.

Background is detected as light, neutral pixels connected to the image border (plus, per image,
enclosed background holes). Edge pixels are then colour-decontaminated against the local
background colour so cut-outs sit cleanly on dark backgrounds without white or grey halos.
"""
import os
import sys
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

RAW = 'assets/img'
OUT = 'assets/media/cut'
PREVIEW = 'tools/preview'
os.makedirs(PREVIEW, exist_ok=True)

DEFAULT = dict(lum=205, spread=26, holes=False, protect=(), band=3, shadow=False, guard=True)
JOBS = {
    'abc-multipurpose': ('fire.jpg', {}),
    'clean-agent-4kg': ('4-kg-clean-agent-fire-extinguisher.jpg', {}),
    'fire-bucket': ('FIRE_BUCKET.jpg', {}),
    'abc-module-2kg': ('abc-modular_2KG_-fire-extinguisher.jpg', {}),
    'abc-modular-5kg': ('5KG_ABC_MODULAR_ONLY_BODY_WITH_STAND.png', {}),
    'bucket-stand': ('6_BUCKET_STAND_ONLY.jpg', {'holes': True, 'guard': False}),
    'water-9l': ('9-litre-water-co2-type-fire-extinguisher.jpg', {'lum': 228, 'spread': 20, 'holes': True}),
    'dcp-25kg': ('25KG_DCP_WHEEL.jpg', {'shadow': True, 'holes': True}),
    'afff-foam-50l': ('50LTR_M_FOAM_300GM.jpg', {'shadow': True, 'holes': True}),
    'double-hydrant': ('isi-s_s_double_hydrant_valve.png', {'lum': 215}),
    'two-way-inlet': ('2-way-inlet-system.jpg', {}),
    'four-way-inlet': ('4_WAY_INLET.jpg', {}),
    'dcp-trolley': ('DCP_BC_TROLLY_MOUNTED_25_KG.jpg', {'shadow': True, 'holes': True}),
    'water-mist-6l': ('6ltr-water-mist.jpg', {'holes': True}),
    'butterfly-valve': ('BUTTER_FLY.jpg', {}),
    'dcp-hose-pipe': ('DCP_HOSE_PIPE_10KG.jpg', {}),
    'hose-reel': ('HOSE_REEL.jpg', {}),
    'hose-box': ('Double-Door-FRP-Hose-Box.jpg', {}),
    'co2-4-5kg': ('co2-4_5kg.jpg', {'lum': 95, 'spread': 20, 'holes': True,
                                    'protect': [(226, 274, 302, 478), (252, 128, 291, 141), (263, 141, 279, 152), (196, 175, 210, 204)]}),
    'k-type': ('K_TYPE_2KG_FIRE_EX_.jpg', {'lum': 225, 'holes': True}),
    'foam-monitor': ('fire-foam-monitor-.jpeg', {}),
    'concealed-sprinkler': ('concealed-sprinkler.jpg', {'lum': 232}),
    'hydrant-valve': ('fire-hydrant-valve.jpg', {'lum': 220}),
    'dual-plate-valve': ('dual-plate-non-return-valve.jpg', {}),
    'jet-spray-nozzle': ('JET_SPRY_NOZZLE.jpg', {'lum': 236}),
    'key-box': ('KEY_BOX.jpg', {}),
    'floor-stand': ('FLOOR_STAND_FIRE_E_.jpeg', {'holes': True}),
    'safety-clip': ('SAFETY_CLIP.jpg', {'lum': 236, 'holes': True}),
    'dcp-wheel-pair': ('50KG_DCP_WHEEL_PAIR.jpg', {'lum': 100, 'spread': 22, 'shadow': True}),
}


def load_rgb(path):
    im = Image.open(path)
    if im.mode in ('RGBA', 'LA', 'P'):
        im = im.convert('RGBA')
        bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        im = bg
    return np.asarray(im.convert('RGB')).astype(np.float32)


def cutout(name, src, cfg):
    c = {**DEFAULT, **cfg}
    rgb = load_rgb(os.path.join(RAW, src))
    h, w, _ = rgb.shape
    lo, hi = rgb.min(2), rgb.max(2)
    lum = rgb.mean(2)
    cand = ((hi - lo) <= c['spread']) & (lum >= c['lum'])
    if c['shadow']:
        cand |= ((hi - lo) <= 14) & (lum >= 150)
    cand[:3, :] = cand[-3:, :] = True
    cand[:, :3] = cand[:, -3:] = True

    protect = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in c['protect']:
        protect[y0:y1, x0:x1] = True
    cand &= ~protect

    lab, n = ndi.label(cand)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(border))
    if c['holes']:
        bg_lum = np.median(lum[bg])
        sizes = ndi.sum(np.ones_like(lum), lab, index=np.arange(1, n + 1))
        means = ndi.mean(lum, lab, index=np.arange(1, n + 1))
        sat = (hi - lo) > 60
        for i in range(1, n + 1):
            if i in border:
                continue
            if sizes[i - 1] >= max(150, 0.002 * h * w) and abs(means[i - 1] - bg_lum) <= 30:
                comp = lab == i
                ring = ndi.binary_dilation(comp, iterations=4) & ~comp
                sl = ndi.find_objects(comp.astype(int))[0]
                solidity = comp[sl].sum() / max(1, ndi.binary_fill_holes(comp[sl]).sum())
                # Labels sit inside coloured bodies and are broken up by printed text;
                # trapped background is clean and featureless.
                if not c['guard'] or sat[ring].mean() < 0.3 or solidity > 0.97:
                    bg |= comp
    bg = ndi.binary_opening(bg, iterations=1) | (bg & ndi.binary_erosion(bg, iterations=2))

    # Local background colour for every pixel = colour of its nearest background pixel.
    _, (iy, ix) = ndi.distance_transform_edt(~bg, return_indices=True)
    B = rgb[iy, ix]
    dist = ndi.distance_transform_edt(~bg)

    # Colour-to-alpha against the local background inside a thin edge band.
    up = np.where(rgb > B, (rgb - B) / np.maximum(255 - B, 1), (B - rgb) / np.maximum(B, 1))
    a_edge = np.clip(up.max(2), 0, 1)
    a_edge = np.clip((a_edge - 0.06) / 0.94, 0, 1)
    band = (~bg) & (dist <= c['band'])
    alpha = np.where(bg, 0.0, 1.0)
    alpha[band] = a_edge[band]
    alpha[protect & ~bg] = 1.0

    a3 = np.maximum(alpha, 1e-3)[..., None]
    fg = np.clip(B + (rgb - B) / a3, 0, 255)
    fg = np.where(band[..., None], fg, rgb)

    # Drop detached specks (watermark letters, dust).
    solid = alpha > 0.5
    lab2, n2 = ndi.label(solid)
    if n2 > 1:
        sizes2 = ndi.sum(solid, lab2, index=np.arange(1, n2 + 1))
        keep = np.isin(lab2, np.where(sizes2 >= sizes2.max() * 0.015)[0] + 1)
        keep = ndi.binary_dilation(keep, iterations=2)
        alpha = alpha * keep

    out = np.dstack([fg, alpha * 255]).astype(np.uint8)
    im = Image.fromarray(out, 'RGBA')
    bbox = im.getchannel('A').point(lambda v: 255 if v > 12 else 0).getbbox()
    im = im.crop(bbox)
    if im.width > 700:
        im = im.resize((700, round(im.height * 700 / im.width)), Image.LANCZOS)
    im.save(os.path.join(OUT, f'{name}.webp'), 'WEBP', quality=86, method=6)
    return im


if __name__ == '__main__':
    only = set(sys.argv[1:])
    tiles = []
    for name, (src, cfg) in JOBS.items():
        if only and name not in only:
            continue
        im = cutout(name, src, cfg)
        tiles.append((name, im))
        print(f'{name:22} {im.size}')
    cell, cols = 260, 6
    rows = (len(tiles) + cols - 1) // cols
    for bgcol, tag in (((18, 19, 22), 'dark'), ((120, 24, 30), 'red')):
        sheet = Image.new('RGBA', (cols * cell, rows * cell), bgcol + (255,))
        for i, (name, im) in enumerate(tiles):
            t = im.copy()
            t.thumbnail((cell - 12, cell - 12))
            sheet.alpha_composite(t, ((i % cols) * cell + 6, (i // cols) * cell + 6))
        sheet.convert('RGB').save(f'{PREVIEW}/sheet_{tag}.jpg', quality=88)
