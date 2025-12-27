#!/usr/bin/env python3

import os
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

import click
import numpy as np
import cv2
import torch
from pytorch360convert import e2p
from p2e import p2e_and_blend_torch
from simple_lama_inpainting import SimpleLama


def fix_top_face_black_spots(equi_rgb_u8, top_mask_u8, lama,
                              fov_deg=(70, 70), h_deg=180, v_deg=50,
                              out_hw=(512, 512), feather=10, mode="bilinear"):
    """
    Fix black spots in the top (sky) area using perspective-based inpainting.

    Args:
        equi_rgb_u8: Equirectangular RGB image (numpy uint8, HWC)
        top_mask_u8: Inpainting mask for perspective view (numpy uint8, grayscale)
        lama: SimpleLama inpainting model instance
        fov_deg: Field of view (horizontal, vertical) in degrees
        h_deg: Horizontal rotation (yaw) in degrees
        v_deg: Vertical rotation (pitch) in degrees - positive looks up
        out_hw: Output perspective image size (height, width)
        feather: Feather radius for smooth blending (0 = no feathering)
        mode: Interpolation mode for e2p ('bilinear' or 'nearest')

    Returns:
        Equirectangular RGB image with inpainted top area (numpy uint8, HWC)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Step 1: Convert equirectangular to perspective view
    # e2p expects CHW float [0,1]
    equi_t = torch.from_numpy(equi_rgb_u8).permute(2, 0, 1).float().to(device) / 255.0

    # Extract perspective view
    # e2p expects CHW input (channels_first=True is default and correct for our equi_t)
    # e2p returns CHW output when channels_first=True
    persp_t = e2p(
        equi_t,
        fov_deg=fov_deg,
        h_deg=h_deg,
        v_deg=v_deg,
        out_hw=out_hw,
        mode=mode,
        channels_first=True  # Input is CHW, output will be CHW
    )

    # Step 2: Convert to numpy uint8 for LAMA inpainting
    # persp_t is CHW, need to convert to HWC for LAMA
    persp_u8 = (persp_t.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)

    # Step 3: Apply LAMA inpainting
    persp_fixed_u8 = np.asarray(lama(persp_u8, top_mask_u8), dtype=np.uint8)

    # Step 4: Convert inpainted perspective and original equirectangular to BHWC format for p2e
    # Add batch dimension and convert to float [0,1]
    persp_fixed_bhwc = torch.from_numpy(persp_fixed_u8).unsqueeze(0).float().to(device) / 255.0
    equi_bhwc = torch.from_numpy(equi_rgb_u8).unsqueeze(0).float().to(device) / 255.0

    # Step 5: Project perspective back to equirectangular with blending
    # Note: p2e_and_blend_torch uses u_deg (maps to h_deg) and v_deg (negated internally)
    merged, _, _ = p2e_and_blend_torch(
        perspective=persp_fixed_bhwc,
        equi_base=equi_bhwc,
        fov_deg=fov_deg,
        u_deg=h_deg,  # Horizontal rotation
        v_deg=v_deg,  # Vertical rotation (will be negated internally)
        feather=feather,
        device=device
    )

    # Step 6: Convert back to numpy uint8 HWC
    # merged is BHWC, squeeze batch dimension
    return (merged.squeeze(0).detach().cpu().numpy() * 255.0).astype(np.uint8)


def fix_panorama_seam(equi_rgb_u8, lama, seam_width=40, pad=128, feather=64, mask_sigma=3.0):
    H, W, _ = equi_rgb_u8.shape

    rolled = np.roll(equi_rgb_u8, shift=W // 2, axis=1)

    cx = W // 2
    x0 = max(0, cx - (seam_width + pad))
    x1 = min(W, cx + (seam_width + pad))

    crop = rolled[:, x0:x1, :].copy()
    crop_w = crop.shape[1]
    scx = crop_w // 2

    mask = np.zeros((H, crop_w), dtype=np.uint8)
    mask[:, max(0, scx - seam_width) : min(crop_w, scx + seam_width)] = 255
    if mask_sigma > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), float(mask_sigma))

    crop_fixed = np.asarray(lama(crop, mask), dtype=np.uint8)

    if feather > 0:
        x = np.arange(crop_w, dtype=np.float32)
        a = np.minimum(
            np.clip(x / feather, 0.0, 1.0),
            np.clip((crop_w - 1 - x) / feather, 0.0, 1.0),
        )
        alpha = np.repeat(a[None, :, None], H, axis=0)
    else:
        alpha = np.ones((H, crop_w, 1), dtype=np.float32)

    blended = (alpha * crop_fixed.astype(np.float32) + (1.0 - alpha) * crop.astype(np.float32)).astype(np.uint8)
    rolled[:, x0:x1, :] = blended

    return np.roll(rolled, shift=-(W // 2), axis=1)


def read_rgb(path: Path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)


def write_rgb(path: Path, rgb_u8: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])


def read_mask(path: Path):
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).astype(np.uint8)


def normalize_mask(mask_u8: np.ndarray, size_hw):
    h, w = size_hw
    if mask_u8.shape[:2] != (h, w):
        mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)
    return ((mask_u8 > 0).astype(np.uint8) * 255)


_LAMA = None
_TOP_MASK = None


def worker_init(top_mask_path: str, out_hw: tuple):
    global _LAMA, _TOP_MASK
    _LAMA = SimpleLama()
    h, w = out_hw
    _TOP_MASK = normalize_mask(read_mask(Path(top_mask_path)), (int(h), int(w)))


def process_one(in_path: str, out_path: str, fov_deg: tuple, h_deg: float, v_deg: float,
                out_hw: tuple, top_feather: int, seam_width: int, pad: int,
                seam_feather: int, mask_sigma: float):
    global _LAMA, _TOP_MASK

    img = read_rgb(Path(in_path))

    img = fix_top_face_black_spots(
        img, _TOP_MASK, _LAMA,
        fov_deg=fov_deg, h_deg=h_deg, v_deg=v_deg,
        out_hw=out_hw, feather=top_feather
    )
    img = fix_panorama_seam(
        img, _LAMA,
        seam_width=seam_width, pad=pad, feather=seam_feather, mask_sigma=mask_sigma
    )

    write_rgb(Path(out_path), img)


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("-i", "--input", "input_path", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("-o", "--output", "output_dir", required=True, type=click.Path(path_type=Path))
@click.option("--top-mask", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--pattern", default="*.jpg", show_default=True)
@click.option("--recursive/--no-recursive", default=True, show_default=True)
@click.option("--skip-existing/--no-skip-existing", default=True, show_default=True)
@click.option("-j", "--workers", default=1, show_default=True, type=int)
@click.option("--fov-w", default=70, show_default=True, type=int, help="Perspective FOV width (degrees)")
@click.option("--fov-h", default=70, show_default=True, type=int, help="Perspective FOV height (degrees)")
@click.option("--h-deg", default=180, show_default=True, type=float, help="Horizontal rotation (degrees)")
@click.option("--v-deg", default=50, show_default=True, type=float, help="Vertical rotation (degrees)")
@click.option("--persp-h", default=512, show_default=True, type=int, help="Perspective height (px)")
@click.option("--persp-w", default=512, show_default=True, type=int, help="Perspective width (px)")
@click.option("--top-feather", default=10, show_default=True, type=int, help="Top blend feather radius")
@click.option("--seam-width", default=40, show_default=True, type=int)
@click.option("--pad", default=128, show_default=True, type=int)
@click.option("--seam-feather", default=30, show_default=True, type=int, help="Seam blend feather radius")
@click.option("--mask-sigma", default=3.0, show_default=True, type=float)
def main(
    input_path: Path,
    output_dir: Path,
    top_mask: Path,
    pattern: str,
    recursive: bool,
    skip_existing: bool,
    workers: int,
    fov_w: int,
    fov_h: int,
    h_deg: float,
    v_deg: float,
    persp_h: int,
    persp_w: int,
    top_feather: int,
    seam_width: int,
    pad: int,
    seam_feather: int,
    mask_sigma: float,
):
    t0 = time.perf_counter()

    if input_path.is_file():
        in_files = [input_path]
        in_root = input_path.parent
    else:
        in_root = input_path
        in_files = sorted(input_path.rglob(pattern) if recursive else input_path.glob(pattern))

    pairs = []
    for p in in_files:
        rel = p.relative_to(in_root)
        out_p = output_dir / rel
        if skip_existing and out_p.exists():
            continue
        out_p.parent.mkdir(parents=True, exist_ok=True)
        pairs.append((str(p), str(out_p)))

    if not pairs:
        return

    if workers == 1:
        worker_init(str(top_mask), (persp_h, persp_w))
        for in_p, out_p in pairs:
            process_one(
                in_p, out_p,
                (fov_w, fov_h), h_deg, v_deg, (persp_h, persp_w), top_feather,
                seam_width, pad, seam_feather, mask_sigma
            )
        print(f"total: {time.perf_counter() - t0:.2f}s", flush=True)
        return

        mp_ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=worker_init,
        initargs=(str(top_mask), (persp_h, persp_w)),
        mp_context=mp.get_context("spawn"),
    ) as ex:
        futs = [
            ex.submit(
                process_one, in_p, out_p,
                (fov_w, fov_h), h_deg, v_deg, (persp_h, persp_w), top_feather,
                seam_width, pad, seam_feather, mask_sigma
            )
            for in_p, out_p in pairs
        ]
        for _ in as_completed(futs):
            pass

    print(f"total: {time.perf_counter() - t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
