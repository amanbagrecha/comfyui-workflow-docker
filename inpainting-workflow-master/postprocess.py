#!/usr/bin/env python3

import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pytorch360convert import e2p
from simple_lama_inpainting import SimpleLama


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
ORDERED_IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"]


def _load_p2e_and_blend_torch():
    try:
        from p2e import p2e_and_blend_torch as fn

        return fn
    except Exception:
        import importlib.util

        fallback_nodes = [
            Path("/workspace/ComfyUI/custom_nodes/p2e/nodes.py"),
            Path(__file__).resolve().parents[1] / "p2e-local" / "nodes.py",
        ]
        chosen_nodes = None
        for cand in fallback_nodes:
            if cand.exists():
                chosen_nodes = cand
                break
        if chosen_nodes is None:
            raise

        spec = importlib.util.spec_from_file_location(
            "p2e_nodes_fallback", str(chosen_nodes)
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"Unable to load fallback p2e module from {chosen_nodes}"
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.p2e_and_blend_torch


p2e_and_blend_torch = _load_p2e_and_blend_torch()


def fix_top_face_black_spots(
    equi_rgb_u8,
    top_mask_u8,
    lama,
    fov_deg=(70, 70),
    h_deg=180.0,
    v_deg=50.0,
    out_hw=(512, 512),
    feather=10,
    mode="bilinear",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    equi_t = torch.from_numpy(equi_rgb_u8).permute(2, 0, 1).float().to(device) / 255.0

    persp_t = e2p(
        equi_t,
        fov_deg=fov_deg,
        h_deg=h_deg,
        v_deg=v_deg,
        out_hw=out_hw,
        mode=mode,
        channels_first=True,
    )

    persp_u8 = (persp_t.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(
        np.uint8
    )

    persp_fixed_u8 = np.asarray(lama(persp_u8, top_mask_u8), dtype=np.uint8)

    persp_fixed_bhwc = (
        torch.from_numpy(persp_fixed_u8).unsqueeze(0).float().to(device) / 255.0
    )
    equi_bhwc = torch.from_numpy(equi_rgb_u8).unsqueeze(0).float().to(device) / 255.0

    merged, _, _ = p2e_and_blend_torch(
        perspective=persp_fixed_bhwc,
        equi_base=equi_bhwc,
        fov_deg=fov_deg,
        u_deg=h_deg,
        v_deg=v_deg,
        feather=feather,
        device=device,
    )

    return (merged.squeeze(0).detach().cpu().numpy() * 255.0).astype(np.uint8)


def fix_panorama_seam(
    equi_rgb_u8, lama, seam_width=40, pad=128, feather=64, mask_sigma=3.0
):
    h, w, _ = equi_rgb_u8.shape

    rolled = np.roll(equi_rgb_u8, shift=w // 2, axis=1)

    cx = w // 2
    x0 = max(0, cx - (seam_width + pad))
    x1 = min(w, cx + (seam_width + pad))

    crop = rolled[:, x0:x1, :].copy()
    crop_w = crop.shape[1]
    scx = crop_w // 2

    mask = np.zeros((h, crop_w), dtype=np.uint8)
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
        alpha = np.repeat(a[None, :, None], h, axis=0)
    else:
        alpha = np.ones((h, crop_w, 1), dtype=np.float32)

    blended = (
        alpha * crop_fixed.astype(np.float32) + (1.0 - alpha) * crop.astype(np.float32)
    ).astype(np.uint8)
    rolled[:, x0:x1, :] = blended

    return np.roll(rolled, shift=-(w // 2), axis=1)


def _pil_tensor_from_rgb(rgb_u8: np.ndarray, device: str, dtype: torch.dtype):
    return (
        torch.from_numpy(rgb_u8)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=dtype)
        / 255.0
    )


def _mask_tensor_from_u8(mask_u8: np.ndarray, device: str, dtype: torch.dtype):
    return (
        torch.from_numpy(mask_u8)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device=device, dtype=dtype)
        / 255.0
    )


def gaussian_blur_t(t: torch.Tensor, radius: float):
    if radius < 0.5:
        return t

    k = int(radius) * 2 + 1
    sigma = float(radius)
    ax = torch.arange(k, dtype=torch.float32, device=t.device) - (k - 1) / 2.0
    kernel_1d = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_1d = kernel_1d.to(dtype=t.dtype)
    c = t.shape[1]
    kh = kernel_1d.view(1, 1, k, 1).expand(c, 1, k, 1)
    kw = kernel_1d.view(1, 1, 1, k).expand(c, 1, 1, k)
    p = k // 2
    out = F.conv2d(F.pad(t, (0, 0, p, p), mode="reflect"), kh, groups=c)
    out = F.conv2d(F.pad(out, (p, p, 0, 0), mode="reflect"), kw, groups=c)
    return out


def dilate_t(mask: torch.Tensor, iterations: int):
    for _ in range(iterations):
        mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    return mask


def downsample(t: torch.Tensor):
    return F.interpolate(t, scale_factor=0.5, mode="bilinear", align_corners=False)


def upsample(t: torch.Tensor, size_hw: Tuple[int, int]):
    return F.interpolate(t, size=size_hw, mode="bilinear", align_corners=False)


def build_gauss_pyr(t: torch.Tensor, levels: int):
    pyr = [t]
    for _ in range(levels):
        h, w = pyr[-1].shape[2:]
        if h <= 1 or w <= 1:
            break
        blurred = gaussian_blur_t(pyr[-1], radius=2)
        pyr.append(downsample(blurred))
    return pyr


def build_lap_pyr(gpyr: List[torch.Tensor]):
    lap = []
    for i in range(len(gpyr) - 1):
        h, w = gpyr[i].shape[2:]
        up = upsample(gpyr[i + 1], (h, w))
        lap.append(gpyr[i] - up)
    lap.append(gpyr[-1])
    return lap


def collapse_lap_pyr(lp: List[torch.Tensor]):
    result = lp[-1]
    for i in range(len(lp) - 2, -1, -1):
        h, w = lp[i].shape[2:]
        result = upsample(result, (h, w)) + lp[i]
    return result


def align_mask_to_image(mask_u8: np.ndarray, target_hw: Tuple[int, int]):
    th, tw = target_hw
    mh, mw = mask_u8.shape[:2]
    if mh <= 0 or mw <= 0:
        raise ValueError("Mask has invalid shape")

    mr = float(mw) / float(mh)
    tr = float(tw) / float(th)

    aligned = mask_u8
    if abs(mr - tr) > 1e-6:
        if mr > tr:
            nw = int(round(mh * tr))
            x0 = max(0, (mw - nw) // 2)
            aligned = aligned[:, x0 : x0 + nw]
        else:
            nh = int(round(mw / tr))
            y0 = max(0, (mh - nh) // 2)
            aligned = aligned[y0 : y0 + nh, :]

    return cv2.resize(aligned, (tw, th), interpolation=cv2.INTER_LANCZOS4)


def orient_mask_to_sky(mask_u8: np.ndarray):
    h, _ = mask_u8.shape
    top_band = mask_u8[: max(1, h // 3), :]
    bot_band = mask_u8[max(0, 2 * h // 3) :, :]
    top_mean = float(np.mean(top_band))
    bot_mean = float(np.mean(bot_band))
    if top_mean < bot_mean:
        return 255 - mask_u8, True
    return mask_u8, False


def threshold_mask(mask_u8: np.ndarray, threshold: int = 127):
    return (mask_u8 >= threshold).astype(np.uint8) * 255


def laplacian_sky_replace(
    base_rgb_u8: np.ndarray,
    sky_rgb_u8: np.ndarray,
    sam3_mask_u8: np.ndarray,
    dilation: int,
    blur_radius: int,
    levels: int,
    use_fp16: bool,
):
    if base_rgb_u8.shape[:2] != sky_rgb_u8.shape[:2]:
        h, w = base_rgb_u8.shape[:2]
        sky_rgb_u8 = cv2.resize(sky_rgb_u8, (w, h), interpolation=cv2.INTER_LANCZOS4)

    h, w = base_rgb_u8.shape[:2]
    mask_aligned = align_mask_to_image(sam3_mask_u8, (h, w))
    mask_oriented, _ = orient_mask_to_sky(mask_aligned)
    mask_bin = threshold_mask(mask_oriented, threshold=127)

    if not np.any(mask_bin):
        return base_rgb_u8

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (use_fp16 and device == "cuda") else torch.float32

    base_t = _pil_tensor_from_rgb(base_rgb_u8, device, dtype)
    sky_t = _pil_tensor_from_rgb(sky_rgb_u8, device, dtype)
    mask_t = _mask_tensor_from_u8(mask_bin, device, dtype)

    blend_mask = dilate_t(mask_t, max(0, int(dilation)))
    blend_mask = gaussian_blur_t(blend_mask, radius=max(0, int(blur_radius))).clamp(
        0, 1
    )

    gp_base = build_gauss_pyr(base_t, max(0, int(levels)))
    gp_sky = build_gauss_pyr(sky_t, max(0, int(levels)))
    gp_mask = build_gauss_pyr(blend_mask, max(0, int(levels)))

    min_len = min(len(gp_base), len(gp_sky), len(gp_mask))
    gp_base = gp_base[:min_len]
    gp_sky = gp_sky[:min_len]
    gp_mask = gp_mask[:min_len]

    lp_base = build_lap_pyr(gp_base)
    lp_sky = build_lap_pyr(gp_sky)
    lp_blend = [
        ls * gm + lb * (1.0 - gm) for lb, ls, gm in zip(lp_base, lp_sky, gp_mask)
    ]

    result_t = collapse_lap_pyr(lp_blend).clamp(0, 1)
    return (result_t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(
        np.uint8
    )


def read_rgb(path: Path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)


def write_rgb(path: Path, rgb_u8: np.ndarray, jpeg_quality: int = 85):
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    params = []
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    cv2.imwrite(str(path), bgr, params)


def read_mask(path: Path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    return mask.astype(np.uint8)


def normalize_mask(mask_u8: np.ndarray, size_hw: Tuple[int, int]):
    h, w = size_hw
    if mask_u8.shape[:2] != (h, w):
        mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask_u8 > 0).astype(np.uint8) * 255


def _iter_input_files(input_path: Path, pattern: str, recursive: bool):
    if input_path.is_file():
        return [input_path], input_path.parent

    files = sorted(input_path.rglob(pattern) if recursive else input_path.glob(pattern))
    files = [p for p in files if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return files, input_path


def _find_sibling_image(parent: Path, stem: str, preferred_ext: str):
    preferred = parent / f"{stem}{preferred_ext}"
    if preferred.exists():
        return preferred
    for ext in ORDERED_IMAGE_EXTS:
        cand = parent / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def _build_mask_index(mask_dir: Path):
    index = {}
    collisions = set()
    for p in mask_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = p.stem
        if stem in index and index[stem] != p:
            collisions.add(stem)
        else:
            index[stem] = p
    for stem in collisions:
        index[stem] = None
    return index


def _resolve_mask_path(
    mask_dir: Path,
    rel_dir: Path,
    base_stem: str,
    mask_index: Dict[str, Optional[Path]],
):
    rel_png = mask_dir / rel_dir / f"{base_stem}.png"
    if rel_png.exists():
        return rel_png

    root_png = mask_dir / f"{base_stem}.png"
    if root_png.exists():
        return root_png

    if base_stem in mask_index:
        return mask_index[base_stem]
    return None


def _build_legacy_jobs(
    in_files: List[Path],
    in_root: Path,
    output_dir: Path,
    skip_existing: bool,
):
    jobs = []
    for p in in_files:
        rel = p.relative_to(in_root)
        out_p = output_dir / rel
        if skip_existing and out_p.exists():
            continue
        out_p.parent.mkdir(parents=True, exist_ok=True)
        jobs.append((str(p), str(out_p), None, None))
    return jobs


def _build_laplacian_jobs(
    in_files: List[Path],
    in_root: Path,
    output_dir: Path,
    mask_dir: Path,
    skip_existing: bool,
    carremoved_suffix: str,
    newsky_suffix: str,
):
    mask_index = _build_mask_index(mask_dir)
    jobs = []
    errors = []
    matched_sources = 0

    for p in in_files:
        stem = p.stem
        if not stem.endswith(carremoved_suffix):
            continue

        base_stem = stem[: -len(carremoved_suffix)]
        if not base_stem:
            continue
        matched_sources += 1

        rel = p.relative_to(in_root)
        out_p = output_dir / rel.with_name(f"{base_stem}{p.suffix.lower()}")
        if skip_existing and out_p.exists():
            continue

        sky_stem = f"{base_stem}{newsky_suffix}"
        sky_path = _find_sibling_image(p.parent, sky_stem, p.suffix.lower())
        if sky_path is None:
            errors.append(f"Missing newsky pair for {p.name}")
            continue

        rel_dir = rel.parent
        mask_path = _resolve_mask_path(mask_dir, rel_dir, base_stem, mask_index)
        if mask_path is None:
            errors.append(f"Missing SAM3 mask for base '{base_stem}'")
            continue

        out_p.parent.mkdir(parents=True, exist_ok=True)
        jobs.append((str(p), str(out_p), str(sky_path), str(mask_path)))

    if errors:
        shown = "\n".join(errors[:20])
        more = len(errors) - min(len(errors), 20)
        if more > 0:
            shown += f"\n... and {more} more"
        raise click.ClickException(shown)

    if not jobs and matched_sources == 0:
        raise click.ClickException(
            f"No '{carremoved_suffix}' files found in input for Laplacian mode"
        )

    return jobs


_LAMA = None
_TOP_MASK = None


def worker_init(top_mask_path: str, out_hw: Tuple[int, int]):
    global _LAMA, _TOP_MASK
    _LAMA = SimpleLama()
    h, w = out_hw
    _TOP_MASK = normalize_mask(read_mask(Path(top_mask_path)), (int(h), int(w)))


def process_one(
    in_path: str,
    out_path: str,
    sky_path: Optional[str],
    sam3_mask_path: Optional[str],
    fov_deg: Tuple[int, int],
    h_deg: float,
    v_deg: float,
    out_hw: Tuple[int, int],
    top_feather: int,
    seam_width: int,
    pad: int,
    seam_feather: int,
    mask_sigma: float,
    dilation: int,
    blur_radius: int,
    levels: int,
    laplacian_fp16: bool,
    jpeg_quality: int,
):
    global _LAMA, _TOP_MASK

    img = read_rgb(Path(in_path))

    if sky_path and sam3_mask_path:
        sky = read_rgb(Path(sky_path))
        sam3_mask = read_mask(Path(sam3_mask_path))
        img = laplacian_sky_replace(
            base_rgb_u8=img,
            sky_rgb_u8=sky,
            sam3_mask_u8=sam3_mask,
            dilation=dilation,
            blur_radius=blur_radius,
            levels=levels,
            use_fp16=laplacian_fp16,
        )

    img = fix_top_face_black_spots(
        img,
        _TOP_MASK,
        _LAMA,
        fov_deg=fov_deg,
        h_deg=h_deg,
        v_deg=v_deg,
        out_hw=out_hw,
        feather=top_feather,
    )

    img = fix_panorama_seam(
        img,
        _LAMA,
        seam_width=seam_width,
        pad=pad,
        feather=seam_feather,
        mask_sigma=mask_sigma,
    )

    write_rgb(Path(out_path), img, jpeg_quality=jpeg_quality)
    return out_path


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(path_type=Path, exists=True),
)
@click.option(
    "-o", "--output", "output_dir", required=True, type=click.Path(path_type=Path)
)
@click.option("--top-mask", required=True, type=click.Path(path_type=Path, exists=True))
@click.option(
    "--sam3-mask-dir",
    default=None,
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    help="SAM3 mask directory. Enables Laplacian sky replacement mode.",
)
@click.option(
    "--carremoved-suffix",
    default="_comfyui_carremoved",
    show_default=True,
    help="Suffix used to identify source images from Comfy outputs",
)
@click.option(
    "--newsky-suffix",
    default="_comfyui_newsky",
    show_default=True,
    help="Suffix used to identify sky images from Comfy outputs",
)
@click.option("--dilation", default=1, show_default=True, type=int)
@click.option("--blur", "blur_radius", default=10, show_default=True, type=int)
@click.option("--levels", default=7, show_default=True, type=int)
@click.option(
    "--laplacian-roi-pad",
    default=96,
    show_default=True,
    type=int,
    help="Deprecated; unused in full-image Laplacian blend mode",
)
@click.option(
    "--laplacian-fp16/--no-laplacian-fp16",
    default=True,
    show_default=True,
    help="Use float16 for Laplacian blend tensors on CUDA",
)
@click.option("--pattern", default="*.jpg", show_default=True)
@click.option("--recursive/--no-recursive", default=True, show_default=True)
@click.option("--skip-existing/--no-skip-existing", default=True, show_default=True)
@click.option("-j", "--workers", default=1, show_default=True, type=int)
@click.option(
    "--fov-w",
    default=70,
    show_default=True,
    type=int,
    help="Perspective FOV width (degrees)",
)
@click.option(
    "--fov-h",
    default=70,
    show_default=True,
    type=int,
    help="Perspective FOV height (degrees)",
)
@click.option(
    "--h-deg",
    default=180,
    show_default=True,
    type=float,
    help="Horizontal rotation (degrees)",
)
@click.option(
    "--v-deg",
    default=50,
    show_default=True,
    type=float,
    help="Vertical rotation (degrees)",
)
@click.option(
    "--persp-h",
    default=512,
    show_default=True,
    type=int,
    help="Perspective height (px)",
)
@click.option(
    "--persp-w", default=512, show_default=True, type=int, help="Perspective width (px)"
)
@click.option(
    "--top-feather",
    default=10,
    show_default=True,
    type=int,
    help="Top blend feather radius",
)
@click.option("--seam-width", default=40, show_default=True, type=int)
@click.option("--pad", default=128, show_default=True, type=int)
@click.option(
    "--seam-feather",
    default=30,
    show_default=True,
    type=int,
    help="Seam blend feather radius",
)
@click.option("--mask-sigma", default=3.0, show_default=True, type=float)
@click.option(
    "--jpeg-quality",
    default=85,
    show_default=True,
    type=click.IntRange(1, 100),
    help="JPEG quality used when output extension is .jpg/.jpeg",
)
def main(
    input_path: Path,
    output_dir: Path,
    top_mask: Path,
    sam3_mask_dir: Optional[Path],
    carremoved_suffix: str,
    newsky_suffix: str,
    dilation: int,
    blur_radius: int,
    levels: int,
    laplacian_roi_pad: int,
    laplacian_fp16: bool,
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
    jpeg_quality: int,
):
    t0 = time.perf_counter()

    if workers < 1:
        raise click.ClickException("workers must be >= 1")
    if dilation < 0:
        raise click.ClickException("dilation must be >= 0")
    if blur_radius < 0:
        raise click.ClickException("blur must be >= 0")
    if levels < 0:
        raise click.ClickException("levels must be >= 0")
    output_dir.mkdir(parents=True, exist_ok=True)

    in_files, in_root = _iter_input_files(input_path, pattern, recursive)
    if not in_files:
        raise click.ClickException(f"No images found under: {input_path}")

    if sam3_mask_dir is not None:
        mode = "laplacian"
        jobs = _build_laplacian_jobs(
            in_files=in_files,
            in_root=in_root,
            output_dir=output_dir,
            mask_dir=sam3_mask_dir,
            skip_existing=skip_existing,
            carremoved_suffix=carremoved_suffix,
            newsky_suffix=newsky_suffix,
        )
    else:
        mode = "legacy"
        jobs = _build_legacy_jobs(
            in_files=in_files,
            in_root=in_root,
            output_dir=output_dir,
            skip_existing=skip_existing,
        )

    if not jobs:
        click.echo("No work to do (all outputs already exist).")
        return

    click.echo(
        f"postprocess_mode={mode} input_count={len(in_files)} pending={len(jobs)} workers={workers}"
    )
    if mode == "laplacian":
        click.echo(
            f"laplacian: dilation={dilation} blur={blur_radius} levels={levels} "
            f"fp16={laplacian_fp16} "
            f"carremoved_suffix='{carremoved_suffix}' newsky_suffix='{newsky_suffix}'"
        )

    if workers > 1 and torch.cuda.is_available():
        click.echo(
            "WARNING: workers>1 on CUDA can increase VRAM pressure in postprocess",
            err=True,
        )

    failures = []

    if workers == 1:
        worker_init(str(top_mask), (persp_h, persp_w))
        for in_p, out_p, sky_p, sam3_p in jobs:
            try:
                process_one(
                    in_p,
                    out_p,
                    sky_p,
                    sam3_p,
                    (fov_w, fov_h),
                    h_deg,
                    v_deg,
                    (persp_h, persp_w),
                    top_feather,
                    seam_width,
                    pad,
                    seam_feather,
                    mask_sigma,
                    dilation,
                    blur_radius,
                    levels,
                    laplacian_fp16,
                    jpeg_quality,
                )
                click.echo(f"OK   {Path(in_p).name} -> {Path(out_p).name}")
            except Exception as exc:
                msg = f"FAIL {Path(in_p).name}: {exc}"
                click.echo(msg, err=True)
                failures.append(msg)
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=worker_init,
            initargs=(str(top_mask), (persp_h, persp_w)),
            mp_context=mp.get_context("spawn"),
        ) as ex:
            futs = {
                ex.submit(
                    process_one,
                    in_p,
                    out_p,
                    sky_p,
                    sam3_p,
                    (fov_w, fov_h),
                    h_deg,
                    v_deg,
                    (persp_h, persp_w),
                    top_feather,
                    seam_width,
                    pad,
                    seam_feather,
                    mask_sigma,
                    dilation,
                    blur_radius,
                    levels,
                    laplacian_fp16,
                    jpeg_quality,
                ): (in_p, out_p)
                for in_p, out_p, sky_p, sam3_p in jobs
            }
            for fut in as_completed(futs):
                in_p, out_p = futs[fut]
                try:
                    fut.result()
                    click.echo(f"OK   {Path(in_p).name} -> {Path(out_p).name}")
                except Exception as exc:
                    msg = f"FAIL {Path(in_p).name}: {exc}"
                    click.echo(msg, err=True)
                    failures.append(msg)

    click.echo(f"total: {time.perf_counter() - t0:.2f}s")
    if failures:
        raise click.ClickException(f"{len(failures)} image(s) failed in postprocess")


if __name__ == "__main__":
    main()
