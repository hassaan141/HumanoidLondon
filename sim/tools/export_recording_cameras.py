import argparse
from pathlib import Path

import torch


def _to_hwc_uint8(image):
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape {tuple(image.shape)}")
    if image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
        image = image.permute(1, 2, 0)
    image = image[..., :3].contiguous().to(torch.uint8).cpu()
    return image


def _write_ppm(path: Path, image: torch.Tensor) -> None:
    image = _to_hwc_uint8(image)
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError(f"Expected RGB image, got shape {tuple(image.shape)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        file.write(image.numpy().tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Export camera frames from a teleop_dataset.pt recording.")
    parser.add_argument("dataset", type=Path, help="Path to teleop_dataset.pt")
    parser.add_argument("--out", type=Path, default=None, help="Output directory")
    parser.add_argument("--frames", type=int, nargs="*", default=None, help="Frame indices to export")
    args = parser.parse_args()

    data = torch.load(args.dataset, map_location="cpu")
    frames = data.get("frames", [])
    if len(frames) == 0:
        raise RuntimeError(f"No frames found in {args.dataset}")

    frame_ids = args.frames
    if frame_ids is None:
        frame_ids = sorted(set([0, len(frames) // 2, len(frames) - 1]))

    out_dir = args.out or args.dataset.with_suffix("")
    print(f"dataset: {args.dataset}")
    print(f"fps: {data.get('fps')}")
    print(f"cameras: {data.get('cameras')}")
    print(f"frames: {len(frames)}")
    print(f"exporting to: {out_dir}")

    for frame_id in frame_ids:
        frame = frames[frame_id]
        images = frame.get("images", {})
        print(f"frame {frame_id}: cameras={list(images.keys())}")
        for camera_name, image in images.items():
            image = _to_hwc_uint8(image)
            print(
                f"  {camera_name}: shape={tuple(image.shape)} "
                f"min={image.min().item()} max={image.max().item()} "
                f"mean={float(image.float().mean()):.2f}"
            )
            _write_ppm(out_dir / f"frame_{frame_id:06d}_{camera_name}.ppm", image)


if __name__ == "__main__":
    main()
