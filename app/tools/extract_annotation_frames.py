"""Extract timestamped frames and an optional contact sheet for annotation help."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract frames for traffic annotation."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--every-seconds", type=float)
    selection.add_argument("--timestamps", type=float, nargs="+")
    parser.add_argument("--contact-sheet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.every_seconds is not None and args.every_seconds <= 0:
        raise ValueError("--every-seconds must be greater than zero")
    if args.timestamps and any(value < 0 for value in args.timestamps):
        raise ValueError("--timestamps cannot contain negative values")
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - project dependency
        raise RuntimeError("OpenCV is required for frame extraction") from exc
    capture = cv2.VideoCapture(str(Path(args.video).resolve()))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {args.video}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    duration = frame_count / fps if fps > 0 else 0.0
    timestamps = (
        _periodic_timestamps(duration, args.every_seconds)
        if args.every_seconds is not None
        else sorted(set(args.timestamps))
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    for timestamp in timestamps:
        if duration > 0 and timestamp > duration + 1e-9:
            raise ValueError(
                f"timestamp {timestamp:.3f}s exceeds video duration {duration:.3f}s"
            )
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"could not extract frame at {timestamp:.3f}s")
        label = f"{timestamp:.3f}s"
        cv2.putText(
            frame,
            label,
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        path = output / f"frame_{round(timestamp * 1000):010d}ms.jpg"
        cv2.imwrite(str(path), frame)
        frames.append(frame)
    capture.release()
    if args.contact_sheet and frames:
        _write_contact_sheet(frames, output / "contact_sheet.jpg", cv2)
    print(f"Extracted {len(frames)} frame(s) to {output}")
    return 0


def _periodic_timestamps(duration: float, interval: float) -> list[float]:
    if duration <= 0:
        return [0.0]
    count = math.floor(duration / interval)
    return [index * interval for index in range(count + 1)]


def _write_contact_sheet(frames: list, path: Path, cv2) -> None:
    import numpy as np

    columns = min(4, len(frames))
    rows = math.ceil(len(frames) / columns)
    thumb_width = 320
    first_height, first_width = frames[0].shape[:2]
    thumb_height = max(1, round(first_height * thumb_width / first_width))
    sheet = np.zeros((rows * thumb_height, columns * thumb_width, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        thumbnail = cv2.resize(frame, (thumb_width, thumb_height))
        row, column = divmod(index, columns)
        sheet[
            row * thumb_height : (row + 1) * thumb_height,
            column * thumb_width : (column + 1) * thumb_width,
        ] = thumbnail
    cv2.imwrite(str(path), sheet)


if __name__ == "__main__":
    raise SystemExit(main())
