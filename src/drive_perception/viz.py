"""Draw detections onto frames.

Kept separate from the detector so the two can change independently: the detector
decides what is in the image, this decides how it looks. Both the detection step and
the tracking step later render through `draw_detections`, which is why the label text
can be overridden. Tracking passes its own strings so an object can show a persistent
id alongside the class.

Colors are blue, orange and magenta rather than the usual red/green/blue. Red and green
are the first pair to collapse under the common forms of colorblindness, and this
palette stays readable for everyone.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from .detector import Detection

# BGR, because OpenCV. Used as the fallback when a class name is not recognised.
PALETTE: list[tuple[int, int, int]] = [
    (255, 150, 50),   # blue
    (40, 170, 255),   # orange
    (200, 60, 200),   # magenta
    (80, 200, 220),   # yellow
]

# Colour follows the class name, not the class id. A COCO-pretrained model calls a car
# id 2 while our fine-tuned model calls it id 0, and an object that changes colour
# between two models is confusing to look at. The COCO names sit alongside the KITTI
# ones so both models render the same object the same way.
NAME_COLORS: dict[str, tuple[int, int, int]] = {
    "car": PALETTE[0],
    "pedestrian": PALETTE[1],
    "person": PALETTE[1],   # COCO name for a pedestrian
    "cyclist": PALETTE[2],
    "bicycle": PALETTE[2],  # COCO stands in for a cyclist
}


def color_for(cls_id: int, cls_name: str | None = None) -> tuple[int, int, int]:
    if cls_name is not None:
        known = NAME_COLORS.get(cls_name.lower())
        if known is not None:
            return known
    return PALETTE[cls_id % len(PALETTE)]


def format_label(det: Detection, show_score: bool = True) -> str:
    return f"{det.cls_name} {det.score:.2f}" if show_score else det.cls_name


def _scale(image: np.ndarray) -> tuple[int, float]:
    """Line thickness and font scale for this image size. KITTI frames are only about
    375 px tall with many small boxes, so heavy lines would swallow the objects."""
    short_side = min(image.shape[:2])
    thickness = max(1, round(short_side / 400))
    font_scale = max(0.35, min(0.8, short_side / 900))
    return thickness, font_scale


def draw_detections(
    image: np.ndarray,
    detections: Sequence[Detection],
    labels: Sequence[str] | None = None,
    show_score: bool = True,
) -> np.ndarray:
    """Return a copy of `image` with boxes and labels drawn on it.

    `labels` overrides the text per detection when given, which is how the tracker adds
    its ids. The input array is never modified."""
    if labels is not None and len(labels) != len(detections):
        raise ValueError(
            f"labels has {len(labels)} entries but there are {len(detections)} detections"
        )

    out = image.copy()
    h, w = out.shape[:2]
    thickness, font_scale = _scale(out)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, det in enumerate(detections):
        color = color_for(det.cls_id, det.cls_name)
        x1, y1, x2, y2 = (int(round(v)) for v in det.box)
        # Clamp so a box running off the frame still draws a sane rectangle.
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        text = labels[i] if labels is not None else format_label(det, show_score)
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 1)
        # Prefer the label above the box; drop it inside when there is no room up top.
        top = y1 - th - baseline
        ty = y1 if top >= 0 else min(y2 + th + baseline, h - 1)
        cv2.rectangle(
            out,
            (x1, ty - th - baseline),
            (min(x1 + tw, w - 1), ty),
            color,
            -1,
        )
        cv2.putText(out, text, (x1, ty - baseline), font, font_scale, (20, 20, 20), 1, cv2.LINE_AA)

    return out


def draw_trails(
    image: np.ndarray,
    trails: Sequence[tuple[Sequence[tuple[float, float]], int, str]],
) -> np.ndarray:
    """Draw the recent path of each tracked object.

    Each entry is a run of points plus the class id and name used to colour it. The
    trail is what makes tracking legible in a still frame: boxes alone look the same
    whether ids are stable or flickering, while a clean path shows the association held."""
    out = image.copy()
    thickness, _ = _scale(out)
    for points, cls_id, cls_name in trails:
        if len(points) < 2:
            continue
        pts = np.asarray([[int(x), int(y)] for x, y in points], dtype=np.int32)
        cv2.polylines(
            out, [pts], False, color_for(cls_id, cls_name), thickness, cv2.LINE_AA
        )
    return out


def draw_hud(image: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    """Overlay a few lines of status text in the top left corner.

    The panel is blended rather than drawn solid so the frame underneath stays visible.
    Used by the video renderer and again by the live demo, which needs the same readout."""
    out = image.copy()
    if not lines:
        return out
    font = cv2.FONT_HERSHEY_SIMPLEX
    _, font_scale = _scale(out)
    pad = 6
    sizes = [cv2.getTextSize(t, font, font_scale, 1)[0] for t in lines]
    width = max(w for w, _ in sizes) + pad * 2
    line_h = max(h for _, h in sizes) + pad
    height = line_h * len(lines) + pad

    panel = out[0:height, 0:width].copy()
    cv2.rectangle(panel, (0, 0), (width, height), (0, 0, 0), -1)
    out[0:height, 0:width] = cv2.addWeighted(panel, 0.55, out[0:height, 0:width], 0.45, 0)

    for i, text in enumerate(lines):
        y = pad + line_h * (i + 1) - pad // 2
        cv2.putText(out, text, (pad, y), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def save(image: np.ndarray, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
