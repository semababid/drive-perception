"""Export the trained detector to ONNX.

ONNX is the step that takes the model out of PyTorch. Once the graph is in ONNX it can
run under ONNX Runtime on any machine, or be compiled by TensorRT into an engine for a
specific GPU, without the training framework being installed at all.

Two choices here are deliberate rather than defaults.

The opset is pinned. Exporters happily emit the newest opset they know, and TensorRT
consistently supports fewer opsets than ONNX Runtime does, so an unpinned export is the
classic failure that works on the development machine and is rejected by the edge
runtime much later. Pinning it means the file that passes the parity check is the same
file the benchmark compiles.

Shapes are static by default. A fixed input lets TensorRT specialise the whole graph,
which is both faster to build and faster to run. Dynamic shapes stay available for cases
that genuinely need a variable batch, at a cost in both.
"""

from __future__ import annotations

from pathlib import Path

# TensorRT is the constraint on this number, not ONNX Runtime. Opset 17 is broadly
# supported by both, which keeps one exported file usable across every backend the
# benchmark measures.
DEFAULT_OPSET = 17


def onnx_path_for(weights: str | Path) -> Path:
    """Where the exporter writes the ONNX file for a given checkpoint."""
    return Path(weights).with_suffix(".onnx")


def export_onnx(
    weights: str | Path,
    imgsz: int = 640,
    opset: int = DEFAULT_OPSET,
    dynamic: bool = False,
    simplify: bool = True,
) -> Path:
    """Export a checkpoint to ONNX and return the written path."""
    from ultralytics import YOLO

    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"{weights} missing. Run scripts/finetune.py first.")

    YOLO(str(weights)).export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        dynamic=dynamic,
        simplify=simplify,
    )
    out = onnx_path_for(weights)
    if not out.exists():
        raise RuntimeError(f"export reported success but {out} is missing")
    return out


def describe_onnx(path: str | Path) -> dict:
    """Read back the graph metadata worth checking after an export.

    The input and output shapes are the part that matters. A wrong input shape or an
    unexpectedly dynamic axis will not fail here, it will fail much later inside the
    runtime with a far less obvious message."""
    import onnx

    path = Path(path)
    model = onnx.load(str(path))

    def shape_of(value) -> list:
        dims = []
        for d in value.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.dim_value > 0 else (d.dim_param or "?"))
        return dims

    return {
        "path": path,
        "size_mb": round(path.stat().st_size / 1e6, 2),
        "opset": max(i.version for i in model.opset_import),
        "ir_version": model.ir_version,
        "inputs": {v.name: shape_of(v) for v in model.graph.input},
        "outputs": {v.name: shape_of(v) for v in model.graph.output},
    }
