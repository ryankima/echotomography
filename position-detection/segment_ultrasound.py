#!/usr/bin/env python3
"""SAM3 ultrasound segmentation using official Meta predictor API.

This script uses facebookresearch/sam3's request API:
    build_sam3_predictor -> start_session -> add_prompt -> propagate_in_video

Example:
python segment_ultrasound.py \
    --weights C:/Users/rngki/Downloads/sam3/sam3.pt \
    --video F:/2026-03-25 21-59-26.mp4 \
    --prompt heart \
    --debug-output test.mp4
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import os
import sys
import types
from typing import Optional

import cv2
import numpy as np
import torch


def choose_largest_mask(masks: np.ndarray) -> np.ndarray:
    if masks.ndim == 2:
        return masks
    if masks.ndim != 3 or masks.shape[0] == 0:
        raise RuntimeError("SAM3 returned masks with an unsupported shape.")
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    return masks[int(np.argmax(areas))]


def to_bool_mask(mask: np.ndarray) -> np.ndarray:
    if mask.dtype == np.bool_:
        return mask
    return mask > 0.5


def overlay_mask(frame_bgr: np.ndarray, mask: np.ndarray, alpha: float, label: str) -> np.ndarray:
    out = frame_bgr.copy()
    mask = to_bool_mask(mask)

    color = np.zeros_like(out)
    color[:, :, 2] = 255
    out[mask] = cv2.addWeighted(frame_bgr, 1.0 - alpha, color, alpha, 0)[mask]

    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 255), 2)
    cv2.putText(
        out,
        f"SAM3: {label}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


class SAM3Segmenter:
    """Adapter for official SAM3 predictor API from facebookresearch/sam3."""

    def __init__(self, weights_path: str, device: str, version: str) -> None:
        self._install_windows_edt_fallback_module()

        try:
            from sam3 import build_sam3_predictor
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Could not import `sam3`. If you are on Windows, ensure global installs include sam3 deps and keep this script's triton fallback enabled."
            ) from exc

        self._patch_sam3_edt_fallback()

        if device != "cuda":
            raise ValueError("Official SAM3 predictor currently expects CUDA in this script.")

        self.predictor = build_sam3_predictor(
            checkpoint_path=weights_path,
            version=version,
            compile=False,
            async_loading_frames=False,
        )

    @staticmethod
    def _install_windows_edt_fallback_module() -> None:
        """Preload sam3 EDT fallback module on Windows when triton is unavailable."""
        if os.name != "nt":
            return
        try:
            has_triton = importlib.util.find_spec("triton") is not None
            if has_triton:
                return
        except Exception:
            return

        edt_mod = types.ModuleType("sam3.model.edt")
        edt_mod.__spec__ = importlib.machinery.ModuleSpec("sam3.model.edt", loader=None)

        def _edt_cpu(data: torch.Tensor) -> torch.Tensor:
            if data.dim() != 3:
                raise RuntimeError(f"Expected [B,H,W] mask tensor, got shape {tuple(data.shape)}")

            device = data.device
            arr = data.detach().to("cpu").numpy().astype(np.uint8)
            out = np.zeros(arr.shape, dtype=np.float32)
            for i in range(arr.shape[0]):
                out[i] = cv2.distanceTransform(arr[i], cv2.DIST_L2, 0)
            return torch.from_numpy(out).to(device=device)

        edt_mod.edt_triton = _edt_cpu
        sys.modules["sam3.model.edt"] = edt_mod

    @staticmethod
    def _patch_sam3_edt_fallback() -> None:
        """Replace sam3 EDT implementation with a CPU OpenCV fallback."""
        import sam3.model.edt as sam3_edt  # type: ignore
        import sam3.model.sam3_tracker_utils as tracker_utils  # type: ignore

        def _edt_cpu(data: torch.Tensor) -> torch.Tensor:
            if data.dim() != 3:
                raise RuntimeError(f"Expected [B,H,W] mask tensor, got shape {tuple(data.shape)}")

            device = data.device
            arr = data.detach().to("cpu").numpy().astype(np.uint8)
            out = np.zeros(arr.shape, dtype=np.float32)
            for i in range(arr.shape[0]):
                out[i] = cv2.distanceTransform(arr[i], cv2.DIST_L2, 0)
            return torch.from_numpy(out).to(device=device)

        sam3_edt.edt_triton = _edt_cpu
        tracker_utils.edt_triton = _edt_cpu

    @staticmethod
    def _to_numpy_masks(maybe_mask) -> Optional[np.ndarray]:
        if maybe_mask is None:
            return None
        if hasattr(maybe_mask, "detach"):
            arr = maybe_mask.detach().cpu().numpy()
        else:
            arr = np.asarray(maybe_mask)
        if arr.size == 0:
            return None
        return arr

    @staticmethod
    def _extract_mask_from_payload(payload: dict) -> Optional[np.ndarray]:
        candidates = []
        for key in ("out_binary_masks", "masks", "mask"):
            if key in payload:
                candidates.append(payload[key])

        outputs = payload.get("outputs")
        if isinstance(outputs, dict):
            for key in ("out_binary_masks", "masks", "mask"):
                if key in outputs:
                    candidates.append(outputs[key])

        for c in candidates:
            arr = SAM3Segmenter._to_numpy_masks(c)
            if arr is None:
                continue
            while arr.ndim > 3:
                arr = np.squeeze(arr, axis=0)
            if arr.ndim == 3:
                return to_bool_mask(choose_largest_mask(arr))
            if arr.ndim == 2:
                return to_bool_mask(arr)
        return None

    def stream_video_masks(
        self,
        video_path: str,
        prompt: str,
        start_frame: int,
        output_prob_thresh: float,
    ):
        response = self.predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_path,
            )
        )
        session_id = response["session_id"]

        try:
            response = self.predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=start_frame,
                    text=prompt,
                    output_prob_thresh=output_prob_thresh,
                )
            )
            first_mask = self._extract_mask_from_payload(response)
            if first_mask is not None:
                yield start_frame, first_mask

            for out in self.predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                    propagation_direction="forward",
                    start_frame_index=start_frame,
                    output_prob_thresh=output_prob_thresh,
                )
            ):
                frame_idx = out.get("frame_index")
                if frame_idx is None:
                    continue
                mask = self._extract_mask_from_payload(out)
                if mask is not None:
                    yield int(frame_idx), mask
        finally:
            _ = self.predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id,
                )
            )


def default_output_path(video_path: str) -> str:
    stem, ext = os.path.splitext(video_path)
    if not ext:
        ext = ".mp4"
    return f"{stem}_segmented{ext}"


def process_video(
    segmenter: SAM3Segmenter,
    video_path: str,
    debug_output_path: str,
    prompt: str,
    alpha: float,
    start_frame: int,
    output_prob_thresh: float,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        debug_output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create output video: {debug_output_path}")

    current_frame = 0
    segmented = 0

    try:
        for out_frame_idx, mask in segmenter.stream_video_masks(
            video_path=video_path,
            prompt=prompt,
            start_frame=start_frame,
            output_prob_thresh=output_prob_thresh,
        ):
            if out_frame_idx < current_frame:
                continue

            while current_frame < out_frame_idx:
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)
                current_frame += 1

            ok, frame = cap.read()
            if not ok:
                break
            writer.write(overlay_mask(frame, mask, alpha=alpha, label=prompt))
            current_frame += 1
            segmented += 1

            if segmented % 30 == 0:
                print(f"Segmented {segmented} frames...")

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            current_frame += 1
    finally:
        cap.release()
        writer.release()

    print(f"Done. Segmented {segmented} frames.")
    print(f"Debug video written to: {debug_output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SAM3-only ultrasound MP4 segmentation")
    parser.add_argument("--weights", required=True, help="Path to local SAM3 weights")
    parser.add_argument("--video", required=True, help="Path to input MP4")
    parser.add_argument("--prompt", required=True, help="Organ prompt text, e.g. 'heart' or 'liver'")
    parser.add_argument("--device", default="cuda", help="Inference device: cuda or cpu")
    parser.add_argument("--version", default="sam3", choices=["sam3", "sam3.1"], help="SAM3 model version")
    parser.add_argument(
        "--debug-output",
        default=None,
        help="Output debug MP4 path (default: <input>_segmented.mp4)",
    )
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha")
    parser.add_argument("--start-frame", type=int, default=0, help="Frame index where prompt is added")
    parser.add_argument("--output-prob-thresh", type=float, default=0.5, help="SAM3 output probability threshold")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not os.path.isfile(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be >= 0")

    debug_output = args.debug_output or default_output_path(args.video)

    segmenter = SAM3Segmenter(weights_path=args.weights, device=args.device, version=args.version)
    process_video(
        segmenter=segmenter,
        video_path=args.video,
        debug_output_path=debug_output,
        prompt=args.prompt,
        alpha=args.alpha,
        start_frame=args.start_frame,
        output_prob_thresh=args.output_prob_thresh,
    )


if __name__ == "__main__":
    main()
