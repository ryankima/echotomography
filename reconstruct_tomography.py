#!/usr/bin/env python3
"""
3D Ultrasound Tomography Reconstructor
Reconstructs a 3D volume from synchronized camera + ultrasound video pairs.

Usage:
    python reconstruct_tomography.py --camera <camera.mp4> --capture <capture.mp4> \
        --output <volume.nii.gz> --voxel-size 0.5

Dependencies: opencv-python, numpy, scipy, nibabel, napari, apriltag
"""

import argparse
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional
import logging

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None

try:
    import nibabel as nib
except ImportError:
    nib = None

try:
    from pupil_apriltags import Detector
    APRILTAG_BACKEND = 'pupil'
except ImportError:
    Detector = None
    APRILTAG_BACKEND = 'fallback'

try:
    import napari
except ImportError:
    napari = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sonosite m-Turbo typical specs (looked up)
DEFAULT_ULTRASOUND_MM_PER_PIXEL = 0.074  # mm/pixel (mid-range estimate)
DEFAULT_ULTRASOUND_SLICE_THICKNESS_MM = 1.0  # mm
DEFAULT_VOXEL_SIZE_MM = 0.5  # mm


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float
    fy: float
    ppx: float
    ppy: float
    width: int
    height: int
    distortion: np.ndarray = None

    def to_matrix(self) -> np.ndarray:
        """Return 3x3 camera matrix."""
        return np.array([
            [self.fx, 0, self.ppx],
            [0, self.fy, self.ppy],
            [0, 0, 1]
        ], dtype=np.float32)


@dataclass
class Pose:
    """3D pose: position (m) + orientation (rotation matrix)."""
    position: np.ndarray  # shape (3,)
    rotation: np.ndarray  # shape (3, 3)

    def transform_point(self, p: np.ndarray) -> np.ndarray:
        """Transform a 3D point from probe frame to world frame."""
        return self.rotation @ p + self.position


class BoardConfiguration:
    """AprilTag board layout (from C++ code)."""
    TAG_SIZE_CM = 15.2
    EDGE_SPACING_CM = 9.0
    EDGE_SPACING_9_TO_1_CM = 10.75
    CENTER_TO_CENTER_CM = TAG_SIZE_CM + EDGE_SPACING_CM
    CENTER_TO_CENTER_9_TO_1_CM = TAG_SIZE_CM + EDGE_SPACING_9_TO_1_CM
    BEND_ANGLE_DEG = 25.0
    BEND_ANGLE_RAD = BEND_ANGLE_DEG * np.pi / 180.0
    BEND_SIN = 0.4226182617
    Z_OFFSET_BENT = CENTER_TO_CENTER_CM * BEND_SIN / 100.0  # convert to m

    @classmethod
    def get_tag_positions(cls) -> dict:
        """Return 3D board tag positions in meters (center of board is origin)."""
        positions = {}

        # Center column (straight): tags 1, 4, 7
        positions[1] = np.array([0.0, cls.CENTER_TO_CENTER_CM, 0.0]) / 100.0
        positions[4] = np.array([0.0, 0.0, 0.0]) / 100.0
        positions[7] = np.array([0.0, -cls.CENTER_TO_CENTER_CM, 0.0]) / 100.0

        # Left column (bent back): tags 0, 3, 6
        x_left = -cls.CENTER_TO_CENTER_CM / 100.0
        z_left = -cls.Z_OFFSET_BENT
        positions[0] = np.array([x_left, cls.CENTER_TO_CENTER_CM / 100.0, z_left])
        positions[3] = np.array([x_left, 0.0, z_left])
        positions[6] = np.array([x_left, -cls.CENTER_TO_CENTER_CM / 100.0, z_left])

        # Right column (bent back): tags 2, 5, 8
        x_right = cls.CENTER_TO_CENTER_CM / 100.0
        z_right = -cls.Z_OFFSET_BENT
        positions[2] = np.array([x_right, cls.CENTER_TO_CENTER_CM / 100.0, z_right])
        positions[5] = np.array([x_right, 0.0, z_right])
        positions[8] = np.array([x_right, -cls.CENTER_TO_CENTER_CM / 100.0, z_right])

        # Top tag 9 (bent back)
        y_top = (cls.CENTER_TO_CENTER_9_TO_1_CM + cls.CENTER_TO_CENTER_CM) / 100.0
        z_top = -cls.Z_OFFSET_BENT
        positions[9] = np.array([0.0, y_top, z_top])

        return positions

    @classmethod
    def get_tag_corners_m(cls) -> np.ndarray:
        """Return 4 corner offsets of a tag (in meters), relative to tag center."""
        half_tag_m = cls.TAG_SIZE_CM * 0.5 / 100.0
        return np.array([
            [-half_tag_m, -half_tag_m, 0.0],
            [half_tag_m, -half_tag_m, 0.0],
            [half_tag_m, half_tag_m, 0.0],
            [-half_tag_m, half_tag_m, 0.0]
        ], dtype=np.float32)


class UltrasoundWindowDetector:
    """Detect ultrasound window in capture frame."""

    @staticmethod
    def detect_window_roi(frame: np.ndarray, margin_px: int = 50) -> Tuple[int, int, int, int]:
        """
        Auto-detect ultrasound window (circular/fan ROI) in capture frame.
        Returns (x_min, y_min, x_max, y_max) bounding box.
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Threshold to find bright region (typical ultrasound display)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        # Morphological ops to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            logger.warning("No contours found; using center crop.")
            h, w = gray.shape
            return max(0, w // 4), max(0, h // 4), min(w, 3 * w // 4), min(h, 3 * h // 4)

        # Find largest contour (assume it's the ultrasound window)
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        # Add margin
        x = max(0, x - margin_px)
        y = max(0, y - margin_px)
        x_max = min(frame.shape[1], x + w + 2 * margin_px)
        y_max = min(frame.shape[0], y + h + 2 * margin_px)

        return x, y, x_max, y_max

    @staticmethod
    def unwrap_fan_slice(frame: np.ndarray, roi: Tuple[int, int, int, int],
                         angle_samples: int = 512, radius_samples: int = 512) -> np.ndarray:
        """Convert the ultrasound fan region into a rectangular slice.

        The capture card shows a sector scan, not a Cartesian image. Mapping the raw
        display directly into 3D produces streaks, so we unwrap the fan into a
        radius-by-angle image first.
        """
        x_min, y_min, x_max, y_max = roi
        cropped = frame[y_min:y_max, x_min:x_max]
        if cropped.size == 0:
            return cropped

        if len(cropped.shape) == 3:
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        else:
            gray = cropped.copy()

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 18, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(
            thresh,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        )

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return gray

        contour = max(contours, key=cv2.contourArea)
        contour_points = contour[:, 0, :].astype(np.float32)

        top_y = float(np.min(contour_points[:, 1]))
        top_band = contour_points[contour_points[:, 1] <= top_y + 3.0]
        if len(top_band) == 0:
            center_x = float(np.mean(contour_points[:, 0]))
        else:
            center_x = float(np.mean(top_band[:, 0]))

        center_y = top_y
        center = (center_x, center_y)

        distances = np.sqrt((contour_points[:, 0] - center_x) ** 2 + (contour_points[:, 1] - center_y) ** 2)
        max_radius = int(np.ceil(np.max(distances)))
        max_radius = max(max_radius, 32)

        polar = cv2.warpPolar(
            gray,
            (angle_samples, radius_samples),
            center,
            max_radius,
            cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS,
        )

        # Keep only angles that actually contain energy from the sector scan.
        angle_energy = polar.mean(axis=1)
        if np.max(angle_energy) > 0:
            keep = angle_energy > (0.12 * np.max(angle_energy))
            if np.any(keep):
                indices = np.flatnonzero(keep)
                start = int(indices[0])
                stop = int(indices[-1]) + 1
                polar = polar[start:stop, :]

        # Convert to radius-by-angle layout so the next stage treats depth vertically.
        unwrapped = polar.T
        return unwrapped


class AprilTagDetector:
    """Detect AprilTags and estimate camera pose."""

    def __init__(self, intrinsics: CameraIntrinsics, use_fallback: bool = False):
        self.intrinsics = intrinsics
        self.board_positions = BoardConfiguration.get_tag_positions()
        self.tag_corners_m = BoardConfiguration.get_tag_corners_m()
        self.use_fallback = use_fallback or Detector is None
        self.detector = None
        self.previous_pose: Optional[Pose] = None

        if not self.use_fallback:
            try:
                self.detector = Detector()
                logger.info("AprilTag detector initialized (pupil-apriltags)")
            except Exception as e:
                logger.warning(f"Failed to initialize AprilTag detector: {e}. Using fallback mode.")
                self.use_fallback = True
        else:
            if Detector is None:
                logger.info("AprilTag detection unavailable (pupil-apriltags not installed). "
                           "Using fallback mode—camera poses will be identity.")
            else:
                logger.info("Using fallback mode (identity poses) for testing.")

    def estimate_camera_pose(self, frame: np.ndarray) -> Optional[Pose]:
        """Detect AprilTags and estimate camera pose in board frame."""
        if self.use_fallback:
            # Return identity pose (camera at origin, no rotation)
            return Pose(position=np.zeros(3, dtype=np.float32), 
                       rotation=np.eye(3, dtype=np.float32))

        if self.detector is None:
            return None

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        try:
            detections = self.detector.detect(gray)
        except Exception as e:
            logger.warning(f"AprilTag detection failed: {e}")
            return None

        if not detections:
            return None

        # Collect 3D-2D correspondences
        object_points = []
        image_points = []

        for det in detections:
            tag_id = det.tag_id
            if tag_id not in self.board_positions:
                continue

            # Use all 4 corners of the tag
            tag_center_m = self.board_positions[tag_id]
            for corner_idx, corner_offset in enumerate(self.tag_corners_m):
                # 3D point in board frame
                p_3d = tag_center_m + corner_offset
                object_points.append(p_3d)

                # 2D point in image (det.corners is shape (4, 2))
                p_2d = det.corners[corner_idx]
                image_points.append(p_2d)

        if len(object_points) < 4:
            return None

        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)
        camera_matrix = self.intrinsics.to_matrix()
        dist_coeffs = self.intrinsics.distortion if self.intrinsics.distortion is not None else np.zeros(5)

        # Solve PnP
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points, image_points, camera_matrix, dist_coeffs,
            iterationsCount=100, reprojectionError=4.0, confidence=0.99
        )

        if not success or inliers is None or len(inliers) < 4:
            return None

        if len(inliers) < 6:
            # Four inliers is enough to solve PnP, but it is too fragile for this scan.
            return None

        # Convert rvec to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        R = R.astype(np.float32)

        # Camera position in board frame: -R^T * t
        tvec = tvec.astype(np.float32).flatten()
        camera_pos = -(R.T @ tvec)

        pose = Pose(position=camera_pos, rotation=R.T)

        if self.previous_pose is not None:
            jump = float(np.linalg.norm(pose.position - self.previous_pose.position))
            if jump > 0.18:
                logger.debug("Rejected pose jump of %.3f m", jump)
                return None

        self.previous_pose = pose
        return pose


def _rotation_to_vector(rotation: np.ndarray) -> np.ndarray:
    vector, _ = cv2.Rodrigues(rotation.astype(np.float32))
    return vector.reshape(3)


def _vector_to_rotation(vector: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(vector.astype(np.float32).reshape(3, 1))
    return rotation.astype(np.float32)


def interpolate_poses(raw_poses: list[Optional[Pose]]) -> list[Optional[Pose]]:
    """Fill missing poses using linear translation interpolation and Rodrigues-vector rotation interpolation."""
    interpolated: list[Optional[Pose]] = [None] * len(raw_poses)
    valid_indices = [index for index, pose in enumerate(raw_poses) if pose is not None]
    if not valid_indices:
        return interpolated

    for index in range(len(raw_poses)):
        pose = raw_poses[index]
        if pose is not None:
            interpolated[index] = pose
            continue

        prev_index = max((valid for valid in valid_indices if valid < index), default=None)
        next_index = min((valid for valid in valid_indices if valid > index), default=None)

        if prev_index is None and next_index is None:
            continue
        if prev_index is None:
            interpolated[index] = raw_poses[next_index]
            continue
        if next_index is None:
            interpolated[index] = raw_poses[prev_index]
            continue

        prev_pose = raw_poses[prev_index]
        next_pose = raw_poses[next_index]
        assert prev_pose is not None and next_pose is not None

        alpha = (index - prev_index) / float(next_index - prev_index)
        position = (1.0 - alpha) * prev_pose.position + alpha * next_pose.position

        prev_vector = _rotation_to_vector(prev_pose.rotation)
        next_vector = _rotation_to_vector(next_pose.rotation)
        rotation = _vector_to_rotation((1.0 - alpha) * prev_vector + alpha * next_vector)

        interpolated[index] = Pose(position=position.astype(np.float32), rotation=rotation)

    return interpolated


def compute_volume_bounds(
    slice_images: list[np.ndarray],
    probe_poses: list[Optional[Pose]],
    px_to_m: float,
    margin_m: float = 0.03,
) -> np.ndarray:
    """Compute world-space bounds that cover every reconstructed slice."""
    pose_positions = [pose.position for pose in probe_poses if pose is not None]
    if not pose_positions:
        raise RuntimeError("No valid poses available to compute volume bounds")

    pose_positions = np.asarray(pose_positions, dtype=np.float32)
    center = np.median(pose_positions, axis=0)
    spread = np.percentile(np.abs(pose_positions - center[None, :]), 75, axis=0)
    spread = np.maximum(spread, np.array([0.03, 0.03, 0.03], dtype=np.float32))

    # Use a robust window around the pose track rather than the raw min/max,
    # which can be dominated by a few bad poses and blow up memory.
    bounds_min = center - spread - margin_m
    bounds_max = center + spread + margin_m

    for slice_image, probe_pose in zip(slice_images, probe_poses):
        if probe_pose is None:
            continue

        h, w = slice_image.shape
        half_w = 0.5 * w * px_to_m
        half_h = 0.5 * h * px_to_m

        local_corners = np.array([
            [-half_w, -half_h, 0.0],
            [half_w, -half_h, 0.0],
            [half_w, half_h, 0.0],
            [-half_w, half_h, 0.0],
        ], dtype=np.float32)
        world_corners = (probe_pose.rotation @ local_corners.T).T + probe_pose.position[None, :]

        bounds_min = np.minimum(bounds_min, np.min(world_corners, axis=0) - margin_m)
        bounds_max = np.maximum(bounds_max, np.max(world_corners, axis=0) + margin_m)

    if not np.isfinite(bounds_min).all() or not np.isfinite(bounds_max).all():
        raise RuntimeError("Unable to compute volume bounds from poses")

    bounds_min -= margin_m
    bounds_max += margin_m
    return np.stack([bounds_min, bounds_max], axis=1)


def log_pose_statistics(cached_poses: list[Optional[Pose]]) -> None:
    """Log simple motion statistics for the valid pose track."""
    valid_positions = np.asarray([pose.position for pose in cached_poses if pose is not None], dtype=np.float32)
    if len(valid_positions) < 2:
        logger.warning("Not enough valid poses to compute motion statistics")
        return

    deltas = np.linalg.norm(np.diff(valid_positions, axis=0), axis=1)
    logger.info(
        "Pose motion stats (m): median step=%.4f, 90th=%.4f, max=%.4f",
        float(np.median(deltas)),
        float(np.percentile(deltas, 90)),
        float(np.max(deltas)),
    )


class ProbeToWorldTransform:
    """Manages probe → world frame transformation."""

    def __init__(self, probe_to_camera_t: np.ndarray, probe_to_camera_R: np.ndarray):
        """
        Args:
            probe_to_camera_t: translation (3,) in camera frame
            probe_to_camera_R: rotation matrix (3, 3)
        """
        self.t = probe_to_camera_t.astype(np.float32)
        self.R = probe_to_camera_R.astype(np.float32)

    def camera_pose_to_probe_pose(self, camera_pose: Pose) -> Pose:
        """Transform camera pose (in world) to probe pose (in world)."""
        # Probe position in world: world_from_camera(probe_in_camera)
        probe_in_camera = self.t
        probe_in_world = camera_pose.transform_point(probe_in_camera)

        # Probe rotation in world: camera_rotation @ probe_rotation_in_camera
        probe_R_in_world = camera_pose.rotation @ self.R

        return Pose(position=probe_in_world, rotation=probe_R_in_world)


class VolumeReconstructor:
    """Accumulate ultrasound slices into 3D volume."""

    MAX_GRID_DIM = 512

    def __init__(self, voxel_size_mm: float, volume_bounds_mm: Optional[np.ndarray] = None):
        """
        Args:
            voxel_size_mm: voxel size in mm
            volume_bounds_mm: [[x_min, x_max], [y_min, y_max], [z_min, z_max]] in mm,
                             or None to auto-set based on first frame
        """
        self.voxel_size_m = voxel_size_mm / 1000.0
        self.volume_bounds_m = volume_bounds_mm / 1000.0 if volume_bounds_mm is not None else None
        self.volume = None
        self.weight = None

    def _initialize_volume(self, center: np.ndarray, size_m: float = 0.2):
        """Initialize volume grid centered around probe center."""
        half_size = size_m / 2.0
        bounds = np.array([
            [center[0] - half_size, center[0] + half_size],
            [center[1] - half_size, center[1] + half_size],
            [center[2] - half_size, center[2] + half_size]
        ])
        self._set_grid_from_bounds(bounds)

    def _set_grid_from_bounds(self, bounds_m: np.ndarray):
        """Set up voxel grid from bounds."""
        extents = bounds_m[:, 1] - bounds_m[:, 0]
        required_voxel_size = float(np.max(extents) / self.MAX_GRID_DIM)
        if required_voxel_size > self.voxel_size_m:
            logger.warning(
                "Requested voxel size %.4f m would create an oversized grid; using %.4f m to preserve bounds.",
                self.voxel_size_m,
                required_voxel_size,
            )
            self.voxel_size_m = required_voxel_size

        nx = int(np.ceil(extents[0] / self.voxel_size_m))
        ny = int(np.ceil(extents[1] / self.voxel_size_m))
        nz = int(np.ceil(extents[2] / self.voxel_size_m))

        self.volume = np.zeros((nx, ny, nz), dtype=np.float32)
        self.weight = np.zeros((nx, ny, nz), dtype=np.float32)
        self.bounds_m = bounds_m

    def set_bounds(self, bounds_m: np.ndarray):
        """Explicitly initialize the voxel grid with precomputed bounds."""
        self._set_grid_from_bounds(bounds_m)

    def add_slice(self, slice_image: np.ndarray, probe_pose: Pose, 
                  px_to_mm: float, thickness_mm: float):
        """
        Add ultrasound slice to volume.
        
        Args:
            slice_image: 2D ultrasound image
            probe_pose: probe pose in world frame
            px_to_mm: pixel-to-mm scaling
            thickness_mm: slice thickness in mm
        """
        if self.volume is None:
            self._initialize_volume(probe_pose.position)

        h, w = slice_image.shape
        px_to_m = px_to_mm / 1000.0
        thickness_m = thickness_mm / 1000.0

        # Create 2D grid of points in probe frame
        # Assume probe X-Y plane is the image plane, probe Z is scanning direction
        y_indices, x_indices = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        # Normalize to [-0.5, 0.5] and scale to physical distance
        x_probe = (x_indices - w / 2.0) * px_to_m
        y_probe = (y_indices - h / 2.0) * px_to_m
        z_probe = np.zeros_like(x_probe)

        # Transform to world frame
        points_probe = np.stack([x_probe.flat, y_probe.flat, z_probe.flat], axis=1)
        points_world = probe_pose.rotation @ points_probe.T + probe_pose.position[:, None]
        points_world = points_world.T

        # Convert to voxel indices
        voxel_x = ((points_world[:, 0] - self.bounds_m[0, 0]) / self.voxel_size_m).astype(int)
        voxel_y = ((points_world[:, 1] - self.bounds_m[1, 0]) / self.voxel_size_m).astype(int)
        voxel_z = ((points_world[:, 2] - self.bounds_m[2, 0]) / self.voxel_size_m).astype(int)

        # Filter valid voxels
        valid = (
            (voxel_x >= 0) & (voxel_x < self.volume.shape[0]) &
            (voxel_y >= 0) & (voxel_y < self.volume.shape[1]) &
            (voxel_z >= 0) & (voxel_z < self.volume.shape[2])
        )

        voxel_x = voxel_x[valid]
        voxel_y = voxel_y[valid]
        voxel_z = voxel_z[valid]
        intensities = slice_image.flat[valid]

        # Accumulate
        np.add.at(self.volume, (voxel_x, voxel_y, voxel_z), intensities)
        np.add.at(self.weight, (voxel_x, voxel_y, voxel_z), 1.0)

    def finalize(self) -> np.ndarray:
        """Average accumulated volume and return."""
        with np.errstate(divide='ignore', invalid='ignore'):
            volume = np.divide(self.volume, self.weight, 
                              out=np.zeros_like(self.volume), 
                              where=self.weight > 0)
        return volume.astype(np.uint8)


def load_intrinsics(args) -> CameraIntrinsics:
    """Load camera intrinsics from arguments or file."""
    if args.intrinsics_json:
        with open(args.intrinsics_json) as f:
            data = json.load(f)
        return CameraIntrinsics(
            fx=data['fx'], fy=data['fy'],
            ppx=data['ppx'], ppy=data['ppy'],
            width=data['width'], height=data['height'],
            distortion=np.array(data.get('distortion', [0]*5), dtype=np.float32)
        )
    else:
        # Use hardcoded values from user
        return CameraIntrinsics(
            fx=1377.98, fy=1376.15,
            ppx=961.974, ppy=522.977,
            width=1920, height=1080,
            distortion=np.zeros(5, dtype=np.float32)
        )


def create_probe_to_camera_transform(args) -> ProbeToWorldTransform:
    """Create probe→camera transform."""
    # Probe offset: length→Y, left→Z, up→X
    # User gave: 16.5 cm along Y, 2.5 cm left (negative Z), 4 cm up (positive X)
    t = np.array([args.probe_up_cm, args.probe_length_cm, -args.probe_left_cm], dtype=np.float32) / 100.0

    # Probe rotation: probe_Z → camera_Y (scanning axis aligned with length)
    # Assuming standard orientation: probe_Z→camera_Y, probe_X→camera_Z, probe_Y→camera_X
    R = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ], dtype=np.float32)

    return ProbeToWorldTransform(t, R)


def main():
    parser = argparse.ArgumentParser(description="3D Ultrasound Tomography Reconstructor")
    parser.add_argument('--camera', required=True, help='Camera video (MP4)')
    parser.add_argument('--capture', required=True, help='Capture card video (MP4)')
    parser.add_argument('--output', default='volume.nii.gz', help='Output volume file')
    parser.add_argument('--voxel-size', type=float, default=DEFAULT_VOXEL_SIZE_MM,
                        help='Voxel size in mm')
    parser.add_argument('--us-mm-per-pixel', type=float, default=DEFAULT_ULTRASOUND_MM_PER_PIXEL,
                        help='Ultrasound pixel spacing (mm/pixel)')
    parser.add_argument('--us-thickness-mm', type=float, default=DEFAULT_ULTRASOUND_SLICE_THICKNESS_MM,
                        help='Ultrasound slice thickness (mm)')
    parser.add_argument('--intrinsics-json', help='Camera intrinsics JSON file')
    parser.add_argument('--probe-length-cm', type=float, default=16.5,
                        help='Probe length offset (cm)')
    parser.add_argument('--probe-left-cm', type=float, default=2.5,
                        help='Probe left offset (cm)')
    parser.add_argument('--probe-up-cm', type=float, default=4.0,
                        help='Probe up offset (cm)')
    parser.add_argument('--no-viewer', action='store_true', help='Skip Napari viewer')
    parser.add_argument('--force-fallback-poses', action='store_true',
                        help='Force fallback identity poses even if AprilTag detection is available')
    parser.add_argument('--smooth-sigma', type=float, default=1.2,
                        help='Gaussian smoothing sigma in voxels applied before saving')

    args = parser.parse_args()

    # Load intrinsics
    logger.info("Loading camera intrinsics...")
    intrinsics = load_intrinsics(args)
    logger.info(f"  fx={intrinsics.fx}, fy={intrinsics.fy}, ppx={intrinsics.ppx}, ppy={intrinsics.ppy}")

    # Initialize AprilTag detector
    logger.info("Initializing AprilTag detector...")
    tag_detector = AprilTagDetector(intrinsics, use_fallback=args.force_fallback_poses)

    # Load videos
    logger.info(f"Opening camera video: {args.camera}")
    cap_camera = cv2.VideoCapture(args.camera)
    if not cap_camera.isOpened():
        logger.error(f"Failed to open {args.camera}")
        sys.exit(1)

    logger.info(f"Opening capture video: {args.capture}")
    cap_capture = cv2.VideoCapture(args.capture)
    if not cap_capture.isOpened():
        logger.error(f"Failed to open {args.capture}")
        sys.exit(1)

    frame_count_camera = int(cap_camera.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count_capture = int(cap_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"  Camera frames: {frame_count_camera}, Capture frames: {frame_count_capture}")

    # Detect ultrasound window from middle frame
    logger.info("Detecting ultrasound window...")
    mid_frame_idx = frame_count_capture // 2
    cap_capture.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
    ret, mid_frame = cap_capture.read()
    cap_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if not ret:
        logger.error("Failed to read middle frame from capture video")
        sys.exit(1)

    us_roi = UltrasoundWindowDetector.detect_window_roi(mid_frame)
    logger.info(f"  Ultrasound ROI: {us_roi}")

    us_preview = UltrasoundWindowDetector.unwrap_fan_slice(mid_frame, us_roi)
    if us_preview.size > 0:
        logger.info(f"  Unwrapped ultrasound slice size: {us_preview.shape}")

    # Initialize reconstructor
    logger.info("Initializing volume reconstructor...")
    reconstructor = VolumeReconstructor(args.voxel_size)

    # Create probe→camera transform
    probe_to_cam = create_probe_to_camera_transform(args)

    # Process frames and cache synchronized slices/poses so we can interpolate gaps.
    logger.info("Processing frames...")
    frame_idx = 0
    raw_pose_hits = 0
    skipped_pose = 0
    skipped_read = 0
    cached_slices: list[np.ndarray] = []
    cached_poses: list[Optional[Pose]] = []

    while True:
        ret_cam, frame_camera = cap_camera.read()
        ret_cap, frame_capture = cap_capture.read()

        if not ret_cam or not ret_cap:
            break

        # Estimate camera pose
        camera_pose = tag_detector.estimate_camera_pose(frame_camera)
        if camera_pose is None:
            skipped_pose += 1
        else:
            raw_pose_hits += 1

        # Transform to probe frame when we have a pose; keep None otherwise.
        probe_pose = probe_to_cam.camera_pose_to_probe_pose(camera_pose) if camera_pose is not None else None

        # Unwrap the sector scan into a rectangular slice before depositing it.
        us_slice = UltrasoundWindowDetector.unwrap_fan_slice(frame_capture, us_roi)
        if us_slice.size == 0:
            skipped_read += 1
            frame_idx += 1
            continue

        cached_slices.append(us_slice)
        cached_poses.append(probe_pose)

        if (frame_idx + 1) % 50 == 0:
            logger.info(f"  Cached {len(cached_slices)} slices, raw pose hits {raw_pose_hits}, missing pose {skipped_pose}")

        frame_idx += 1

    cap_camera.release()
    cap_capture.release()

    interpolated_poses = interpolate_poses(cached_poses)
    filled_count = sum(
        1 for raw_pose, interpolated_pose in zip(cached_poses, interpolated_poses)
        if raw_pose is None and interpolated_pose is not None
    )
    logger.info(
        f"Reconstruction complete: {len(cached_slices)} synchronized slices, "
        f"{raw_pose_hits} raw poses, {skipped_pose} missing poses"
    )
    logger.info(f"Interpolated poses for {filled_count} cached slices")
    log_pose_statistics(interpolated_poses)

    if cached_slices:
        px_to_m = args.us_mm_per_pixel / 1000.0
        bounds_m = compute_volume_bounds(cached_slices, interpolated_poses, px_to_m)
        reconstructor.set_bounds(bounds_m)
        logger.info(
            "Volume bounds (m): "
            f"x=[{bounds_m[0,0]:.3f}, {bounds_m[0,1]:.3f}], "
            f"y=[{bounds_m[1,0]:.3f}, {bounds_m[1,1]:.3f}], "
            f"z=[{bounds_m[2,0]:.3f}, {bounds_m[2,1]:.3f}]"
        )

    # Add every slice to the volume using the interpolated pose track.
    for slice_image, probe_pose in zip(cached_slices, interpolated_poses):
        if probe_pose is None:
            skipped_read += 1
            continue
        reconstructor.add_slice(slice_image, probe_pose, args.us_mm_per_pixel, args.us_thickness_mm)

    # Finalize volume
    logger.info("Finalizing volume...")
    volume = reconstructor.finalize()
    if args.smooth_sigma > 0:
        if gaussian_filter is None:
            logger.warning("scipy is not available; skipping smoothing")
        else:
            logger.info(f"Applying Gaussian smoothing (sigma={args.smooth_sigma})...")
            volume = gaussian_filter(volume.astype(np.float32), sigma=args.smooth_sigma)
            volume = np.clip(volume, 0, 255).astype(np.uint8)
    logger.info(f"  Volume shape: {volume.shape}, dtype: {volume.dtype}, range: [{volume.min()}, {volume.max()}]")

    # Save volume
    logger.info(f"Saving volume to {args.output}...")
    if args.output.endswith('.nii') or args.output.endswith('.nii.gz'):
        if nib is None:
            logger.error("nibabel not installed. Install: pip install nibabel")
            sys.exit(1)
        # Create affine (identity for now)
        affine = np.eye(4)
        nib_img = nib.Nifti1Image(volume, affine)
        nib.save(nib_img, args.output)
    else:
        np.save(args.output, volume)
    logger.info("  Saved!")

    # View in Napari
    if not args.no_viewer:
        if napari is None:
            logger.warning("napari not installed. Install: pip install napari[all]")
        else:
            logger.info("Opening Napari viewer (close window to exit)...")
            try:
                viewer = napari.Viewer()
                viewer.add_image(volume, name='Ultrasound Volume', colormap='gray')
                napari.run()
            except Exception as e:
                logger.warning(f"Failed to open Napari viewer: {e}")
                logger.info(f"Volume saved to {args.output}. Open manually with your viewer of choice.")


if __name__ == '__main__':
    main()
