"""
Real-time upper-body tracking using MediaPipe Pose to drive the AeroPiper arm.
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Callable, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore[import]
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "OpenCV (cv2) is required for webcam arm tracking. "
        "Activate the 'aeropiper' conda environment and install it via "
        "`conda activate aeropiper && pip install opencv-python`."
    ) from exc

try:
    import mediapipe as mp  # type: ignore[import]
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "MediaPipe is required for upper-body landmark tracking. "
        "Activate the 'aeropiper' conda environment and install it via "
        "`conda activate aeropiper && pip install mediapipe`."
    ) from exc

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="SymbolDatabase.GetPrototype",
)

MP_POSE = mp.solutions.pose
MP_DRAW = mp.solutions.drawing_utils
MP_CONNECTIONS = MP_POSE.POSE_CONNECTIONS

ARM_LABELS = [
    "BaseYaw",
    "ShoulderPitch",
    "ShoulderRoll",
    "ElbowFlex",
    "WristPitch",
    "WristRoll",
]


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return np.zeros_like(vec)
    return vec / norm


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    v1_u = _normalize(v1)
    v2_u = _normalize(v2)
    dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return float(np.arccos(dot))


class ArmGestureController:
    """
    Track the user's left arm with a webcam and emit six normalized DOFs in [-1, 1]:

    [base_yaw, shoulder_pitch, shoulder_roll, elbow_flex, wrist_pitch, wrist_roll]
    """

    def __init__(
        self,
        camera_index: int = 0,
        show_preview: bool = True,
        mirror_preview: bool = True,
        smoothing: float = 0.5,
        idle_decay: float = 0.96,
        visibility_threshold: float = 0.45,
        max_step: float = 0.35,
        frame_provider: Optional[Callable[[], Optional[np.ndarray]]] = None,
    ):
        self._camera_index = camera_index
        self._show_preview = show_preview
        self._mirror_preview = mirror_preview
        self._alpha = float(np.clip(smoothing, 0.0, 1.0))
        self._idle_decay = float(np.clip(idle_decay, 0.80, 0.999))
        self._visibility_threshold = visibility_threshold
        self._max_step = float(np.clip(max_step, 0.05, 1.5))
        self._preview_window = "AeroPiper Left Arm"

        self._frame_provider = frame_provider
        self._owns_capture = frame_provider is None
        self._capture = self._open_camera(camera_index) if self._owns_capture else None
        self._pose = MP_POSE.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.5,
        )

        self._arm = np.zeros(len(ARM_LABELS), dtype=np.float32)
        self._initialized = False
        self._stop_requested = False
        self._last_detection_time = 0.0
        self._last_measurement: Optional[np.ndarray] = None
        self._last_landmarks = None
        self._elbow_raw_min: Optional[float] = None
        self._elbow_raw_max: Optional[float] = None
        self._elbow_calibration_samples = 0

    def _open_camera(self, index: int):
        if os.name == "nt":
            capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        else:
            capture = cv2.VideoCapture(index)
        if not capture.isOpened():
            raise RuntimeError(
                f"Unable to open webcam index {index}. "
                "Activate the 'aeropiper' conda environment and ensure a camera is connected."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        capture.set(cv2.CAP_PROP_FPS, 30)
        return capture

    def update(self) -> np.ndarray:
        """Read the current frame, update pose estimates, and return 6 normalized DOFs."""

        if self._stop_requested:
            return self._arm.copy()

        frame = self._read_frame()
        if frame is None:
            self._stop_requested = True
            return self._arm.copy()

        measurement, annotated = self._extract_measurement(frame)
        now = time.time()

        if measurement is not None:
            measurement = self._limit_step(measurement)
            if not self._initialized:
                self._arm = measurement
                self._initialized = True
            else:
                self._arm = (
                    self._alpha * measurement + (1.0 - self._alpha) * self._arm
                )
            self._last_detection_time = now
        elif self._initialized and now - self._last_detection_time > 0.3:
            self._arm *= self._idle_decay
            if self._last_measurement is not None:
                self._last_measurement *= self._idle_decay

        if self._show_preview:
            display = annotated if annotated is not None else frame
            self._render_preview(display)

        return self._arm.copy()

    def should_stop(self) -> bool:
        return self._stop_requested

    def close(self):
        if self._owns_capture and self._capture is not None:
            self._capture.release()
            self._capture = None
        if hasattr(self._pose, "close"):
            self._pose.close()
        if self._show_preview:
            try:
                cv2.destroyWindow(self._preview_window)
            except cv2.error:
                pass

    def _read_frame(self) -> Optional[np.ndarray]:
        if self._frame_provider is not None:
            frame = self._frame_provider()
            if frame is None:
                return None
            return frame
        if self._capture is None:
            return None
        ok, frame = self._capture.read()
        if not ok:
            return None
        return frame

    def _extract_measurement(
        self, frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)
        if not results.pose_landmarks:
            self._last_landmarks = None
            return None, None

        coords = self._landmarks_to_array(results.pose_landmarks.landmark)
        measurement = self._landmarks_to_arm(coords)
        if measurement is None:
            return None, None

        annotated = frame.copy()
        MP_DRAW.draw_landmarks(
            annotated,
            results.pose_landmarks,
            MP_CONNECTIONS,
            landmark_drawing_spec=MP_DRAW.DrawingSpec(color=(0, 255, 0), thickness=2),
            connection_drawing_spec=MP_DRAW.DrawingSpec(color=(255, 255, 255), thickness=2),
        )
        self._last_landmarks = results.pose_landmarks
        return measurement, annotated

    def _landmarks_to_array(self, landmarks):
        coords = np.zeros((len(landmarks), 4), dtype=np.float32)
        for idx, lm in enumerate(landmarks):
            coords[idx, 0] = lm.x
            coords[idx, 1] = lm.y
            coords[idx, 2] = lm.z
            coords[idx, 3] = getattr(lm, "visibility", 1.0)
        # Convert to a camera-centric coordinate frame: +X right, +Y up, +Z forward.
        coords[:, 1] = 1.0 - coords[:, 1]  # invert y so upward = positive
        coords[:, 2] = -coords[:, 2]       # invert z so forward = positive
        return coords

    def _landmarks_to_arm(self, coords: np.ndarray) -> Optional[np.ndarray]:
        lm = MP_POSE.PoseLandmark
        required = [
            lm.LEFT_SHOULDER,
            lm.RIGHT_SHOULDER,
            lm.LEFT_ELBOW,
            lm.LEFT_WRIST,
            lm.LEFT_HIP,
            lm.LEFT_INDEX,
            lm.LEFT_PINKY,
        ]
        visibilities = coords[:, 3]
        for landmark in required:
            if visibilities[landmark.value] < self._visibility_threshold:
                return None

        left_shoulder = coords[lm.LEFT_SHOULDER.value, :3]
        right_shoulder = coords[lm.RIGHT_SHOULDER.value, :3]
        left_elbow = coords[lm.LEFT_ELBOW.value, :3]
        left_wrist = coords[lm.LEFT_WRIST.value, :3]
        left_hip = coords[lm.LEFT_HIP.value, :3]
        left_index = coords[lm.LEFT_INDEX.value, :3]
        left_pinky = coords[lm.LEFT_PINKY.value, :3]

        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        scale = max(shoulder_width, 1e-3)

        upper = (left_elbow - left_shoulder) / scale
        forearm = (left_wrist - left_elbow) / scale
        wrist_from_shoulder = (left_wrist - left_shoulder) / scale
        torso_vec = _normalize((left_hip - left_shoulder) / scale)
        hand_dir = (left_index - left_wrist) / scale
        palm_span = (left_index - left_pinky) / scale

        if np.linalg.norm(forearm) < 1e-4 or np.linalg.norm(upper) < 1e-4:
            return None

        base_yaw = np.arctan2(wrist_from_shoulder[0], wrist_from_shoulder[2] + 1e-5)
        shoulder_pitch = np.arctan2(
            wrist_from_shoulder[1], np.linalg.norm(wrist_from_shoulder[[0, 2]]) + 1e-5
        )
        shoulder_roll = np.arctan2(
            wrist_from_shoulder[0],
            np.linalg.norm([wrist_from_shoulder[1], wrist_from_shoulder[2]]) + 1e-5,
        )

        elbow_angle = _angle_between(-upper, forearm)
        elbow_raw = 2.0 * ((np.pi - elbow_angle) / np.pi) - 1.0  # -1 straight, +1 fully bent
        elbow_flex = self._normalize_elbow(elbow_raw)

        forearm_unit = _normalize(forearm)
        palm_perp = palm_span - np.dot(palm_span, forearm_unit) * forearm_unit
        palm_perp = _normalize(palm_perp)

        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        ref = world_up - np.dot(world_up, forearm_unit) * forearm_unit
        if np.linalg.norm(ref) < 1e-4:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            ref = ref - np.dot(ref, forearm_unit) * forearm_unit
        ref = _normalize(ref)

        wrist_pitch = float(np.clip(np.dot(_normalize(hand_dir), _normalize(-forearm)), -1.0, 1.0))

        wrist_roll = np.arctan2(
            np.dot(np.cross(ref, palm_perp), forearm_unit),
            np.dot(ref, palm_perp) + 1e-5,
        )

        values = np.asarray(
            [
                base_yaw / np.deg2rad(100.0),
                shoulder_pitch / np.deg2rad(100.0),
                shoulder_roll / np.deg2rad(80.0),
                elbow_flex,
                wrist_pitch,
                wrist_roll / np.deg2rad(135.0),
            ],
            dtype=np.float32,
        )

        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(values, -1.0, 1.0)

    def _limit_step(self, measurement: np.ndarray) -> np.ndarray:
        limited = np.asarray(measurement, dtype=np.float32)
        if self._last_measurement is None:
            self._last_measurement = limited
            return limited
        delta = limited - self._last_measurement
        bounded = self._last_measurement + np.clip(delta, -self._max_step, self._max_step)
        self._last_measurement = bounded
        return bounded

    def _render_preview(self, frame):
        if frame is None or not self._show_preview:
            return
        overlay = frame.copy()
        self.annotate_frame(overlay)
        if self._mirror_preview:
            overlay = cv2.flip(overlay, 1)
        stats = " ".join(f"{name}:{val:+4.2f}" for name, val in zip(ARM_LABELS, self._arm))
        cv2.putText(
            overlay,
            "Left arm tracking (Q/ESC exit)",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            stats,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        try:
            cv2.imshow(self._preview_window, overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                self._stop_requested = True
        except cv2.error:
            print("OpenCV preview unavailable, continuing without it.")
            self._show_preview = False

    def annotate_frame(self, frame: Optional[np.ndarray]):
        if frame is None or self._last_landmarks is None:
            return
        MP_DRAW.draw_landmarks(
            frame,
            self._last_landmarks,
            MP_CONNECTIONS,
            landmark_drawing_spec=MP_DRAW.DrawingSpec(color=(0, 255, 0), thickness=2),
            connection_drawing_spec=MP_DRAW.DrawingSpec(color=(255, 255, 255), thickness=2),
        )

    def _normalize_elbow(self, raw_value: float) -> float:
        if self._elbow_calibration_samples < 200:
            if self._elbow_raw_min is None or raw_value < self._elbow_raw_min:
                self._elbow_raw_min = raw_value
            if self._elbow_raw_max is None or raw_value > self._elbow_raw_max:
                self._elbow_raw_max = raw_value
            self._elbow_calibration_samples += 1

        if (
            self._elbow_raw_min is not None
            and self._elbow_raw_max is not None
            and (self._elbow_raw_max - self._elbow_raw_min) > 1e-3
        ):
            span = self._elbow_raw_max - self._elbow_raw_min
            normalized = (raw_value - self._elbow_raw_min) / span  # 0..1
            return float(np.clip(normalized * 2.0 - 1.0, -1.0, 1.0))

        return float(np.clip(raw_value, -1.0, 1.0))

