#!/usr/bin/env python3
"""
dance.py

Usage:
    python dance.py --input dance.mp4 --output dance_skel.mp4 --trails --json dance_keypoints.json

Requirements:
    pip install opencv-python mediapipe numpy
"""

import argparse
import cv2
import mediapipe as mp
import numpy as np
import json
from collections import deque

# ---------- Helpers ----------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Colors (BGR)
COLOR_LANDMARK = (0, 255, 255)  # yellow-ish
COLOR_CONNECTION = (0, 200, 0)  # green-ish
COLOR_TRAIL = (0, 120, 255)     # orange-ish
CIRCLE_RADIUS = 4
LINE_THICKNESS = 2

# Keypoints to keep trails for (use MediaPipe landmark indices)
# We'll track wrists, ankles, shoulders, hips, and nose
TRAIL_LANDMARKS = [
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE,
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
]

def normalized_to_pixel(n_x, n_y, width, height):
    """Convert normalized landmark (x,y in [0,1]) to pixel coordinates, clipping to frame."""
    x_px = int(np.clip(n_x * width, 0, width - 1))
    y_px = int(np.clip(n_y * height, 0, height - 1))
    return x_px, y_px

# ---------- Main processing function ----------
def process_video(
    input_path,
    output_path,
    save_json_path=None,
    trails=True,
    trail_len=20,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # works widely; try 'XVID'/'avc1' if needed
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize trail queues
    trails_deques = {int(k): deque(maxlen=trail_len) for k in TRAIL_LANDMARKS}

    # JSON write structure
    frames_keypoints = [] if save_json_path else None

    # Initialize MediaPipe Pose
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # 0/1/2 tradeoff speed/accuracy
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as pose:

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            orig = frame.copy()
            # Convert BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = pose.process(frame_rgb)
            frame_rgb.flags.writeable = True

            # Prepare overlay canvas (we will draw on orig)
            if results.pose_landmarks:
                # Draw connections and landmarks using MediaPipe's drawing util for clean skeleton,
                # then we will optionally draw trails.
                mp_drawing.draw_landmarks(
                    orig,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=COLOR_LANDMARK, thickness=LINE_THICKNESS, circle_radius=CIRCLE_RADIUS),
                    mp_drawing.DrawingSpec(color=COLOR_CONNECTION, thickness=LINE_THICKNESS, circle_radius=CIRCLE_RADIUS),
                )

                # Convert normalized landmarks to pixel coords array for further custom drawing / JSON
                keypoints_px = {}
                for idx, lm in enumerate(results.pose_landmarks.landmark):
                    x_px, y_px = normalized_to_pixel(lm.x, lm.y, width, height)
                    keypoints_px[idx] = {"x": x_px, "y": y_px, "z": lm.z, "visibility": lm.visibility}

                # Trails: append selected keypoint positions
                if trails:
                    for lm_enum in TRAIL_LANDMARKS:
                        idx = int(lm_enum)
                        if idx in keypoints_px:
                            trails_deques[idx].append((keypoints_px[idx]["x"], keypoints_px[idx]["y"]))

                    # Draw trails: polyline for each tracked landmark
                    for idx, dq in trails_deques.items():
                        if len(dq) >= 2:
                            pts = np.array(dq, dtype=np.int32)
                            # Draw fading trail: draw segments with decreasing thickness/alpha
                            # We don't have alpha on simple OpenCV drawing; emulate by decreasing thickness.
                            n = len(pts)
                            for i in range(n - 1):
                                pt1 = tuple(pts[i])
                                pt2 = tuple(pts[i + 1])
                                thickness = max(1, int( (i + 1) / n * 6 ))
                                cv2.line(orig, pt1, pt2, COLOR_TRAIL, thickness=thickness, lineType=cv2.LINE_AA)

                # Save keypoints for this frame
                if frames_keypoints is not None:
                    # store a compact mapping to landmark names + coords
                    frame_kps = {"frame_idx": frame_idx, "landmarks": {}}
                    for idx, kp in keypoints_px.items():
                        name = mp_pose.PoseLandmark(idx).name if idx < len(mp_pose.PoseLandmark) else str(idx)
                        frame_kps["landmarks"][name] = {"x": kp["x"], "y": kp["y"], "z": kp["z"], "visibility": kp["visibility"]}
                    frames_keypoints.append(frame_kps)

            # Optionally show FPS / frame index
            cv2.putText(orig, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            # Write to output video
            out.write(orig)

            frame_idx += 1

            # If you want to preview live while processing uncomment below:
            # cv2.imshow("Skeleton Overlay", orig)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    # Cleanup
    cap.release()
    out.release()
    # cv2.destroyAllWindows()

    # Save JSON
    if frames_keypoints is not None and save_json_path:
        with open(save_json_path, "w") as f:
            json.dump(frames_keypoints, f, indent=2)

    print(f"Processed {frame_idx} frames. Output saved to: {output_path}")
    if save_json_path:
        print(f"Keypoints JSON saved to: {save_json_path}")

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create skeleton overlay video from dance input using MediaPipe Pose.")
    parser.add_argument("--input", "-i", required=True, help="Path to input video (short dance clip).")
    parser.add_argument("--output", "-o", required=True, help="Path to output video (e.g., out.mp4).")
    parser.add_argument("--json", "-j", default=None, help="Optional: save per-frame keypoints JSON.")
    parser.add_argument("--trails", action="store_true", help="Enable drawing motion trails for keypoints.")
    parser.add_argument("--trail_len", type=int, default=20, help="Number of frames to keep in motion trail.")
    parser.add_argument("--det_conf", type=float, default=0.5, help="Min detection confidence.")
    parser.add_argument("--track_conf", type=float, default=0.5, help="Min tracking confidence.")
    args = parser.parse_args()

    process_video(
        args.input,
        args.output,
        save_json_path=args.json,
        trails=args.trails,
        trail_len=args.trail_len,
        min_detection_confidence=args.det_conf,
        min_tracking_confidence=args.track_conf,
    )

