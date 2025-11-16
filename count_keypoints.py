# --- LIBRARIES ---
import pyzed.sl as sl
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# --- Path to SVO ---
svo_path = "videos/HD1080_SN38536458_15-01-13.svo2"

# --- Initialize ZED ---
init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.NEURAL
init_params.coordinate_units = sl.UNIT.METER
init_params.set_from_svo_file(svo_path)
init_params.svo_real_time_mode = False

zed = sl.Camera()
if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    raise SystemExit("Failed to open SVO file.")

runtime_params = sl.RuntimeParameters()
runtime_params.measure3D_reference_frame = sl.REFERENCE_FRAME.CAMERA
image = sl.Mat()

# --- ZED BODY DETECTION ---
bodies = sl.Bodies()
body_params = sl.BodyTrackingParameters()
body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
body_params.enable_tracking = False
body_params.enable_segmentation = False
body_params.enable_body_fitting = True
body_params.body_format = sl.BODY_FORMAT.BODY_18

if body_params.enable_tracking:
    positional_tracking_param = sl.PositionalTrackingParameters()
    positional_tracking_param.set_floor_as_origin = True
    zed.enable_positional_tracking(positional_tracking_param)

err = zed.enable_body_tracking(body_params)
if err != sl.ERROR_CODE.SUCCESS:
    print("Enable Body Tracking failed:", err)
    zed.close()
    exit()

body_runtime_param = sl.BodyTrackingRuntimeParameters()
body_runtime_param.detection_confidence_threshold = 40

# --- Load YOLO model ---
model = YOLO("yolo11n-pose.pt").to("cuda")

# --- Storage ---
data = []

# --- Process all frames ---
total_frames = zed.get_svo_number_of_frames()
print(f"Total frames: {total_frames}")

for frame_idx in range(total_frames):
    zed.set_svo_position(frame_idx)

    if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
        continue

    # Get image
    zed.retrieve_image(image, sl.VIEW.LEFT)
    frame = image.get_data()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # --- YOLO 2D keypoints ---
    results = model(frame, conf=0.6, verbose=False)
    yolo_valid_count = 0
    if results[0].keypoints is not None and hasattr(results[0].keypoints, "xy"):
        for person in results[0].keypoints.xy:
            kpts_2d = person.cpu().numpy()
            yolo_valid_count += np.sum(np.any(kpts_2d != 0.0, axis=1))

    # --- ZED 3D keypoints ---
    zed_valid_count = 0
    if zed.retrieve_bodies(bodies, body_runtime_param) == sl.ERROR_CODE.SUCCESS and bodies.is_new:
        neck_idx = 1
        for body in bodies.body_list:
            kp3d = np.array(body.keypoint)
            if kp3d.size == 0:
                continue
            kp_no_neck = np.delete(kp3d, neck_idx, axis=0)
            valid_mask = np.logical_and(
                ~np.isnan(kp_no_neck).any(axis=1),
                np.any(kp_no_neck != 0.0, axis=1)
            )
            zed_valid_count += np.sum(valid_mask)

    print(f"Frame {frame_idx}: YOLO={yolo_valid_count}, ZED={zed_valid_count}")

    # --- Append results ---
    data.append({
        "frame": frame_idx,
        "yolo_keypoints": yolo_valid_count,
        "zed_keypoints": zed_valid_count
    })

# --- Save results ---
df = pd.DataFrame(data)
df.to_csv("keypoint_counts_indoor.csv", index=False)
print("Saved results to keypoint_counts_indoor.csv")

# --- Cleanup ---
zed.close()
cv2.destroyAllWindows()
