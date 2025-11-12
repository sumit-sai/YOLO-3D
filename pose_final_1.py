#---LIBRARIES---
import pyzed.sl as sl
import cv2
from skimage.metrics import hausdorff_distance
from ultralytics import YOLO
import open3d as o3d
import numpy as np
import pyvista as pv
import time
import copy
from scipy.spatial import cKDTree

COCO18_BONES = [
    (0, 1), (1, 2), (1, 5),
    (0, 14), (0, 15),
    (14, 16), (15, 17),
    (5, 6), (6, 7),
    (2, 3),(3,4),
    (2,8), (8, 11),(5,11),
    (11, 12),(12, 13),
    (8, 9), (9, 10)
]
COCO17_BONES = [
    (0,1), (0,2), (1,3), (2,4),
    (5,6),
    (5,7), (7,9), (6,8), (8,10),
    (5,11), (6,12),
    (11,12), (11,13), (13,15), (12,14), (14,16)
]

def chamfer(p1, p2):
    # remove invalid points
    mask1 = ~np.isnan(p1).any(axis=1) & np.any(p1 != 0, axis=1)
    mask2 = ~np.isnan(p2).any(axis=1) & np.any(p2 != 0, axis=1)
    p1_valid, p2_valid = p1[mask1], p2[mask2]

    t1, t2 = cKDTree(p1_valid), cKDTree(p2_valid)
    d1, _ = t2.query(p1_valid)
    d2, _ = t1.query(p2_valid)
    return np.mean(d1) + np.mean(d2),np.mean(d1),np.mean(d2)

def hausdorff(p1, p2):
    # remove invalid points
    mask1 = ~np.isnan(p1).any(axis=1) & np.any(p1 != 0, axis=1)
    mask2 = ~np.isnan(p2).any(axis=1) & np.any(p2 != 0, axis=1)
    p1_valid, p2_valid = p1[mask1], p2[mask2]

    if len(p1_valid) == 0 or len(p2_valid) == 0:
        return np.inf

    t1, t2 = cKDTree(p1_valid), cKDTree(p2_valid)
    d1, _ = t1.query(p2_valid)
    d2, _ = t2.query(p1_valid)
    return max(np.max(d1), np.max(d2))

#---FUNCTION TO PROJECT 2D TO 3D---
def project_yolo_keypoints_to_3d(res, depth_map, fx, fy, cx, cy):
    people_3d = []
    if not hasattr(res.keypoints, "xy") or len(res.keypoints.xy) == 0:
        return people_3d  # no detections

    for keypoints_2d in res.keypoints.xy:
        if keypoints_2d.numel() == 0:
            continue  # skip empty tensor

        kpts_2d = keypoints_2d.cpu().numpy()
        if kpts_2d.shape[0] == 0:
            continue  # skip if shape is empty
        kpts_3d = np.zeros((kpts_2d.shape[0], 3), dtype=np.float32)

        for i, (u, v) in enumerate(kpts_2d):
            # skip invalid 2D coords
            if u == 0.0 and v == 0.0:
                kpts_3d[i] = [0.0, 0.0, 0.0]
                continue

            u_int, v_int = int(round(u)), int(round(v))

            success , z = depth_map.get_value(u_int,v_int)

            if z > 0 and np.isfinite(z):
                X = (u - cx) / fx * z
                Y = (v - cy) / fy * z
                Z = z
                kpts_3d[i] = [X, Y, Z]
        people_3d.append(kpts_3d)

    return people_3d

def draw_skeleton_open3d(keypoints_3d, coco):
    kpts = np.asarray(keypoints_3d, dtype=float)
    if kpts.size == 0 or kpts.shape[1] < 3:
        return None
    kpts=np.nan_to_num(kpts, nan=0.0, posinf=0.0, neginf=0.0)
    valid_mask = np.logical_and(~np.isnan(kpts).any(axis=1), np.any(kpts != 0, axis=1))
    if coco == 17:
        COCO_BONES = COCO17_BONES
        line_colour = [0.0, 0.0, 1.0]
        sphere_colour = [0.0, 0.0, 1.0]
    else:
        COCO_BONES = COCO18_BONES
        line_colour = [1.0, 0.0, 0.0]
        sphere_colour = [1.0, 0.0, 0.0]

    # build lines (only between valid keypoints)
    lines = []
    for i, j in COCO_BONES:
        if i < len(kpts) and j < len(kpts) and valid_mask[i] and valid_mask[j]:
            lines.append([int(i), int(j)])
    lines_arr = np.array(lines, dtype=np.int32) if len(lines) else np.empty((0, 2), dtype=np.int32)

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(kpts),
        lines=o3d.utility.Vector2iVector(lines_arr)
    )
    if len(lines):
        colors = np.tile(line_colour, (len(lines), 1))
        line_set.colors = o3d.utility.Vector3dVector(colors)

    spheres = []
    for p in keypoints_3d:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(p)
        sphere.paint_uniform_color(sphere_colour)
        spheres.append(sphere)

    return [line_set]+spheres


# --- Path to your SVO2 file ---
svo_path = "videos/HD1080_SN38536458_15-01-13.svo2"
# --- Initialize ZED ---
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.NEURAL # 
init_params.coordinate_units = sl.UNIT.METER #
init_params.set_from_svo_file(svo_path)
init_params.svo_real_time_mode = False

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    raise SystemExit("Failed to open SVO file.")
runtime_params = sl.RuntimeParameters()
runtime_params.measure3D_reference_frame = sl.REFERENCE_FRAME.CAMERA
image = sl.Mat()
depth_map = sl.Mat()

#---CALIBRATION PARAMETERS
calib = zed.get_camera_information().camera_configuration.calibration_parameters
fx, fy, cx, cy = calib.left_cam.fx, calib.left_cam.fy, calib.left_cam.cx, calib.left_cam.cy

#---ZED BODY DETECTION PARAMETERS---
bodies = sl.Bodies()
body_params = sl.BodyTrackingParameters()
body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
body_params.enable_tracking = True
body_params.enable_segmentation = False
body_params.enable_body_fitting = True
body_params.body_format=sl.BODY_FORMAT.BODY_18

if body_params.enable_tracking:
    positional_tracking_param = sl.PositionalTrackingParameters()
    positional_tracking_param.set_floor_as_origin = True
    zed.enable_positional_tracking(positional_tracking_param)

err = zed.enable_body_tracking(body_params)
if err != sl.ERROR_CODE.SUCCESS:
    print("Enable Body Tracking : "+repr(err)+". Exit program.")
    zed.close()
    exit()

body_runtime_param = sl.BodyTrackingRuntimeParameters()
body_runtime_param.detection_confidence_threshold = 40

# --- Fixed 30 FPS (0.5 s = 15 frames) ---
half_sec_frames = 5

print("Controls: 'D' → +0.5s | 'A' → -0.5s | 'Q' → quit")

cv2.namedWindow("ZED SVO Playback", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ZED SVO Playback", 1280, 720)
# --- Grab and show the first frame ---
if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
    zed.retrieve_image(image, sl.VIEW.LEFT)
    frame = image.get_data()
    cv2.imshow("ZED SVO Playback", frame)

# --- loading YOLO model ---
model = YOLO("yolo11n-pose.pt")

zed.set_svo_position(125)
# --- Main loop ---
while True:
    chamfer_dist = []
    hausdorff_dist=[]
    key = cv2.waitKey(0) & 0xFF  # wait for key
    if key == ord('q'):
        break

    current_frame = zed.get_svo_position()
    total_frames = zed.get_svo_number_of_frames()

    if key == ord('d'):  # forward
        new_pos = min(current_frame + half_sec_frames, total_frames - 1)
    elif key == ord('a'):  # backward
        new_pos = max(current_frame - half_sec_frames, 0)
    elif key == ord('s'):  # backward
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            err = zed.retrieve_bodies(bodies, body_runtime_param)
            if bodies.is_new:
                body_array = bodies.body_list
        else:
            print("didn't work")
        geometries = []  # store all skeletons

        gt_skeletons = []
        for p in bodies.body_list:
            kp3d = np.array(p.keypoint)
            gt_skeletons.append(kp3d)

        for obj in bodies.body_list:
            kp3d = np.array(obj.keypoint)

            skeleton = draw_skeleton_open3d(kp3d,18)
            if skeleton is not None:
                geometries.append(skeleton)

        if len(results[0].keypoints.xy) > 0:
            people_3d = project_yolo_keypoints_to_3d(result, depth_map, fx, fy, cx, cy)

            for obj in people_3d:

                skeleton = draw_skeleton_open3d(obj, 17)
                if skeleton is not None:
                    geometries.append(skeleton)


        neck_idx = 1  # neck joint
        neck_para= True
        for i in people_3d:
            det_skel = np.array(i)
            mapping_gt_to_yolo = [0, 14, 13, 16, 15, 4, 1, 5, 2, 6, 3, 10, 7, 11, 8, 12, 9]
            gt_skeletons_processed = []
            for gt in gt_skeletons:
                gt_1=gt
                gt_no_neck = np.delete(gt_1, neck_idx, axis=0)
                gt_reordered = np.array([gt_no_neck[idx] for idx in mapping_gt_to_yolo])
                if neck_para==True:
                    gt_skeletons_processed.append(np.array(gt))
                else:
                    gt_skeletons_processed.append(gt_reordered)

            cd = [chamfer(det_skel, gt) for gt in gt_skeletons_processed]
            distances,d1s,d2s=zip(*cd)
            best_idx = np.argmin(distances)
            best_match = gt_skeletons_processed[best_idx]
            best_distance = distances[best_idx]

            hausdorff_dist.append(hausdorff(det_skel, best_match))
            chamfer_dist.append({
                                    "total": distances[best_idx],
                                    "d1": d1s[best_idx],
                                    "d2": d2s[best_idx]
                                })

        o3d.visualization.draw_geometries([g for skeleton in geometries for g in skeleton])

    zed.set_svo_position(new_pos)

    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
        frame = image.get_data()
        frame=cv2.cvtColor(frame,cv2.COLOR_BGRA2BGR)
        text = f"Frame: {new_pos}/{total_frames - 1}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 5
        color_text = (255, 255, 255)
        color_bg = (0, 0, 0)

        # Get text size
        (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Text position
        pos = (30, 50)

        # Draw filled rectangle (background)
        cv2.rectangle(frame,
                      (pos[0] - 5, pos[1] - h - 5),
                      (pos[0] + w + 5, pos[1] + 5),
                      color_bg, -1)

        # Draw text on top
        cv2.putText(frame, text, pos, font, font_scale, color_text, thickness)
        results=model(frame)
        result=results[0]
        annotated_frame = result.plot() if result.keypoints is not None else frame.copy()

        if chamfer_dist:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box, chamfer_vals, haus_val in zip(boxes, chamfer_dist, hausdorff_dist):
                x1, y1, x2, y2 = map(int, box)

                # Extract values
                total = chamfer_vals["total"]
                d1 = chamfer_vals["d1"]
                d2 = chamfer_vals["d2"]

                # Text labels
                label1 = f"C: {total:.3f}"
                label2 = f"d1:{d1:.3f}"
                label3 = f"d2:{d2:.3f}"
                label4 = f"H: {haus_val:.3f}"

                # Font settings
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2

                # Get text sizes
                (w1, h1), _ = cv2.getTextSize(label1, font, font_scale, thickness)
                (w2, h2), _ = cv2.getTextSize(label2, font, font_scale, thickness)
                (w3, h3), _ = cv2.getTextSize(label3, font, font_scale, thickness)
                (w4, h4), _ = cv2.getTextSize(label4, font, font_scale, thickness)

                # Choose text positions
                text_pos1 = (x2, y2 - 170)
                text_pos2 = (x2, y2 - 130)
                text_pos3 = (x2, y2 - 90)
                text_pos4 = (x2, y2 - 50)
                # Background rectangles
                cv2.rectangle(annotated_frame,
                              (text_pos1[0] - 5, text_pos1[1] - h1 - 5),
                              (text_pos1[0] + w1 + 5, text_pos1[1] + 5),
                              (0, 0, 0), -1)
                cv2.rectangle(annotated_frame,
                              (text_pos2[0] - 5, text_pos2[1] - h2 - 5),
                              (text_pos2[0] + w2 + 5, text_pos2[1] + 5),
                              (0, 0, 0), -1)
                cv2.rectangle(annotated_frame,
                              (text_pos3[0] - 5, text_pos3[1] - h3 - 5),
                              (text_pos3[0] + w3 + 5, text_pos3[1] + 5),
                              (0, 0, 0), -1)
                cv2.rectangle(annotated_frame,
                              (text_pos4[0] - 5, text_pos4[1] - h4 - 5),
                              (text_pos4[0] + w4 + 5, text_pos4[1] + 5),
                              (0, 0, 0), -1)

                # Draw text
                cv2.putText(annotated_frame, label1, text_pos1, font, font_scale, (0, 255, 255), thickness)
                cv2.putText(annotated_frame, label2, text_pos2, font, font_scale, (0, 255, 255), thickness)
                cv2.putText(annotated_frame, label3, text_pos3, font, font_scale, (0, 255, 255), thickness)
                cv2.putText(annotated_frame, label4, text_pos4, font, font_scale, (0, 255, 255), thickness)

            chamfer_dist=[]
            hausdorff_dist=[]

        cv2.imshow("ZED SVO Playback", annotated_frame)

zed.close()
cv2.destroyAllWindows()
# --- Cleanup ---
cv2.destroyAllWindows()
zed.close()

