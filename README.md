# YOLO-3D

3D Pedestrian Pose Extraction Using Stereo Vision and YOLO Pose
Estimation

## Overview

This project loads ZED `.svo` / `.svo2` video files, performs YOLO 2D
keypoint detection, projects those keypoints into 3D using ZED depth
data, and visualizes the resulting 3D skeletons using Open3D. It also
retrieves ZED's native 3D body keypoints for comparison. Chamfer and
Hausdorff distances are computed to evaluate similarity between
YOLO-based and ZED-based 3D poses.

## How to Run

1.  Download SVO videos from:\
    **https://gogl.to/3q2w**

2.  Set the SVO file path inside the script under USER SETTINGS section of the code:

    ``` python
    svo_path = "path/to/your/file.svo2"
    ```

3.  Set the starting frame index:

    ``` python
    svo_pos = 150
    ```

4.  Run the script:

        python pose_estimate_3d.py
    An opencv window will open showing the first frame of the svo video you set in svo_path variable.

6.  Keyboard controls:

    -   **D** -- Move forward 0.5s\
    -   **A** -- Move backward 0.5s\
    -   **S** -- Show 3D skeleton in Open3D\
    -   **Q** -- Quit

7.  Close the Open3D window before navigating to another frame.
8.  After closing the window, you can see chamfer and hausdorff distances for each person in openCV window

Additionally you can change skip_frames variable to change how many frames will be skipped when navigating the svo video.
You can also use ZED's own Zed Explorer App (available with sdk) to view svo videos or scrub through entire video and get info about which frame to naviagte to.
