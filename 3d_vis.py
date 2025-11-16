import numpy as np
import open3d as o3d

# Input 3D keypoints (COCO18 order)
keypoints = np.array([
    [0.16497, -0.8705, 5.7282],
    [0.1762, -0.67276, 5.7309],
    [0.034628, -0.69566, 5.7504],
    [-0.13766, -0.6053, 5.6533],
    [-0.35002, -0.62404, 5.8801],
    [0.32156, -0.66282, 5.8069],
    [0.41443, -0.41116, 5.8986],
    [0.4758, -0.18835, 5.9158],
    [0.087441, -0.17288, 5.7334],
    [0.10203, 0.2121, 5.8597],
    [0.12469, 0.55426, 5.8785],
    [0.27956, -0.1633, 5.7494],
    [0.3445, 0.19083, 5.8222],
    [0.39133, 0.55259, 5.8609],
    [0.1225, -0.9159, 5.7761],
    [0.20749, -0.90401, 5.7016],
    [0.09072, -0.892, 5.8617],
    [0.26231, -0.92063, 5.9451]
])

# COCO-18 skeleton connections (as pairs of indices)
skeleton_pairs = [
    (0, 1), (1, 2), (1, 5),
    (0, 14), (0, 15),
    (14, 16), (15, 17),
    (5, 6), (6, 7),
    (2, 3),(3,4),
    (2,8), (8, 11),(5,11),
    (11, 12),(12, 13),
    (8, 9), (9, 10)
]

# Create point cloud for joints
points = o3d.geometry.PointCloud()
points.points = o3d.utility.Vector3dVector(keypoints)
points.paint_uniform_color([1, 0.706, 0])  # yellow

# Create skeleton lines
lines = o3d.geometry.LineSet()
lines.points = o3d.utility.Vector3dVector(keypoints)
lines.lines = o3d.utility.Vector2iVector(skeleton_pairs)
lines.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in skeleton_pairs])  # blue

# Visualize
o3d.visualization.draw_geometries([points, lines], window_name="3D Skeleton (COCO18)",
                                  width=800, height=600)
