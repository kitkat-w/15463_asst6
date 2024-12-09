In src/,
structure_light.py implements part 1, extrinsics and intrinsics are stored in data/frog
structure_light_scene1.py implements part 2 object 1
structure_light_scene2.py implements part 2 object 2

In data/frog,
instrinsic_calibration.npz stores resulting intrinsic calibration parameters
extrinsic_calibration.npz stores rotation and translation matrices for each ground plane
all_pts.npz are reconstructed 3D points for all frames during calibration of shadow lines
all_planes.npz are the estimated shadow plane parameters (points P1 and normals n) for all frames