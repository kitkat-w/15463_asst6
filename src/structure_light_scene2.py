import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from cp_hw6 import pixel2ray, set_axes_equal
import os

def cam2plane(P_cam, R, T):
    # P_cam: Nx3 points in camera coords
    # R, T: rotation and translation from camera to plane
    # P_plane = R^T (P_cam - T)
    return (R.T @ (P_cam - T.flatten()[:,None])).T

def plane2cam(P_plane, R, T):
    # P_plane: Nx3 points in plane coords
    # P_cam = R P_plane + T
    return (R @ P_plane.T + T).T

def estimate_shadow_edges(deltaI, y_min, y_max, x_min, x_max):
    T, H, W = deltaI.shape
    lines = []

    for t in range(1, T):  # Start from t=1 to ensure valid indexing
        zero_crossings = []

        # Loop through the specified region and detect zero-crossings
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):  # Ensure x-1 is valid
                if x < x_min or x >= x_max:
                    continue
                if deltaI[t, y, x-2] < 0 and deltaI[t, y, x-1] < 0 and deltaI[t, y, x] >= 0 and deltaI[t, y, x+1] >= 0 and deltaI[t, y,x] != deltaI[t-1, y, x]:
                    zero_crossings.append([x-1,y, 1])  # Homogeneous coordinates

        # Fit a line using SVD if enough points are detected
        if len(zero_crossings) > 2:  # Minimum points to fit a line
            A = np.array(zero_crossings)  
            _, _, V = np.linalg.svd(A)
            line = V[-1]  # Line parameters [a, b, c]
            lines.append(line)
        else:
            lines.append(None)  # Not enough points to fit a line
            continue

        # plt.figure()
        # plt.imshow(deltaI[t], cmap='gray')
        # plt.scatter(A[:,0],A[:,1], c='b', s = 5)
        # plt.show()

        # if 1:
        #     a, b, c = V[-1]
        #     x_vals = np.linspace(y_min, y_max)
        #     x_vals = (-a * y_vals - c) / b
        #     valid_idx = (y_vals >= y_min) & (y_vals < y_max)
        #     x_vals = x_vals[valid_idx]
        #     y_vals = y_vals[valid_idx]

        #     plt.figure(figsize=(10, 8))
        #     plt.imshow(deltaI[t], cmap='gray')
        #     plt.scatter(x_vals, y_vals, color='red', label='Shadow Line')
        #     plt.title(f"Shadow Line for Frame {t}")
        #     plt.xlabel("X")
        #     plt.ylabel("Y")
        #     # plt.gca().invert_yaxis()  # Match image coordinates
        #     plt.legend()
        #     plt.show()
    # lines = np.array(lines)
    return lines

def estimate_t_shadow(deltaI):
    I_out = np.zeros(deltaI.shape[1:])
    for i in range(deltaI.shape[1]):
        for j in range(deltaI.shape[2]):

            signs = np.sign(deltaI[:, i, j])
            shifted_signs = np.roll(signs, shift=1, axis=0)
            zero_crossings = np.where((signs * shifted_signs) < 0)
            idx = zero_crossings[0]
            if idx.any():
                #print(idx, zero_crossings)
                I_out[i,j] = idx[0]
               
    return I_out

def intersect_ray_with_plane(ray_origin, ray_dir, plane_normal, plane_point):
    denom = np.dot(ray_dir, plane_normal)
    if np.abs(denom) < 1e-12:
        # Ray is parallel to the plane
        return None
    t = np.dot((plane_point - ray_origin), plane_normal) / denom
    if t < 0:
        # Intersection is behind the ray origin
        return None
    return ray_origin + t * ray_dir

def compute_shadow_line_3D(line_params, y1, y2, mtx, dist, R, T):
    a, b, c = line_params
    x1, x2 = ((-b * y - c) / (a + 1e-12) for y in (y1, y2))

    # Backproject to rays in camera coordinates
    p_img = np.float32([[x1, y1], [x2, y2]]).reshape(-1, 1, 2)
    rays = pixel2ray(p_img, mtx, dist).squeeze().T  # Shape: 3 x 2

    # Transform rays to plane coordinates
    r_plane = R.T @ rays

    # Reference line point in plane coordinates (line passing through camera origin)
    line_pt_plane = (R.T @ (-T)).ravel()

    # Plane parameters in plane coordinates
    plane_pt = np.array([0, 0, 0])
    plane_normal = np.array([0, 0, 1])

    # Intersect the two rays with the plane using the helper function
    P1_plane = intersect_ray_with_plane(line_pt_plane, r_plane[:, 0], plane_normal, plane_pt)
    P2_plane = intersect_ray_with_plane(line_pt_plane, r_plane[:, 1], plane_normal, plane_pt)

    if P1_plane is None or P2_plane is None:
        return None

    # Transform the points back to camera coordinates
    P_cam = plane2cam(np.array([P1_plane, P2_plane]), R, T)
    return P_cam


def main():
    data_path = '../data/scene7/'
    file_names = sorted([fn for fn in os.listdir(data_path) if fn.endswith('.png') or fn.endswith('.jpg')])
    # images = []
    # for fname in file_names:
    #     im = io.imread(os.path.join(data_path, fname))
    #     gray = color.rgb2gray(im)
    #     images.append(gray)
    # images = np.array(images)  # T x H x W
    # # images = images[::,::2,::2]

    # np.save('images_scene7.npy', images)
    images = np.load('images_scene7.npy')

    color_img = cv.imread(os.path.join(data_path, file_names[0]))
    color_img = cv.cvtColor(color_img, cv.COLOR_BGR2RGB)


    T, H, W = images.shape
    Imax = np.max(images, axis=0)
    Imin = np.min(images, axis=0)
    Ishadow = (Imax + Imin)/2
    Ishadow = Ishadow.reshape(1, H, W)
    deltaI = images - Ishadow

    plt.imshow(deltaI[10], cmap='gray')
    plt.show()

    # You must define unobstructed regions for horizontal and vertical planes.
    # For example (these values are just placeholders and must be adjusted):
    horiz_y_min, horiz_y_max, horiz_x_min, horiz_x_max = 400, 900, 1680, 2560    # region for horizontal plane shadow line detection
    vert_y_min, vert_y_max, vert_x_min, vert_x_max = 1360, 2100, 2100, 2600    # region for vertical plane shadow line detection


    vertical_lines = estimate_shadow_edges(deltaI, horiz_y_min, horiz_y_max, horiz_x_min, horiz_x_max)
    horizontal_lines = estimate_shadow_edges(deltaI, vert_y_min, vert_y_max, vert_x_min, vert_x_max)
    # exit()


    t_shadow = estimate_t_shadow(deltaI)

    plt.imshow(t_shadow, cmap='jet')
    plt.show()

    # print(horizontal_lines.shape, vertical_lines.shape)

    # Load calibration results
    calib_intrinsic = np.load('../data/my_calib/intrinsic_calib.npz')
    mtx = calib_intrinsic['mtx']
    dist = calib_intrinsic['dist']
    calib_extrinsic = np.load('../data/scene7/extrinsic_calib.npz')
    rmat_h = calib_extrinsic['rmat_h']
    tvec_h = calib_extrinsic['tvec_h']
    rmat_v = calib_extrinsic['rmat_v']
    tvec_v = calib_extrinsic['tvec_v']


    all_planes = []
    all_p1, all_p2, all_p3, all_p4 = [], [], [], []
    for t in range(T-1):
        line_h = horizontal_lines[t]
        line_v = vertical_lines[t]
        if line_h is None or line_v is None:
            # no line detected this frame
            print(t)
            all_planes.append(None)
            continue

        # P1,P2 on horizontal plane line
        P12 = compute_shadow_line_3D(line_h, 1600, 1800, mtx, dist, rmat_h, tvec_h)
        # P3,P4 on vertical plane line
        P34 = compute_shadow_line_3D(line_v, 400, 800, mtx, dist, rmat_v, tvec_v)

        if P12.shape[0]<2 or P34.shape[0]<2:
            all_planes.append(None)
            continue
        # print (P12)
        P1, P2 = P12[0], P12[1]
        P3, P4 = P34[0], P34[1]

        # print (P1.shape, P2.shape, P3.shape, P4.shape)
        # Compute shadow plane normal
        # normal = normalize((P2-P1) x (P4-P3))
        v1 = P2 - P1
        v2 = P4 - P3
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-12:
            all_planes.append(None)
            continue
        # all_p1.append(cam2plane([P1], rmat_h, tvec_h))
        # all_p2.append(cam2plane([P2], rmat_h, tvec_h))
        # all_p3.append(cam2plane([P3], rmat_v, tvec_v))
        # all_p4.append(cam2plane([P4], rmat_v, tvec_v))
        all_p1.append(P1)
        all_p2.append(P2)
        all_p3.append(P3)
        all_p4.append(P4)
        normal = normal / norm_len

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(P1[0], P1[1], P1[2], c='r', label='P1')
        # ax.scatter(P2[0], P2[1], P2[2], c='g', label='P2')
        # ax.scatter(P3[0], P3[1], P3[2], c='b', label='P3')
        # ax.scatter(P4[0], P4[1], P4[2], c='y', label='P4')
        # plt.show()

        # Save plane parameters: a point (P1) and normal
        all_planes.append((P1, normal))
    
    all_p1 = np.array(all_p1)
    all_p2 = np.array(all_p2)
    all_p3 = np.array(all_p3)
    all_p4 = np.array(all_p4)
    print (all_p1.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(all_p1[:,0], all_p1[:,1], all_p1[:,2], c='r', label='P1')
    ax.scatter(all_p2[:,0], all_p2[:,1], all_p2[:,2], c='g', label='P2')
    ax.scatter(all_p3[:,0], all_p3[:,1], all_p3[:,2], c='b', label='P3')
    ax.scatter(all_p4[:,0], all_p4[:,1], all_p4[:,2], c='y', label='P4')
    plt.show()

    # Now reconstruct a cropped region around the object
    # Example crop:
    x1,x2,y1,y2 = 1680, 2505, 1080, 1360
    cropped = images[:,y1:y2,x1:x2]
    tshadow_crop = t_shadow[y1:y2,x1:x2]

    # Reconstruct each pixel
    # For each pixel (x,y), find tshadow. Use that frame's plane. Intersect pixel ray with plane.
    thresh = 0.1
    Zpts = []
    Cpts = []
    for yy in range(y1,y2):
        for xx in range(x1,x2):
            diff = Imax[yy,xx] - Imin[yy,xx]

            tt = t_shadow[yy,xx]
            if abs(diff) < thresh:
                continue

            if int(np.floor(tt)) >= len(all_planes):
                continue
            if all_planes[int(np.floor(tt))] is None:
                # no valid shadow crossing
                continue
            P1, normal = all_planes[int(np.floor(tt))]

            # backproject pixel (xx,yy)
            p_img = np.array([[xx,yy]], dtype=np.float32).reshape(-1,1,2)
            ray_dir = pixel2ray(p_img, mtx, dist)[0,0,:]
            ray_origin = np.array([0.0,0.0,0.0])

            # Plane eq: (P - P1)·normal=0
            P = intersect_ray_with_plane(ray_origin, ray_dir, normal, P1)
            if P is None:
                continue
                print (tt)
            if P[2] > 1800 or P[2] < 1600:
                continue
            Zpts.append(P)
            # Color the point using one of the non-shadowed frames:
            # pick a frame where pixel intensity is max or min. Just pick t=0 for simplicity
            intensity = color_img[yy,xx]/255.0
            Cpts.append(intensity)

    Zpts = np.array(Zpts)
    Cpts = np.array(Cpts)

    # Cpts = (Cpts - Cpts.min()) / (Cpts.max() - Cpts.min()) 
    # Cpts = (Cpts) / 255

    # Visualize point cloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(Zpts[:,0], Zpts[:,1], Zpts[:,2], c=Cpts, s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)
    plt.show()

if __name__ == "__main__":
    main()
