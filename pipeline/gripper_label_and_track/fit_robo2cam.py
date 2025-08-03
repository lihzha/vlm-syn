"""
Estimate 6‑DoF end‑effector poses in the camera frame when only
  • a static RGB sequence
  • 6‑DoF EE poses in the robot‑base frame
are available and camera intrinsics / extrinsics are unknown.
"""

import json
import os
from glob import glob

import cv2 as cv
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from pipeline.gripper_label_and_track.utils.get_gripper_pos import get_gripper_pos_raw
from pipeline.gripper_label_and_track.utils.point_trackers.tapnet.tapnet.offline_inference import (
    get_keypoint_tracking,
)
from pipeline.gripper_label_and_track.utils.visualize_movement import visualize_movement

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ───────────────────────────── helper geometry ────────────────────────────── #


def pose_vec_to_mat(p):
    """[x,y,z,qw,qx,qy,qz] -> 4×4 SE(3)"""
    t, q = p[:3], p[3:]
    T = np.eye(4)
    T[:3, :3] = R.from_quat(q).as_matrix()
    T[:3, 3] = t
    return T


def mat_to_pose_vec(T):
    """4×4 -> [x,y,z,qw,qx,qy,qz]"""
    rot = R.from_matrix(T[:3, :3]).as_quat()
    return np.concatenate([T[:3, 3], rot])


def project(K, R_cb, t_cb, pts_b):
    """project Nx3 pts (robot base) into pixels"""
    pts_c = (R_cb @ pts_b.T + t_cb[:, None]).T  # (N,3)
    uv_h = (K @ pts_c.T).T  # (N,3)
    uv = uv_h[:, :2] / uv_h[:, 2:3]
    return uv, pts_c[:, 2]  # return z for filtering


def residuals(x, Pw, uv):
    # unpack parameters ------------------------------------------------------
    rvec, tvec = x[:3], x[3:6]
    fx, fy, cx, cy = x[6:]
    R_cb = R.from_rotvec(rvec).as_matrix()
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # project ----------------------------------------------------------------
    uv_hat, z = project(K, R_cb, tvec, Pw)  # (N,2), (N,)

    # image reprojection error (always N*2 long) -----------------------------
    repro = (uv_hat - uv).ravel()

    # depth‑positivity penalty (same N long) ---------------------------------
    # if z >= ε → penalty = 0
    # if z  < ε → linear penalty that grows as point goes behind the camera
    eps = 1e-3
    depth_penalty = np.maximum(0.0, eps - z)  # positive when z < eps

    # scale the depth term so units are comparable to pixels
    depth_penalty *= 10.0  # tweak if needed

    return np.hstack([repro, depth_penalty])


use_query = False
save_query = True

for traj_idx in range(0, 1):
    # ─────────────────── load data ──────────────────── #

    img_paths = sorted(glob(f"data/traj_{traj_idx}/*.png"))
    poses_base = np.load(f"data/traj_{traj_idx}/poses_base.npy")  # (N,7)
    assert len(img_paths) == len(poses_base)

    downsample_factor = len(img_paths) // len(
        img_paths
    )  # downsample to original frames

    img_paths = img_paths[::downsample_factor]
    poses_base = poses_base[::downsample_factor, ..., :7]  # downsample for speed

    # create a new image folder and a new poses_base.npy
    os.makedirs(f"data/traj_{traj_idx}_downsampled", exist_ok=True)
    for i, path in enumerate(img_paths):
        new_path = os.path.join(f"data/traj_{traj_idx}_downsampled", f"{i:04d}.png")
        cv.imwrite(new_path, cv.imread(path))
        img_wh = cv.imread(path).shape[:2][::-1]

    np.save(f"data/traj_{traj_idx}_downsampled/poses_base_downsampled.npy", poses_base)

    pts3d = []  # robot‑base 3‑D points       (N*,3)
    pts2d = []  # corresponding pixel points  (N*,2)

    # ─────────────────── get keypoint for the 1st frame, and use tracker to get keypoint for the rest of frames ──────────────────── #

    gripper_uv, _, prediction = get_gripper_pos_raw(
        cv.imread(img_paths[0]),
        plot=True,
        del_cache=False,
        image_dims=img_wh,
        use_query=use_query,
        save_query=save_query,
    )

    if prediction is not None:
        tracks = get_keypoint_tracking(
            f"data/traj_{traj_idx}_downsampled",
            np.array([[0, gripper_uv[1], gripper_uv[0]]]),
            visualize=True,
            image_dims=img_wh,
        )[0]  # (B, N, 2)

        for i, path in enumerate(img_paths):
            img = cv.imread(path)
            if img_wh is None:
                h, w = img.shape[:2]
                img_wh = (w, h)
            uv = tracks[i]  # (y,x) in pixels
            pts2d.append(uv)
            pts3d.append(poses_base[i, :3])  # use only EE position

        pts3d = np.asarray(pts3d, float)
        pts2d = np.asarray(pts2d, float)
        assert len(pts3d) >= 20, "too few detections – improve detector"

        # ──────────────── initial guesses (cheap but usually works) ──────────────── #

        fx0 = fy0 = 1.2 * max(img_wh)  # rough focal
        cx0, cy0 = img_wh[0] / 2, img_wh[1] / 2
        rvec0 = np.zeros(3)  # camera roughly faces +z of robot base
        tvec0 = np.array([0, 0, 1])  # 1 m in front

        # fx0 = 378.0632
        # fy0 = 376.4866
        # cx0 = cy0 = 259
        # rvec0 = np.zeros(3)  # camera roughly faces +z of robot base
        # tvec0 = np.array([0.0, 0.0, 0.0])

        x0 = np.hstack([rvec0, tvec0, [fx0, fy0, cx0, cy0]])

        # ────────────────────────── solve with Levenberg‑Marquardt ────────────────── #

        opt = least_squares(
            residuals,
            x0,
            args=(pts3d.copy(), pts2d.copy()),
            method="lm",
            max_nfev=10000,
            verbose=2,
        )

        # opt = least_squares(
        #         residuals, x0, args=(pts3d, pts2d),
        #         loss="soft_l1", f_scale=1.0,  # Tukey, Cauchy, … also work
        #         verbose=2, max_nfev=5000)

        rvec, tvec = opt.x[:3], opt.x[3:6]
        fx, fy, cx, cy = opt.x[6:]
        K_opt = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        R_cb = R.from_rotvec(rvec).as_matrix()
        t_cb = tvec

        # ─────────── convert every EE pose from base → camera coordinates ─────────── #

        T_cam_ee = []
        for p in poses_base:
            T_b_ee = pose_vec_to_mat(p)
            T_c_b = np.eye(4)
            T_c_b[:3, :3], T_c_b[:3, 3] = R_cb, t_cb
            T_c_ee = T_c_b @ T_b_ee
            T_cam_ee.append(T_c_ee)
        T_cam_ee = np.stack(T_cam_ee)  # (N,4,4)

        # relative motions ΔT_cam = T_{k→k+1} in camera frame
        dT_cam = []
        for k in range(len(T_cam_ee) - 1):
            dT = np.linalg.inv(T_cam_ee[k]) @ T_cam_ee[k + 1]
            dT_cam.append(dT)
        dT_cam = np.stack(dT_cam)

        # ─────────────────────────── save / inspect results ───────────────────────── #

        np.savez_compressed(
            f"data/traj_{traj_idx}_downsampled/ee_camera_frame.npz",
            T_cam_ee=T_cam_ee,
            dT_cam=dT_cam,
            K=K_opt,
            R_cb=R_cb,
            t_cb=t_cb,
        )
        print(f"{traj_idx} Calibration finished")
        # write to log
        with open(f"data/traj_{traj_idx}_downsampled/calibration_log.json", "w") as f:
            f.write(
                json.dumps(
                    {
                        "fx": fx,
                        "fy": fy,
                        "cx": cx,
                        "cy": cy,
                        "R_cam_from_base": R_cb.tolist(),
                        "t_cam_from_base": t_cb.tolist(),
                        "tracked_frames": int(len(pts3d)),
                        "downsample_factor": downsample_factor,
                        "message": opt.message,
                        "result": "success",
                    },
                    indent=2,
                )
            )
        visualize_movement(
            frames_path=f"data/traj_{traj_idx}_downsampled",
            T_cam_ee_path=f"data/traj_{traj_idx}_downsampled/ee_camera_frame.npz",
            out_path=f"data/traj_{traj_idx}_downsampled/motion_vis.mp4",
            fps=3,
        )

    else:
        print(f"Skipping trajectory {traj_idx} due to no gripper detection")
        with open(f"data/traj_{traj_idx}_downsampled/calibration_log.json", "w") as f:
            f.write(
                json.dumps(
                    {
                        "message": "No gripper detection",
                        "downsample_factor": downsample_factor,
                        "result": "failure",
                    },
                    indent=2,
                )
            )
