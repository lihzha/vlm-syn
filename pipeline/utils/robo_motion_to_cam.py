# from typing import List, Tuple

# import cv2 as cv
# import numpy as np
# from scipy.optimize import least_squares
# from scipy.spatial.transform import Rotation as R

# # ───────────────────────────── helper geometry ────────────────────────────── #


# def pose_vec_to_mat(p: np.ndarray) -> np.ndarray:
#     """[x,y,z,qw,qx,qy,qz] -> 4×4 SE(3)"""
#     t, q = p[:3], p[3:]
#     T = np.eye(4)
#     T[:3, :3] = R.from_quat(q).as_matrix()
#     T[:3, 3] = t
#     return T


# def project(
#     K: np.ndarray, R_cb: np.ndarray, t_cb: np.ndarray, pts_b: np.ndarray
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Project N×3 pts (robot base) into pixels. Returns (uv, z)"""
#     pts_c = (R_cb @ pts_b.T + t_cb[:, None]).T  # (N,3)
#     uv_h = (K @ pts_c.T).T  # homogeneous (N,3)
#     uv = uv_h[:, :2] / uv_h[:, 2:3]
#     return uv, pts_c[:, 2]  # z for positivity check


# # ───────────────── End‑effector pixel detector (motion + optical flow) ───────────────── #


# def _initial_ee_px_via_motion(images: np.ndarray) -> Tuple[int, int]:
#     """Return an initial EE pixel by accumulating per‑pixel motion energy."""
#     gray = [cv.cvtColor(im, cv.COLOR_BGR2GRAY) for im in images]
#     accum = np.zeros_like(gray[0], np.float32)
#     for g_prev, g_next in zip(gray[:-1], gray[1:]):
#         accum += cv.absdiff(g_next, g_prev).astype(np.float32)
#     accum = cv.GaussianBlur(accum, (31, 31), 0)
#     min_val, max_val, min_loc, max_loc = cv.minMaxLoc(accum)
#     return max_loc  # (u,v)


# def track_end_effector_pixels(images: np.ndarray) -> np.ndarray:
#     """Track the EE centre across frames.

#     Parameters
#     ----------
#     images : np.ndarray [T,H,W,3]
#         Sequence of RGB images (BGR order) for one trajectory.

#     Returns
#     -------
#     uv : np.ndarray [T,2]
#         Pixel coordinates (u,v) of the EE centre per frame.
#     """
#     if len(images) < 2:
#         raise ValueError("Need ≥2 frames to build motion energy map.")

#     # 1) Get initial point from motion energy
#     u0, v0 = _initial_ee_px_via_motion(images)
#     p0 = np.array([[u0, v0]], dtype=np.float32).reshape(-1, 1, 2)  # (1,1,2)

#     # 2) Track with pyramidal LK optical flow
#     lk_params = dict(
#         winSize=(21, 21),
#         maxLevel=3,
#         criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01),
#     )

#     gray_prev = cv.cvtColor(images[0], cv.COLOR_BGR2GRAY)
#     pts_prev = p0.copy()
#     uv_list = [pts_prev.reshape(2)]

#     for idx in range(1, len(images)):
#         gray_next = cv.cvtColor(images[idx], cv.COLOR_BGR2GRAY)
#         pts_next, status, err = cv.calcOpticalFlowPyrLK(
#             gray_prev, gray_next, pts_prev, None, **lk_params
#         )
#         if status[0, 0] == 0 or err[0, 0] > 10.0:  # tracking failed → reuse last
#             pts_next = pts_prev
#         uv_list.append(pts_next.reshape(2))
#         gray_prev, pts_prev = gray_next, pts_next

#     return np.asarray(uv_list, dtype=np.float32)


# # ──────────────────── main multi‑trajectory calibration routine ──────────────────── #


# def estimate_camera_from_trajectories(
#     trajectories: List[Tuple[np.ndarray, np.ndarray]],
#     downsample_factor: int = 5,
# ) -> Tuple[List[np.ndarray], List[float]]:
#     """Estimate camera intrinsics & extrinsics from multiple trajectories.

#     Parameters
#     ----------
#     trajectories : list of (images, poses_base)
#         * images      – np.ndarray [T,H,W,3] in BGR (OpenCV default)
#         * poses_base  – np.ndarray [T,7]  (x,y,z,qw,qx,qy,qz) in robot base frame
#     downsample_factor : int, optional
#         Process every *n*‑th frame to speed up optimisation.

#     Returns
#     -------
#     T_cam_ee_all : list[np.ndarray [N,4,4]]
#         Per‑trajectory camera‑space EE poses for the down‑sampled frames.
#     cam_params   : list[float]
#         Optimised [rvec, tvec, fx, fy, cx, cy] (len==10).
#     """
#     pts3d_all: List[np.ndarray] = []  # aggregate across trajs
#     pts2d_all: List[np.ndarray] = []
#     img_wh = None
#     lengths: List[int] = []  # for per‑trajectory slicing later

#     all_images_ds = []

#     # 1) Detect pixels on every trajectory ------------------------------------------------------
#     for images, poses_base in trajectories:
#         assert len(images) == len(poses_base), "images and poses length mismatch"
#         images_ds = images[
#             ::downsample_factor
#         ].copy()  # copy to avoid modifying original data
#         all_images_ds.append(images_ds)
#         poses_ds = poses_base[::downsample_factor]
#         lengths.append(len(poses_ds))

#         # Track EE across this trajectory once, then reuse
#         uvs_ds = track_end_effector_pixels(images_ds)

#         # Draw the tracked point for visual inspection
#         for img, (u, v) in zip(images_ds, uvs_ds):
#             cv.circle(img, (int(u), int(v)), 6, (0, 0, 255), -1)

#         for uv, pose in zip(uvs_ds, poses_ds):
#             if img_wh is None:
#                 h, w = images_ds[0].shape[:2]
#                 img_wh = (w, h)
#             pts2d_all.append(uv)
#             pts3d_all.append(pose[:3])  # only position of EE

#     pts3d = np.asarray(pts3d_all, float)
#     pts2d = np.asarray(pts2d_all, float)

#     if len(pts3d) < 40:  # heuristic – need enough points for a stable solution
#         raise RuntimeError(
#             "Too few EE detections across all trajectories. Improve detector or add data."
#         )

#     # 2) Initial parameter guess ---------------------------------------------------------------
#     fx0 = fy0 = 1.2 * max(img_wh)  # crude focal estimate
#     cx0, cy0 = img_wh[0] / 2, img_wh[1] / 2
#     rvec0 = np.zeros(3)
#     tvec0 = np.array([0, 0, 1.0])  # camera 1 m in front of base
#     x0 = np.hstack([rvec0, tvec0, [fx0, fy0, cx0, cy0]])

#     # 3) Define residuals ----------------------------------------------------------------------
#     def residuals(x: np.ndarray, Pw: np.ndarray, uv: np.ndarray) -> np.ndarray:
#         rvec, tvec = x[:3], x[3:6]
#         fx, fy, cx, cy = x[6:]
#         R_cb = R.from_rotvec(rvec).as_matrix()
#         K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

#         uv_hat, z = project(K, R_cb, tvec, Pw)
#         repro = (uv_hat - uv).ravel()

#         eps = 1e-3
#         depth_penalty = np.maximum(0.0, eps - z) * 10.0
#         return np.hstack([repro, depth_penalty])

#     # 4) Optimise via Levenberg‑Marquardt -------------------------------------------------------
#     opt = least_squares(
#         residuals,
#         x0,
#         args=(pts3d.copy(), pts2d.copy()),
#         method="lm",
#         max_nfev=6000,
#         verbose=2,
#     )

#     rvec_opt, tvec_opt = opt.x[:3], opt.x[3:6]
#     fx_opt, fy_opt, cx_opt, cy_opt = opt.x[6:]
#     R_cb_opt = R.from_rotvec(rvec_opt).as_matrix()

#     # 5) Convert every EE pose of every trajectory into camera coordinates ---------------------
#     cam_params = [*rvec_opt, *tvec_opt, fx_opt, fy_opt, cx_opt, cy_opt]
#     T_cam_ee_all: List[np.ndarray] = []

#     for (images, poses_base), L in zip(trajectories, lengths):
#         poses_ds = poses_base[::downsample_factor]
#         T_c_b = np.eye(4)
#         T_c_b[:3, :3], T_c_b[:3, 3] = R_cb_opt, tvec_opt
#         T_traj = []
#         for p in poses_ds:
#             T_b_ee = pose_vec_to_mat(p)
#             T_c_ee = T_c_b @ T_b_ee
#             T_traj.append(T_c_ee)
#         T_cam_ee_all.append(np.stack(T_traj))

#     return T_cam_ee_all, all_images_ds


# # ───────────────────────── Example usage (pseudo‑code) ───────────────────────── #
# if __name__ == "__main__":
#     # Suppose you have data as lists of (images, poses_base)
#     trajectories: List[Tuple[np.ndarray, np.ndarray]] = []
#     # trajectories.append((images1, poses1))
#     # trajectories.append((images2, poses2))
#     # ... load your data here ...

#     T_cam_ee_all, cam_params = estimate_camera_from_trajectories(trajectories)
#     print("Optimized camera parameters:", cam_params)
#     for idx, T_traj in enumerate(T_cam_ee_all):
#         print(f"Trajectory {idx}: {T_traj.shape} poses in camera frame")


import warnings
from typing import Dict, List, Tuple

import cv2 as cv
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

# ───────────────────────────── optional Segment‑Anything import ────────────────────────────── #
try:
    from segment_anything import (  # type: ignore
        SamAutomaticMaskGenerator,
        sam_model_registry,
    )

    _SAM_AVAILABLE = True
except ImportError:  # graceful fallback
    warnings.warn(
        "segment‑anything not installed – falling back to simple HSV detector.\n"
        "Install it with: pip install 'segment‑anything' and download a checkpoint to improve accuracy."
    )
    _SAM_AVAILABLE = False

# Instantiate SAM once to avoid reloading on every frame (if available) -------------------------
if _SAM_AVAILABLE:
    _SAM_CHECKPOINT_PATH = (
        "sam_vit_h_4b8939.pth"  # ← update if you keep the ckpt elsewhere
    )
    try:
        _sam = sam_model_registry["vit_h"](checkpoint=_SAM_CHECKPOINT_PATH)
        _sam.eval().to("cuda" if cv.cuda.getCudaEnabledDeviceCount() else "cpu")
        _mask_generator = SamAutomaticMaskGenerator(_sam)
    except FileNotFoundError:
        warnings.warn(
            f"SAM checkpoint not found at {_SAM_CHECKPOINT_PATH}; HSV fallback will be used."
        )
        _SAM_AVAILABLE = False

# ───────────────────────────── helper geometry ────────────────────────────── #


def pose_vec_to_mat(p: np.ndarray) -> np.ndarray:
    """[x,y,z,qw,qx,qy,qz] -> 4×4 SE(3)"""
    if len(p) == 7:
        # [x,y,z,qw,qx,qy,qz] format
        t, q = p[:3], p[3:]
    elif len(p) == 6:
        # [x,y,z,raw, pitch, yaw] format
        t, q = p[:3], R.from_euler("xyz", p[3:]).as_quat()
    T = np.eye(4)
    T[:3, :3] = R.from_quat(q).as_matrix()
    T[:3, 3] = t
    return T


def project(
    K: np.ndarray, R_cb: np.ndarray, t_cb: np.ndarray, pts_b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Project N×3 pts (robot base) into pixels. Returns (uv, z)"""
    pts_c = (R_cb @ pts_b.T + t_cb[:, None]).T  # (N,3)
    uv_h = (K @ pts_c.T).T  # homogeneous (N,3)
    uv = uv_h[:, :2] / uv_h[:, 2:3]
    return uv, pts_c[:, 2]  # z for positivity check


# ───────────────────────────── EE pixel detectors ──────────────────────────── #


def _detect_ee_px_hsv(img: np.ndarray) -> np.ndarray | None:
    """Fallback: simple HSV‑value threshold."""
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    val = hsv[..., 2]
    _, thresh = cv.threshold(val, 220, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv.contourArea)
    M = cv.moments(c)
    if M["m00"] == 0:
        return None
    u = int(M["m10"] / M["m00"])
    v = int(M["m01"] / M["m00"])
    return np.array([u, v], float)


def _detect_ee_px_sam(img: np.ndarray) -> np.ndarray | None:
    """Use Segment‑Anything to segment the robot and return the centroid of the largest mask."""
    if not _SAM_AVAILABLE:
        return _detect_ee_px_hsv(img)

    masks: List[Dict] = _mask_generator.generate(img)
    if not masks:
        return _detect_ee_px_hsv(img)

    # Choose the largest mask – in typical lab scenes the robot arm is biggest moving object
    best_mask = max(masks, key=lambda m: m["area"])
    seg = best_mask["segmentation"].astype(np.uint8)  # boolean → uint8
    M = cv.moments(seg)
    if M["m00"] == 0:
        return _detect_ee_px_hsv(img)
    u = M["m10"] / M["m00"]
    v = M["m01"] / M["m00"]
    return np.array([u, v], float)


# Choose detector implementation ---------------------------------------------------------------
detect_ee_px = _detect_ee_px_sam if _SAM_AVAILABLE else _detect_ee_px_hsv

# ──────────────────── main multi‑trajectory calibration routine ──────────────────── #


def estimate_camera_from_trajectories(
    trajectories: List[Tuple[np.ndarray, np.ndarray]], adaptive_downsample: bool = True
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Estimate camera intrinsics & extrinsics from multiple trajectories.

    Parameters
    ----------
    trajectories : list of (images, poses_base)
        * images      – np.ndarray [T,H,W,3] in BGR (OpenCV default)
        * poses_base  – np.ndarray [T,7]  (x,y,z,qw,qx,qy,qz) in robot base frame
    downsample_factor : int, optional
        Process every *n*‑th frame to speed up optimisation.

    Returns
    -------
    T_cam_ee_all : np.ndarray [M,N,4,4]
        Per‑trajectory camera‑space EE poses. M = number of trajectories, N = downsampled length.
    cam_params   : list[float]
        Optimised [rvec, tvec, fx, fy, cx, cy] (len==10).
    """
    pts3d_all: List[np.ndarray] = []  # aggregate across trajs
    pts2d_all: List[np.ndarray] = []
    img_wh = None
    lengths: List[int] = []  # for per‑trajectory slicing later

    all_images_ds = []

    # 1) Detect pixels on every trajectory ------------------------------------------------------
    for images, poses_base in trajectories:
        assert len(images) == len(poses_base), "images and poses length mismatch"
        if adaptive_downsample:
            downsample_factor = max(
                (len(images) // 20), 1
            )  # downsample to ~20 frames per trajectory
            images_ds = images[::downsample_factor]
            all_images_ds.append(images_ds)
            poses_ds = poses_base[::downsample_factor]
            lengths.append(len(poses_ds))
        else:
            images_ds = images
            all_images_ds.append(images_ds)
            poses_ds = poses_base
            lengths.append(len(poses_ds))

        for img, pose in zip(images_ds, poses_ds):
            if img_wh is None:
                h, w = img.shape[:2]
                img_wh = (w, h)
            uv = detect_ee_px(img)
            if uv is None:
                continue  # skip frame without reliable detection
            pts2d_all.append(uv)
            pts3d_all.append(pose[:3])  # only position of EE

    pts3d = np.asarray(pts3d_all, float)
    pts2d = np.asarray(pts2d_all, float)

    if len(pts3d) < 40:  # heuristic – need enough points for a stable solution
        raise RuntimeError(
            "Too few EE detections across all trajectories. Improve detector or add data."
        )

    # 2) Initial parameter guess ---------------------------------------------------------------
    fx0 = fy0 = 1.2 * max(img_wh)  # crude focal estimate
    cx0, cy0 = img_wh[0] / 2, img_wh[1] / 2
    rvec0 = np.zeros(3)
    tvec0 = np.array([0, 0, 1.0])  # camera 1 m in front of base
    x0 = np.hstack([rvec0, tvec0, [fx0, fy0, cx0, cy0]])

    # 3) Define residuals ----------------------------------------------------------------------
    def residuals(x: np.ndarray, Pw: np.ndarray, uv: np.ndarray) -> np.ndarray:
        rvec, tvec = x[:3], x[3:6]
        fx, fy, cx, cy = x[6:]
        R_cb = R.from_rotvec(rvec).as_matrix()
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        uv_hat, z = project(K, R_cb, tvec, Pw)
        repro = (uv_hat - uv).ravel()

        eps = 1e-3
        depth_penalty = np.maximum(0.0, eps - z) * 10.0
        return np.hstack([repro, depth_penalty])

    # 4) Optimise via Levenberg‑Marquardt -------------------------------------------------------
    opt = least_squares(
        residuals,
        x0,
        args=(pts3d.copy(), pts2d.copy()),
        method="lm",
        max_nfev=6000,
        verbose=2,
    )

    rvec_opt, tvec_opt = opt.x[:3], opt.x[3:6]
    fx_opt, fy_opt, cx_opt, cy_opt = opt.x[6:]
    R_cb_opt = R.from_rotvec(rvec_opt).as_matrix()

    # 5) Convert every EE pose of every trajectory into camera coordinates ---------------------
    cam_params = [*rvec_opt, *tvec_opt, fx_opt, fy_opt, cx_opt, cy_opt]
    T_cam_ee_all: List[np.ndarray] = []

    idx_start = 0
    for (images, poses_base), L in zip(trajectories, lengths):
        poses_ds = poses_base[::downsample_factor]
        T_c_b = np.eye(4)
        T_c_b[:3, :3], T_c_b[:3, 3] = R_cb_opt, tvec_opt
        T_traj = []
        for p in poses_ds:
            T_b_ee = pose_vec_to_mat(p)
            T_c_ee = T_c_b @ T_b_ee
            T_traj.append(T_c_ee)
        T_cam_ee_all.append(np.stack(T_traj))
        idx_start += L

    return T_cam_ee_all, all_images_ds


# ───────────────────────── Example usage (pseudo‑code) ───────────────────────── #
if __name__ == "__main__":
    # Suppose you have data as lists of (images, poses_base)
    trajectories: List[Tuple[np.ndarray, np.ndarray]] = []
    # trajectories.append((images1, poses1))
    # trajectories.append((images2, poses2))
    # ... load your data here ...

    T_cam_ee_all, cam_params = estimate_camera_from_trajectories(trajectories)
    print("Optimized camera parameters:", cam_params)
    print("T_cam_ee_all shape:", T_cam_ee_all.shape)
