"""Utility functions for homography estimation and warping.

This module contains the normalized DLT homography estimation and a
pure-Python inverse-mapping warp with bilinear interpolation.
"""
import math
import numpy as np


def normalize_points(pts):
    pts = np.asarray(pts, dtype=np.float64)
    centroid = pts.mean(axis=0)
    dists = np.sqrt(((pts - centroid) ** 2).sum(axis=1))
    mean_dist = dists.mean()
    if mean_dist == 0:
        scale = 1.0
    else:
        scale = math.sqrt(2) / mean_dist
    T = np.array([[scale, 0, -scale * centroid[0]],
                  [0, scale, -scale * centroid[1]],
                  [0, 0, 1]])
    ones = np.ones((pts.shape[0], 1))
    homo = np.hstack([pts, ones])
    norm = (T @ homo.T).T
    return norm[:, :2], T


def estimate_homography(src_pts, dst_pts):
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    if src.shape[0] < 4 or dst.shape[0] < 4:
        raise ValueError("Need at least 4 points")
    if src.shape != dst.shape:
        raise ValueError("src and dst must have same shape")

    nsrc, Tsrc = normalize_points(src)
    ndst, Tdst = normalize_points(dst)

    N = src.shape[0]
    A = []
    for i in range(N):
        x, y = nsrc[i]
        u, v = ndst[i]
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.array(A, dtype=np.float64)
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    Hn = h.reshape((3, 3))
    H = np.linalg.inv(Tdst) @ Hn @ Tsrc
    if abs(H[2, 2]) < 1e-12:
        H = H / (H.flat[-1] + 1e-12)
    else:
        H = H / H[2, 2]
    return H


def bilinear_interpolate(img, x, y):
    h, w = img.shape[:2]
    if x < 0 or x >= w - 1 or y < 0 or y >= h - 1:
        xi = int(round(x))
        yi = int(round(y))
        if xi < 0 or xi >= w or yi < 0 or yi >= h:
            if img.ndim == 3:
                return np.zeros((img.shape[2],), dtype=img.dtype)
            else:
                return 0
        return img[yi, xi].astype(np.float64)

    x0 = int(math.floor(x))
    x1 = x0 + 1
    y0 = int(math.floor(y))
    y1 = y0 + 1

    Ia = img[y0, x0].astype(np.float64)
    Ib = img[y0, x1].astype(np.float64)
    Ic = img[y1, x0].astype(np.float64)
    Id = img[y1, x1].astype(np.float64)

    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)

    res = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return res


def warp_image(src_img, H, dst_shape):
    src = src_img
    h_dst, w_dst = dst_shape[0], dst_shape[1]
    channels = src.shape[2] if src.ndim == 3 else 1
    if channels == 1:
        dst = np.zeros((h_dst, w_dst), dtype=src.dtype)
    else:
        dst = np.zeros((h_dst, w_dst, channels), dtype=src.dtype)

    Hinv = np.linalg.inv(H)
    for y in range(h_dst):
        for x in range(w_dst):
            pd = np.array([x, y, 1.0], dtype=np.float64)
            ps = Hinv @ pd
            if abs(ps[2]) < 1e-12:
                continue
            sx = ps[0] / ps[2]
            sy = ps[1] / ps[2]
            val = bilinear_interpolate(src, sx, sy)
            if channels == 1:
                dst[y, x] = np.clip(val, 0, 255)
            else:
                dst[y, x, :] = np.clip(val, 0, 255)
    return dst.astype(src.dtype)
