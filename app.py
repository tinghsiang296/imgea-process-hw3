#!/usr/bin/env python3
"""
Homography estimation and image warping without using built-in warp functions.

Features:
- Estimate homography from >=4 point correspondences (DLT with normalization)
- Warp image using inverse mapping + bilinear interpolation
- Two interactive tests:
  1) Rectify a photographed rectangular poster by clicking 4 corners
  2) Project an image onto an arbitrary quadrilateral by clicking 4 dest points

Usage:
  python app.py

Click on images as prompted (matplotlib window). Close the plot to continue.
"""

import sys
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt


def normalize_points(pts):
    # pts: N x 2
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
    # convert pts to homogeneous and apply T
    ones = np.ones((pts.shape[0], 1))
    homo = np.hstack([pts, ones])
    norm = (T @ homo.T).T
    return norm[:, :2], T


def estimate_homography(src_pts, dst_pts):
    """Estimate homography H such that dst ~ H * src.
    src_pts, dst_pts: Nx2 arrays, N>=4
    Returns 3x3 homography with H[2,2]=1
    """
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
    # SVD
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    Hn = h.reshape((3, 3))
    # Denormalize: H = Tdst^{-1} * Hn * Tsrc
    H = np.linalg.inv(Tdst) @ Hn @ Tsrc
    # normalize so H[2,2]=1
    if abs(H[2, 2]) < 1e-12:
        H = H / (H.flat[-1] + 1e-12)
    else:
        H = H / H[2, 2]
    return H


def bilinear_interpolate(img, x, y):
    """Bilinear interpolation for single coordinate (x float, y float).
    img: H x W x C or H x W
    x: horizontal (col), y: vertical (row)
    Returns pixel value (C,) or scalar.
    """
    h, w = img.shape[:2]
    if x < 0 or x >= w - 1 or y < 0 or y >= h - 1:
        # If outside or on edge where interpolation needs neighbors, clamp and return nearest
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

    res = wa[:, None] * Ia if False else wa * Ia
    res = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return res


def warp_image(src_img, H, dst_shape):
    """Warp src_img into destination canvas using homography H mapping src->dst.
    dst_shape: (h, w, [c]) or (h, w)
    Implementation uses inverse mapping: for each dst pixel (x_d,y_d) compute src coords by H^{-1}.
    """
    src = src_img
    h_dst, w_dst = dst_shape[0], dst_shape[1]
    channels = src.shape[2] if src.ndim == 3 else 1
    if channels == 1:
        dst = np.zeros((h_dst, w_dst), dtype=src.dtype)
    else:
        dst = np.zeros((h_dst, w_dst, channels), dtype=src.dtype)

    Hinv = np.linalg.inv(H)

    # For each pixel in destination, map back
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


def show_points(img, pts, title=""):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pts = np.asarray(pts)
    plt.scatter(pts[:, 0], pts[:, 1], c='r', s=50)
    for i, p in enumerate(pts):
        plt.text(p[0] + 5, p[1] + 5, str(i + 1), color='yellow', fontsize=12)
    plt.title(title)


def interactive_select_points(img, n=4, title='Select points'):
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title + f" (click {n} points in order)")
    pts = plt.ginput(n, timeout=0)
    plt.close(fig)
    pts = [(p[0], p[1]) for p in pts]
    return np.array(pts)


def rectify_poster(img):
    print("請在圖片上依順時針或逆時針點選矩形海報的 4 個角（建議從左上開始）...")
    src_pts = interactive_select_points(img, 4, title='Select poster corners')
    # Determine output size by measuring width and height from source points
    # We assume ordering corresponds to rectangle corners: tl, tr, br, bl
    # Compute width as average of top and bottom edge lengths, height as average of left and right
    def dist(a, b):
        return np.linalg.norm(a - b)
    w1 = dist(src_pts[0], src_pts[1])
    w2 = dist(src_pts[3], src_pts[2])
    h1 = dist(src_pts[0], src_pts[3])
    h2 = dist(src_pts[1], src_pts[2])
    w = int(round(max(w1, w2)))
    h = int(round(max(h1, h2)))
    dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)

    H = estimate_homography(src_pts, dst_pts)
    warped = warp_image(img, H, (h, w))

    # Show before and after with correspondences
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].scatter(src_pts[:, 0], src_pts[:, 1], c='r', s=50)
    for i, p in enumerate(src_pts):
        axes[0].text(p[0] + 5, p[1] + 5, str(i + 1), color='yellow')
    axes[0].set_title('Original with selected points')

    axes[1].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    axes[1].scatter(dst_pts[:, 0], dst_pts[:, 1], c='r', s=50)
    for i, p in enumerate(dst_pts):
        axes[1].text(p[0] + 5, p[1] + 5, str(i + 1), color='yellow')
    axes[1].set_title('Rectified')
    plt.show()


def project_image_onto_quad(texture_img, background_img=None):
    if background_img is None:
        # create white background same size as texture for interactive selection
        h, w = texture_img.shape[:2]
        background = 255 * np.ones((max(600, h), max(800, w), 3), dtype=np.uint8)
    else:
        background = background_img.copy()

    print("請在背景圖上點選 4 個目標角點（依順時針或逆時針），貼圖會投影到這四邊形上")
    dst_pts = interactive_select_points(background, 4, title='Select quad on background')

    # Source corners of texture (tl,tr,br,bl)
    th, tw = texture_img.shape[:2]
    src_pts = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float64)

    H = estimate_homography(src_pts, dst_pts)

    warped = warp_image(texture_img, H, (background.shape[0], background.shape[1]))

    # Composite: write non-black pixels (or any mapped pixel) onto background
    mask = np.any(warped != 0, axis=2)
    comp = background.copy()
    comp[mask] = warped[mask]

    # show
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
    axes[0].scatter(dst_pts[:, 0], dst_pts[:, 1], c='r')
    axes[0].set_title('Background with target quad')
    axes[1].imshow(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Result (texture projected)')
    plt.show()


def main():
    print("Perspective Transform / Homography demo (no built-in warp)")
    while True:
        print('\n選項:')
        print(' 1) Rectify photographed poster (拉正)')
        print(' 2) Project an image onto quadrilateral (貼圖投影)')
        print(' 3) Exit')
        choice = input('請輸入選項數字: ').strip()
        if choice == '1':
            path = input('輸入待拉正圖檔路徑 (或按 Enter 使用 sample): ').strip()
            if path == '':
                path = None
            if path is None:
                print('請提供一張拍歪的海報或文件照片路徑')
                continue
            img = cv2.imread(path)
            if img is None:
                print('讀取失敗，請確認路徑正確。')
                continue
            rectify_poster(img)
        elif choice == '2':
            tex_path = input('輸入欲投影的貼圖圖檔路徑: ').strip()
            if tex_path == '':
                print('請提供貼圖路徑')
                continue
            tex = cv2.imread(tex_path)
            if tex is None:
                print('讀取貼圖失敗')
                continue
            bg_choice = input('是否提供背景圖 (y/n)? ').strip().lower()
            bg = None
            if bg_choice == 'y':
                bg_path = input('輸入背景圖路徑: ').strip()
                bg = cv2.imread(bg_path)
                if bg is None:
                    print('讀取背景失敗，使用白色背景')
                    bg = None
            project_image_onto_quad(tex, bg)
        elif choice == '3':
            print('退出')
            break
        else:
            print('無效選項')


if __name__ == '__main__':
    main()


