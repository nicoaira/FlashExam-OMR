#!/usr/bin/env python3
import cv2
import numpy as np
import json
import argparse

# --- AUTO DETECT CORNERS ---
def auto_find_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 2000 or area > 10000:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            x,y,w,h = cv2.boundingRect(approx)
            ratio = w/float(h)
            if 0.8 <= ratio <= 1.2:
                cx, cy = x + w/2, y + h/2
                squares.append((cx, cy))
    if len(squares) != 4:
        raise RuntimeError(f"Auto corner detection found {len(squares)} markers")
    # sort into TL,TR,BR,BL
    sq = sorted(squares, key=lambda p:(p[1],p[0]))
    tl, tr = sq[:2]
    bl, br = sq[2:]
    if bl[0] > br[0]:
        bl, br = br, bl
    return np.array([tl, tr, br, bl], dtype='float32')

# globals for fallback click handlers
pts = []
grid_pts = []
scale = 1.0

def click_grid(event, x, y, flags, param):
    global grid_pts
    if event == cv2.EVENT_LBUTTONDOWN and len(grid_pts) < 2:
        ox, oy = int(x/scale), int(y/scale)
        grid_pts.append((ox, oy))
        cv2.circle(param, (x, y), 5, (0,0,255), -1)
        cv2.imshow(window_grid, param)

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('image', help='Path to a sample scanned sheet')
    p.add_argument('--cols', type=int, required=True, help='Number of columns')
    p.add_argument('--rows', type=int, required=True, help='Rows per column')
    p.add_argument('--options', default='A,B,C,D', help='Comma list of options')
    args = p.parse_args()

    img_full = cv2.imread(args.image)
    h,w = img_full.shape[:2]

    # attempt auto-detect
    try:
        pts_arr = auto_find_corners(img_full)
        print("Auto-detected corners:", pts_arr.tolist())
    except Exception as e:
        print("Auto detect failed:", e)
        print("Please manually click TL and BR of bubble area only.")
        pts_arr = None   # skip warp until grid bbox click

    # warp if auto-detected, otherwise we still need corners manual
    if pts_arr is None:
        print("Manual corner picking not yet implemented—please ensure your sheet uses the standard marks.")
        exit(1)

    # perform perspective warp
    tl, tr, br, bl = pts_arr
    maxW = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
    maxH = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype='float32')
    M = cv2.getPerspectiveTransform(pts_arr, dst)
    warp_full = cv2.warpPerspective(img_full, M, (maxW, maxH))
    wh, ww = warp_full.shape[:2]

    # scale display
    MAX_W, MAX_H = 1200, 800
    scale = min(1.0, MAX_W/ww, MAX_H/wh)
    warp_disp = cv2.resize(warp_full, (int(ww*scale), int(wh*scale)))

    # now just pick grid bbox
    window_grid = 'Pick TL and BR of bubble area'
    cv2.namedWindow(window_grid, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_grid, warp_disp.shape[1], warp_disp.shape[0])
    cv2.setMouseCallback(window_grid, click_grid, warp_disp)
    cv2.imshow(window_grid, warp_disp)
    cv2.waitKey(0)
    cv2.destroyWindow(window_grid)
    if len(grid_pts) != 2:
        print("Need 2 grid corner clicks"); exit(1)

    # normalize and save
    (gx1,gy1),(gx2,gy2) = grid_pts
    x1n, y1n = gx1/ww, gy1/wh
    x2n, y2n = gx2/ww, gy2/wh
    colw = (x2n - x1n)/args.cols
    x_offsets = [x1n + i*colw for i in range(args.cols)]
    y_start = y1n
    y_step = (y2n - y1n)/(args.rows-1)

    cfg = {
        'warp_w': ww, 'warp_h': wh,
        'cols': args.cols, 'rows': args.rows,
        'options': args.options.split(','),
        'x_offsets': x_offsets,
        'y_start': y_start, 'y_step': y_step,
        'col_width': colw
    }
    with open('grid_config.json','w') as f:
        json.dump(cfg, f, indent=2)
    print("Saved grid_config.json")
