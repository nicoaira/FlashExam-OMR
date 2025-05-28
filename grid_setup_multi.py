#!/usr/bin/env python3
import cv2, numpy as np, json, argparse, sys

def auto_find_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sq=[]
    for c in cnts:
        a = cv2.contourArea(c)
        if not (2000<a<10000): continue
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,0.02*peri,True)
        if len(approx)==4:
            x,y,w,h = cv2.boundingRect(approx)
            if 0.8<=w/float(h)<=1.2:
                sq.append((x+w/2,y+h/2))
    if len(sq)!=4:
        raise RuntimeError(f"Auto‐corner detect found {len(sq)} markers")
    # TL,TR,BR,BL ordering
    sq = sorted(sq, key=lambda p:(p[1],p[0]))
    tl, tr = sq[0], sq[1]
    bl, br = sq[2], sq[3]
    if bl[0]>br[0]: bl,br = br,bl
    return np.array([tl,tr,br,bl],dtype='float32')

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('image', help='Sample scanned sheet')
    p.add_argument('--columns', type=int, required=True, help='Number of question-columns')
    p.add_argument('--rows', type=int, required=True, help='Questions per column')
    p.add_argument('--options', default='A,B,C,D', help='Comma-separated option labels')
    args = p.parse_args()

    opts = args.options.split(',')
    img = cv2.imread(args.image)
    if img is None:
        print("Cannot open", args.image); sys.exit(1)

    # 1) auto‐warp via the 4 black corners
    try:
        pts = auto_find_corners(img)
    except Exception as e:
        print("Auto‐corner failed:", e); sys.exit(1)

    tl,tr,br,bl = pts
    wA, wB = np.linalg.norm(br-bl), np.linalg.norm(tr-tl)
    hA, hB = np.linalg.norm(tr-br), np.linalg.norm(tl-bl)
    W, H = int(max(wA,wB)), int(max(hA,hB))
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]],dtype='float32')
    M = cv2.getPerspectiveTransform(pts, dst)
    warp = cv2.warpPerspective(img, M, (W,H))

    # Per-grid calibration and ROI selection
    MAX_W, MAX_H = 1200, 800
    cal_scale = min(1.0, MAX_W/warp.shape[1], MAX_H/warp.shape[0])
    disp = cv2.resize(warp, (int(warp.shape[1]*cal_scale), int(warp.shape[0]*cal_scale)))
    config = {'warp_w': W, 'warp_h': H,
              'columns': args.columns, 'rows': args.rows,
              'options': opts,
              'grids': []}
    window = 'Grid setup: g=add grid, q=finish'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, min(warp.shape[1], 1200), min(warp.shape[0], 800))
    # --- Step 0: Select name and ID rectangles ---
    rect_labels = ["name_rect", "id_rect"]
    rects = {}
    for label in rect_labels:
        print(f"Select the rectangle for {label.replace('_rect','').upper()} (drag to select, ENTER to confirm)")
        roi_win = f"Select {label}"
        cv2.namedWindow(roi_win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(roi_win, min(warp.shape[1], 1200), min(warp.shape[0], 800))
        r = cv2.selectROI(roi_win, warp, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(roi_win)
        x, y, w, h = [int(v) for v in r]
        if w == 0 or h == 0:
            print(f"No selection made for {label}"); sys.exit(1)
        # Normalize
        rects[label] = [x/W, y/H, w/W, h/H]
    config['name_rect'] = rects['name_rect']
    config['id_rect'] = rects['id_rect']

    for grid_idx in range(args.columns):
        # 1. Calibrate bubble for this grid
        while True:
            cal_pts = []
            base_disp = disp.copy()
            def cal_mouse(event, x, y, flags, param):
                global disp, base_disp
                temp_disp = base_disp.copy()
                # draw crosshair
                cv2.line(temp_disp, (0, y), (temp_disp.shape[1], y), (0,255,255), 1)
                cv2.line(temp_disp, (x, 0), (x, temp_disp.shape[0]), (0,255,255), 1)
                # draw existing green marks at their display coords
                for ox, oy, dx, dy in cal_pts:
                    cv2.circle(temp_disp, (dx, dy), 2, (0,255,0), -1)
                cv2.imshow('Calibrate bubbles', temp_disp)
                if event == cv2.EVENT_LBUTTONDOWN and len(cal_pts) < 4:
                    ox, oy = int(x / cal_scale), int(y / cal_scale)
                    cal_pts.append((ox, oy, x, y))
            cv2.namedWindow('Calibrate bubbles', cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('Calibrate bubbles', cal_mouse)
            print(f'Grid {grid_idx+1}: Calibration: click centers of A1 then B1, then left rim and right rim.')
            while len(cal_pts) < 4:
                cv2.waitKey(50)
            cv2.destroyWindow('Calibrate bubbles')
            if len(cal_pts) == 4:
                pA, pB, rimL, rimR = cal_pts
                spacing_px = abs(pB[0] - pA[0])
                radius_px = abs(rimR[0] - rimL[0]) // 2
                print(f'Grid {grid_idx+1} calibration: spacing={spacing_px}px, radius={radius_px}px')
                choice = input("Accept this calibration? (y = yes, r = retry, c = cancel): ").strip().lower()
                if choice == 'y':
                    break
                elif choice == 'c':
                    print('Calibration cancelled. Exiting.')
                    sys.exit(1)
                else:
                    print('Retry calibration for this grid.')
                    continue
            else:
                print(f'Grid {grid_idx+1} calibration incomplete, retrying.')
                continue
        # 2. Select grid ROI for this grid using two crosshair clicks
        while True:
            print(f'Grid {grid_idx+1}: Click top-left and bottom-right corners for this grid.')
            roi_pts = []  # (x, y) in display coords
            base_disp = disp.copy()
            def roi_mouse(event, x, y, flags, param):
                global disp, base_disp
                temp_disp = base_disp.copy()
                # draw crosshair
                cv2.line(temp_disp, (0, y), (temp_disp.shape[1], y), (0,255,255), 1)
                cv2.line(temp_disp, (x, 0), (x, temp_disp.shape[0]), (0,255,255), 1)
                # draw existing points
                for px, py in roi_pts:
                    cv2.circle(temp_disp, (px, py), 2, (0,255,0), -1)
                # if two points, draw rectangle
                if len(roi_pts) == 1:
                    cv2.rectangle(temp_disp, (roi_pts[0][0], roi_pts[0][1]), (x, y), (0,255,0), 1)
                cv2.imshow(window, temp_disp)
                if event == cv2.EVENT_LBUTTONDOWN and len(roi_pts) < 2:
                    roi_pts.append((x, y))
            cv2.setMouseCallback(window, roi_mouse)
            while len(roi_pts) < 2:
                cv2.waitKey(50)
            cv2.setMouseCallback(window, lambda *args: None)
            x0, y0 = roi_pts[0]
            x1, y1 = roi_pts[1]
            x, y = min(x0, x1), min(y0, y1)
            w, h = abs(x1 - x0), abs(y1 - y0)
            if w == 0 or h == 0:
                print("Empty ROI, try again.")
                continue
            xf, yf = int(x/cal_scale), int(y/cal_scale)
            wf, hf = int(w/cal_scale), int(h/cal_scale)
            # Preview and confirm
            overlay = warp.copy()
            cv2.rectangle(overlay, (xf, yf), (xf+wf, yf+hf), (0,255,0), 2)
            prev = cv2.resize(overlay, (disp.shape[1], disp.shape[0]))
            cv2.imshow(window, prev)
            choice = input("Accept this grid ROI? (y = yes, r = retry, c = cancel): ").strip().lower()
            if choice == 'y':
                break
            elif choice == 'c':
                print('Grid ROI cancelled. Exiting.')
                sys.exit(1)
            else:
                print('Retry grid ROI selection.')
                continue
        # Save this grid's ROI and calibration
        config['grids'].append({
            'x': xf/W, 'y': yf/H, 'w': wf/W, 'h': hf/H,
            'bubble_spacing_px': spacing_px,
            'bubble_radius_px': radius_px
        })
        cv2.rectangle(disp, (x,y),(x+w,y+h),(0,255,0),2)
        print(f"Grid #{len(config['grids'])} accepted.")
    cv2.destroyAllWindows()

    # 3) save final config after grids
    with open('grid_config.json','w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved grid_config.json with {len(config['grids'])} grids and per-grid bubble calibration.")