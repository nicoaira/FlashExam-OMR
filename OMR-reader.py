# core dependencies
import cv2
import numpy as np
import pandas as pd
import glob
import os
import json
import subprocess

# Will load grid config and populate bubble_positions
def init_grid():
    import json
    cfg = json.load(open('grid_config.json'))
    # Validate config format
    if 'grids' not in cfg and 'x_offsets' not in cfg:
        raise RuntimeError(
            "Invalid grid_config.json: missing grid definitions. "
            "Please regenerate using grid_setup_multi.py with --columns <num> --rows <num> --options <labels>."
        )
    # Base warp size
    WARP_W = cfg['warp_w']; WARP_H = cfg['warp_h']
    # Read options and rows
    OPTIONS = tuple(cfg['options'])
    ROWS = cfg['rows']
    bp = []
    if 'grids' in cfg:
        # Multi-grid mode
        grids = cfg['grids']
        COLS = len(grids)
        for col, g in enumerate(grids):
            x0, y0, w0, h0 = g['x'], g['y'], g['w'], g['h']
            for i in range(ROWS):
                for j, opt in enumerate(OPTIONS):
                    # center of bubble cell
                    nx = x0 + w0 * (j + 0.5) / len(OPTIONS)
                    ny = y0 + h0 * (i + 0.5) / ROWS
                    qnum = col * ROWS + i + 1
                    bp.append((qnum, opt, (nx, ny)))
    else:
        # Legacy single-grid mode
        COLS = cfg.get('columns', cfg.get('cols'))
        x_offsets = cfg['x_offsets']
        y_start = cfg['y_start']; y_step = cfg['y_step']
        col_width = cfg.get('col_width', 0.2)
        for col in range(COLS):
            for i in range(ROWS):
                y = y_start + i*y_step
                xs = np.linspace(x_offsets[col], x_offsets[col] + col_width, len(OPTIONS))
                for j, x in enumerate(xs):
                    qnum = col * ROWS + i + 1
                    bp.append((qnum, OPTIONS[j], (x, y)))
    # Export globals
    globals().update({
        'WARP_W': WARP_W, 'WARP_H': WARP_H,
        'COLS': COLS, 'ROWS': ROWS,
        'OPTIONS': OPTIONS,
        'bubble_positions': bp
    })
    return

def find_markers(img):
    """Find the four filled corner squares and return their centroids."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 2000:                 # ignore tiny blobs
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ratio = w / float(h)
            if 0.8 <= ratio <= 1.2:    # roughly square
                cx, cy = x + w/2, y + h/2
                squares.append((cx, cy))

    if len(squares) != 4:
        # dump threshold image for inspection
        cv2.imwrite("debug_markers.png", th)
        raise RuntimeError(f"Could not find 4 markers, found {len(squares)} – see debug_markers.png")

    # sort into tl, tr, br, bl
    squares = sorted(squares, key=lambda p: (p[1], p[0]))
    tl, tr = squares[:2]
    bl, br = squares[2:]
    if bl[0] > br[0]:
        bl, br = br, bl

    return np.array([tl, tr, br, bl], dtype="float32")

def warp_sheet(img):
    pts = find_markers(img)
    dst = np.array([[0,0],[WARP_W,0],[WARP_W,WARP_H],[0,WARP_H]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, M, (WARP_W, WARP_H))

def detect_answers(warped):
    # Use template positions to compute fills
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    answers = {}
    for q, opt, (nx, ny) in bubble_positions:
        x = int(nx * WARP_W)
        y = int(ny * WARP_H)
        # crop around the bubble
        r = 20
        mask = gray[y-r:y+r, x-r:x+r]
        _, m = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        fill = cv2.countNonZero(m)
        # store fills per option
        if q not in answers:
            answers[q] = []
        answers[q].append((fill, opt, (x, y)))
    # pick max fill per question
    results = {}
    for q, lst in answers.items():
        fill, opt, pos = max(lst, key=lambda x: x[0])
        results[q] = (opt, pos)
    return results

def process_folder(folder, out_csv="results.csv"):
    rows = []
    for fname in glob.glob(os.path.join(folder, "*.png")):
        img = cv2.imread(fname)
        warped = warp_sheet(img)
        results = detect_answers(warped)  # returns {q: (opt, pos)}
        # split into answers and positions
        ans = {q: opt for q, (opt, pos) in results.items()}
        positions = {q: pos for q, (opt, pos) in results.items()}
        row = {"file": os.path.basename(fname)}
        row.update({f"Q{q}": ans[q] for q in sorted(ans)})
        rows.append(row)
        # draw a circle around the chosen bubble for each question using template positions
        debug = warped.copy()
        for q, (opt, pos) in results.items():
            x, y = pos
            cv2.circle(debug, (x, y), 30, (0, 0, 255), 2)
        debug_name = os.path.splitext(fname)[0] + '_debug.png'
        cv2.imwrite(debug_name, debug)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")

if __name__=="__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("input_folder", help="folder with scanned .png sheets")
    p.add_argument("--csv", default="results.csv")
    args = p.parse_args()
    # If grid not configured, launch interactive setup
    if not os.path.exists('grid_config.json'):
        print('grid_config.json not found; launching grid setup GUI...')
        # pick first sheet image as sample
        imgs = glob.glob(os.path.join(args.input_folder, '*.png'))
        if not imgs:
            print('No sample PNG found in folder. Please provide an example scan.')
            sys.exit(1)
        sample = imgs[0]
        subprocess.call(['python3', 'grid_setup.py', sample,
                         '--cols', input('Columns? '),
                         '--rows', input('Rows per column? '),
                         '--options', input('Options (comma-separated)? ')])
        print('Configuration complete. Re-run script.')
        sys.exit(0)
    # Load grid configuration and bubble positions
    init_grid()
    process_folder(args.input_folder, args.csv)