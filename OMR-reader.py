#!/usr/bin/env python3
# core dependencies
import cv2
import numpy as np
import pandas as pd
import glob
import os
import json
import subprocess
import csv as csvmod
import importlib

# Default minimum fill threshold
MIN_FILL = 200
DEBUG = 1  # 0: none, 1: all except bubble fill, 2: all

# Will load grid config and populate bubble_positions
def set_min_fill(val):
    global MIN_FILL
    MIN_FILL = val

def init_grid():
    import json
    cfg = json.load(open('grid_config.json'))
    if 'grids' not in cfg and 'x_offsets' not in cfg:
        raise RuntimeError(
            "Invalid grid_config.json: missing grid definitions. "
            "Please regenerate using grid_setup_multi.py with --columns <num> --rows <num> --options <labels>."
        )
    WARP_W = cfg['warp_w']; WARP_H = cfg['warp_h']
    OPTIONS = tuple(cfg['options'])
    ROWS = cfg['rows']
    bp = []
    grid_bubble_params = []
    if 'grids' in cfg:
        grids = cfg['grids']
        COLS = len(grids)
        for col, g in enumerate(grids):
            x0, y0, w0, h0 = g['x'], g['y'], g['w'], g['h']
            spacing = g.get('bubble_spacing_px')
            radius = g.get('bubble_radius_px')
            x0_px = x0 * WARP_W
            y0_px = y0 * WARP_H
            h0_px = h0 * WARP_H
            grid_bubble_params.append({'spacing': spacing, 'radius': radius, 'x0_px': x0_px, 'y0_px': y0_px, 'h0_px': h0_px})
            for i in range(ROWS):
                y_px = y0_px + (i + 0.5) * (h0_px / ROWS)
                for j, opt in enumerate(OPTIONS):
                    x_px = x0_px + (radius or 0) + j * (spacing or 0)
                    nx = x_px / WARP_W
                    ny = y_px / WARP_H
                    qnum = col * ROWS + i + 1
                    bp.append((qnum, opt, (nx, ny), col))
    else:
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
                    bp.append((qnum, OPTIONS[j], (x, y), col))
    globals().update({
        'WARP_W': WARP_W, 'WARP_H': WARP_H,
        'COLS': COLS, 'ROWS': ROWS,
        'OPTIONS': OPTIONS,
        'bubble_positions': bp,
        'grid_bubble_params': grid_bubble_params
    })
    return

def find_markers(img):
    """Find the four filled corner squares and return their centroids."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if DEBUG >= 1:
        print(f"[DEBUG] Total contours found: {len(cnts)}")
    squares = []
    for idx, c in enumerate(cnts):
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if DEBUG >= 1:
            print(f"[DEBUG] Contour {idx}: area={area}, bbox=({x},{y},{w},{h})")
        if area < 2000:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            ratio = w / float(h)
            if DEBUG >= 1:
                print(f"[DEBUG] Contour {idx} is quadrilateral, ratio={ratio}")
            if 0.8 <= ratio <= 1.2:
                cx, cy = x + w/2, y + h/2
                squares.append((cx, cy))
                if DEBUG >= 1:
                    print(f"[DEBUG] Contour {idx} accepted as marker at ({cx},{cy})")
    if DEBUG >= 1:
        print(f"[DEBUG] Total markers found: {len(squares)}")
    if len(squares) != 4:
        cv2.imwrite("debug_markers.png", th)
        if DEBUG >= 1:
            print(f"[DEBUG] Could not find 4 markers, found {len(squares)} – see debug_markers.png")
        raise RuntimeError(f"Could not find 4 markers, found {len(squares)} – see debug_markers.png")
    pts = np.array(squares, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    if DEBUG >= 1:
        print(f"[DEBUG] Marker coordinates: tl={tl}, tr={tr}, br={br}, bl={bl}")
    return np.array([tl, tr, br, bl], dtype="float32")

def warp_sheet(img):
    pts = find_markers(img)
    dst = np.array([[0,0],[WARP_W,0],[WARP_W,WARP_H],[0,WARP_H]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, M, (WARP_W, WARP_H))

def detect_answers(warped):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    answers = {}
    for q, opt, (nx, ny), col in bubble_positions:
        x = int(nx * WARP_W)
        y = int(ny * WARP_H)
        if 'grid_bubble_params' in globals() and col < len(grid_bubble_params):
            r = int(grid_bubble_params[col]['radius'] or 20)
        else:
            r = 20
        y1 = max(0, y-r); y2 = min(WARP_H, y+r)
        x1 = max(0, x-r); x2 = min(WARP_W, x+r)
        mask = gray[y1:y2, x1:x2]
        _, m = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        fill = cv2.countNonZero(m)
        if DEBUG == 2:
            print(f"Q{q} Opt:{opt} Fill:{fill} (min_fill={MIN_FILL})")
        if q not in answers:
            answers[q] = []
        answers[q].append((fill, opt, (x, y), col))
    results = {}
    for q, lst in answers.items():
        fill, opt, pos, col = max(lst, key=lambda x: x[0])
        if fill < MIN_FILL:
            if DEBUG == 2:
                print(f"Q{q} selected: - (no bubble above threshold, max fill={fill})")
            elif DEBUG == 1:
                print(f"Q{q} selected: - (no bubble above threshold)")
            opt = ""
        else:
            if DEBUG == 2:
                print(f"Q{q} selected: {opt} (fill={fill})")
            elif DEBUG == 1:
                print(f"Q{q} selected: {opt}")
        results[q] = (opt, pos, col)
    return results

def process_folder(folder, out_csv="results.csv", output_dir="output",
                   answers_csv=None, answers_json=None, scoring_json=None,
                   get_info=False, hand_writing=False, device='cpu'):
    os.makedirs(output_dir, exist_ok=True)
    detections_dir = os.path.join(output_dir, "detections")
    os.makedirs(detections_dir, exist_ok=True)
    students_info_dir = os.path.join(output_dir, "students-info")
    os.makedirs(students_info_dir, exist_ok=True)

    info_rows = []
    handwriting_ocr = None
    if hand_writing:
        handwriting_ocr = importlib.import_module('handwriting_ocr')

    with open('grid_config.json') as f:
        grid_cfg = json.load(f)
    name_rect = grid_cfg.get('name_rect')
    id_rect = grid_cfg.get('id_rect')

    rows = []
    grades_rows = []

    correct_answers = None
    if answers_json:
        with open(answers_json) as f:
            correct_answers = json.load(f)
        correct_answers = {int(k): v for k, v in correct_answers.items()}
    elif answers_csv:
        correct_answers = {}
        with open(answers_csv, newline='') as f:
            reader = csvmod.DictReader(f)
            for row in reader:
                q = int(row['Pregunta'])
                correct_answers[q] = row['Respuesta'].strip().upper()

    scoring = {"correct": 1, "incorrect": 0, "unanswered": 0}
    if scoring_json:
        with open(scoring_json) as f:
            scoring.update(json.load(f))

    for fname in glob.glob(os.path.join(folder, "*.png")):
        print(f"Processing image {os.path.basename(fname)}...")
        img = cv2.imread(fname)
        warped = warp_sheet(img)

        base = os.path.splitext(os.path.basename(fname))[0]
        student_dir = os.path.join(students_info_dir, base)
        os.makedirs(student_dir, exist_ok=True)
        for label, rect in zip(["name", "id"], [name_rect, id_rect]):
            if rect:
                x, y, w, h = rect
                pts = find_markers(img)
                dst = np.array([[0,0],[WARP_W,0],[WARP_W,WARP_H],[0,WARP_H]], dtype="float32")
                M = cv2.getPerspectiveTransform(pts, dst)
                Minv = cv2.getPerspectiveTransform(dst, pts)
                x1w = int(x * WARP_W)
                y1w = int(y * WARP_H)
                x2w = int((x + w) * WARP_W)
                y2w = int((y + h) * WARP_H)
                warped_corners = np.array([
                    [x1w, y1w],
                    [x2w, y1w],
                    [x2w, y2w],
                    [x1w, y2w]
                ], dtype="float32").reshape(-1,1,2)
                orig_corners = cv2.perspectiveTransform(warped_corners, Minv).reshape(4,2)
                x_min, y_min = orig_corners.min(axis=0).astype(int)
                x_max, y_max = orig_corners.max(axis=0).astype(int)
                x_min = max(0, x_min); y_min = max(0, y_min)
                x_max = min(img.shape[1], x_max); y_max = min(img.shape[0], y_max)
                crop = img[y_min:y_max, x_min:x_max]
                cv2.imwrite(os.path.join(student_dir, f"{label}.png"), crop)

        if hand_writing:
            name_img = os.path.join(student_dir, "name.png")
            id_img = os.path.join(student_dir, "id.png")
            name_text, id_text = handwriting_ocr.recognize_name_id(name_img, id_img, device=device)
            info_rows.append({"image": os.path.basename(fname), "name": name_text, "id": id_text})

        results = detect_answers(warped)
        ans = {q: (opt or '') for q, (opt, pos, col) in results.items()}
        positions = {q: pos for q, (opt, pos, col) in results.items()}
        row = {"file": os.path.basename(fname)}
        row.update({f"Q{q}": ans[q] for q in sorted(ans)})
        rows.append(row)

        debug = warped.copy()
        grades_row = {"file": os.path.basename(fname)}
        total_score = 0

        for q, (opt, pos, col) in results.items():
            x, y = pos
            if 'grid_bubble_params' in globals() and col < len(grid_bubble_params):
                radius = int(grid_bubble_params[col]['radius'] or 30)
            else:
                radius = 30
            grade_mark = '-'
            if correct_answers and q in correct_answers:
                correct = correct_answers[q]
                correct_opts = [c.strip().upper() for c in correct.replace(';', ',').split(',')]
                if not opt:
                    color = (128, 128, 128)
                    score = scoring.get('unanswered', 0)
                    grade_mark = 'nr'
                elif opt in correct_opts:
                    color = (0, 200, 0)
                    grade_mark = '+'
                    score = scoring.get('correct', 1)
                else:
                    color = (0, 0, 255)
                    grade_mark = '-'
                    score = scoring.get('incorrect', 0)
                total_score += score
                if opt:
                    overlay = debug.copy()
                    cv2.circle(overlay, (x, y), radius, color, -1)
                    cv2.addWeighted(overlay, 0.25, debug, 0.75, 0, debug)
                    cv2.circle(debug, (x, y), radius, color, 2)
            else:
                if opt:
                    overlay = debug.copy()
                    cv2.circle(overlay, (x, y), radius, (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.25, debug, 0.75, 0, debug)
                    cv2.circle(debug, (x, y), radius, (0, 0, 255), 2)
            grades_row[f"Q{q}"] = grade_mark

        if correct_answers:
            grades_row['grade'] = total_score
            grades_rows.append(grades_row)

        debug_name = os.path.join(detections_dir, f"{base}_detections.png")
        cv2.imwrite(debug_name, debug)

    # Save results CSV
    csv_path = os.path.join(output_dir, os.path.basename(out_csv))
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

    # Save grades CSV if applicable
    if grades_rows:
        all_qs = sorted({k for row in grades_rows for k in row if k.startswith('Q')}, key=lambda x: int(x[1:]))
        cols = ['file'] + all_qs + ['grade']
        grades_csv_path = os.path.join(output_dir, 'grades.csv')
        with open(grades_csv_path, 'w', newline='') as f:
            writer = csvmod.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for row in grades_rows:
                for q in all_qs:
                    row.setdefault(q, '-')
                writer.writerow(row)
        print(f"Saved grades to {grades_csv_path}")

    # Save handwriting info if enabled
    if hand_writing and info_rows:
        info_csv_path = os.path.join(students_info_dir, "info.csv")
        with open(info_csv_path, "w", newline="") as f:
            writer = csvmod.DictWriter(f, fieldnames=["image", "name", "id"])
            writer.writeheader()
            writer.writerows(info_rows)
        print(f"Saved handwriting info to {info_csv_path}")

    # Generate PDF if requested
    if get_info:
        try:
            from generate_students_info_pdf import generate_pdf
            output_pdf = os.path.join(output_dir, "image-to-names.pdf")
            generate_pdf(students_info_dir, output_pdf)
        except Exception as e:
            print(f"[WARN] Could not generate PDF: {e}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("input_folder", help="folder with scanned .png sheets")
    p.add_argument("--csv", default="results.csv")
    p.add_argument("--min-fill", type=int, default=200, help="Minimum fill threshold for answer detection (default: 200)")
    p.add_argument("--output", default="output", help="Output directory for results and detections (default: output)")
    p.add_argument("--answers-csv", help="CSV file with correct answers (Pregunta,Respuesta)")
    p.add_argument("--answers-json", help="JSON file with correct answers, e.g. {'1':'A','2':'A,D'}")
    p.add_argument("--scoring-json", help="JSON file with scoring for correct/incorrect/unanswered")
    p.add_argument("--get-info", action="store_true", help="Crop and save name/id fields and generate PDF (no OCR)")
    p.add_argument("--hand-writing", action="store_true", help="Enable handwriting OCR for name/id fields and generate info.csv")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for OCR model (cpu or cuda)")
    p.add_argument("--debug", type=int, default=1, choices=[0,1,2], help="Debug level: 0=none, 1=all except bubble fill, 2=all")
    p.add_argument("--image-to-name-csv", help="Path to a user-provided image-to-name CSV. If provided, it will be copied to the output directory as image-to-name.csv and the pipeline will still run.")
    args = p.parse_args()

    MIN_FILL = args.min_fill
    DEBUG = args.debug

    if not os.path.exists(args.input_folder):
        print(f"Input folder '{args.input_folder}' does not exist.")
        exit(1)

    # Check for grid configuration
    if not os.path.exists('grid_config.json'):
        imgs = glob.glob(os.path.join(args.input_folder, "*.png"))
        if not imgs:
            print('No sample PNG found in folder. Please provide an example scan.')
            sys.exit(1)
        sample = imgs[0]
        subprocess.call([
            'python3', 'grid_setup.py', sample,
            '--cols', input('Columns? '),
            '--rows', input('Rows per column? '),
            '--options', input('Options (comma-separated)? ')
        ])
        print('Configuration complete. Re-run script.')
        sys.exit(0)

    # Initialize grid
    init_grid()

    # If user provided an image-to-name CSV, copy it to output but still run the pipeline
    if args.image_to_name_csv:
        import shutil
        dest_csv = os.path.join(args.output, "image-to-name.csv")
        os.makedirs(args.output, exist_ok=True)
        shutil.copyfile(args.image_to_name_csv, dest_csv)
        print(f"Copied {args.image_to_name_csv} to {dest_csv}")

    # Run the main processing pipeline
    process_folder(
        args.input_folder,
        args.csv,
        args.output,
        args.answers_csv,
        args.answers_json,
        args.scoring_json,
        get_info=args.get_info,
        hand_writing=args.hand_writing,
        device=args.device
    )
