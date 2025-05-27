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

    # 2) interactive grid selection via keyboard
    config = {'warp_w': W, 'warp_h': H,
              'columns': args.columns, 'rows': args.rows,
              'options': opts, 'grids': []}
    # scale display
    MAX_W, MAX_H = 1200, 800
    scale = min(1.0, MAX_W/warp.shape[1], MAX_H/warp.shape[0])
    disp = cv2.resize(warp, (int(warp.shape[1]*scale), int(warp.shape[0]*scale)))
    window = 'Grid setup: g=add grid, q=finish'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print("Keys: 'g' = add grid, 'q' = finish.")
    while True:
        cv2.imshow(window, disp)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('g'):
            # draw ROI
            x, y, w, h = cv2.selectROI(disp, False, False)
            if w == 0 or h == 0:
                print("Empty ROI, try again.")
                continue
            xf, yf = int(x/scale), int(y/scale)
            wf, hf = int(w/scale), int(h/scale)
            # preview subdivisions
            overlay = warp.copy()
            for i in range(args.rows+1):
                yy = int(yf + i*(hf/args.rows))
                cv2.line(overlay, (xf,yy), (xf+wf,yy), (255,0,0), 1)
            for j in range(len(opts)+1):
                xx = int(xf + j*(wf/len(opts)))
                cv2.line(overlay, (xx,yf), (xx,yf+hf), (255,0,0), 1)
            disp_prev = cv2.resize(overlay, (disp.shape[1], disp.shape[0]))
            cv2.imshow(window, disp_prev)
            print("Press 'y' to accept grid, any other key to discard.")
            k2 = cv2.waitKey(0) & 0xFF
            if k2 == ord('y'):
                config['grids'].append({'x': xf/W, 'y': yf/H, 'w': wf/W, 'h': hf/H})
                cv2.rectangle(disp, (x,y), (x+w,y+h), (0,255,0), 2)
                print(f"Grid #{len(config['grids'])} accepted.")
            else:
                print("Grid discarded.")
        elif key == ord('q'):
            break
    cv2.destroyAllWindows()

    # save
    with open('grid_config.json','w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved grid_config.json with {len(config['grids'])} grids.")