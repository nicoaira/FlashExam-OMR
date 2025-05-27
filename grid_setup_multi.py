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

    # 2) bubble calibration first: determine pixel radius and spacing (with retry)
    MAX_W, MAX_H = 1200, 800
    cal_scale = min(1.0, MAX_W/warp.shape[1], MAX_H/warp.shape[0])
    warp_disp = cv2.resize(warp, (int(warp.shape[1]*cal_scale), int(warp.shape[0]*cal_scale)))
    spacing_px = None; radius_px = None
    while True:
        cal_pts = []
        def on_click_cal(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(cal_pts) < 4:
                ox, oy = int(x / cal_scale), int(y / cal_scale)
                cal_pts.append((ox, oy))
                cv2.circle(warp_disp, (x, y), 2, (0,255,255), -1)
                cv2.imshow('Calibrate bubbles', warp_disp)
        cv2.namedWindow('Calibrate bubbles', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Calibrate bubbles', on_click_cal)
        cv2.imshow('Calibrate bubbles', warp_disp)
        print('Calibration: click centers of A1 then B1, then left rim and right rim.')
        cv2.waitKey(0)
        cv2.destroyWindow('Calibrate bubbles')
        if len(cal_pts) != 4:
            retry = input(f'Got {len(cal_pts)} points, need 4. Retry calibration? (y/n): ')
            if retry.lower().startswith('y'):
                warp_disp = cv2.resize(warp, (int(warp.shape[1]*cal_scale), int(warp.shape[0]*cal_scale)))
                continue
            else:
                print('Skipping calibration.')
                break
        pA, pB, rimL, rimR = cal_pts
        spacing_px = abs(pB[0] - pA[0])
        radius_px = abs(rimR[0] - rimL[0]) // 2
        print(f'Calibration result: spacing={spacing_px}px, radius={radius_px}px')
        choice = input('Accept calibration? (y = yes, r = retry, c = cancel): ')
        if choice.lower().startswith('y'):
            break
        elif choice.lower().startswith('r'):
            warp_disp = cv2.resize(warp, (int(warp.shape[1]*cal_scale), int(warp.shape[0]*cal_scale)))
            continue
        else:
            print('Calibration canceled.')
            spacing_px = None; radius_px = None
            break
    # prepare config with calibration
    config = {'warp_w': W, 'warp_h': H,
              'columns': args.columns, 'rows': args.rows,
              'options': opts,
              'bubble_spacing_px': spacing_px,
              'bubble_radius_px': radius_px,
              'grids': []}
    # scale display for grid ROI selection
    disp = cv2.resize(warp, (int(warp.shape[1]*cal_scale), int(warp.shape[0]*cal_scale)))
    window = 'Grid setup: g=add grid, q=finish'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print("Keys: 'g' = add grid, 'q' = finish.")
    while True:
        cv2.imshow(window, disp)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('g'):
            # custom ROI drawing with real-time grid preview
            state = {'drawing': False, 'rect': None}
            # mouse callback to update selection
            def mouse_cb(event, mx, my, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    state['drawing'] = True
                    state['rect'] = None
                    state['ix'], state['iy'] = mx, my
                elif event == cv2.EVENT_MOUSEMOVE and state['drawing']:
                    disp_temp = disp.copy()
                    x0, y0 = state['ix'], state['iy']
                    x1, y1 = mx, my
                    cv2.rectangle(disp_temp, (x0,y0), (x1,y1), (0,255,0), 1)
                    # draw preview guiding lines
                    x, y, w_box, h_box = min(x0,x1), min(y0,y1), abs(x1-x0), abs(y1-y0)
                    # boundaries
                    cv2.line(disp_temp, (x,y), (x+w_box,y), (255,0,0),1)
                    cv2.line(disp_temp, (x,y+h_box), (x+w_box,y+h_box), (255,0,0),1)
                    cv2.line(disp_temp, (x,y), (x,y+h_box), (255,0,0),1)
                    cv2.line(disp_temp, (x+w_box,y), (x+w_box,y+h_box), (255,0,0),1)
                    # vertical center-lines
                    for j in range(len(opts)):
                        cx = int(x + (j+0.5)*(w_box/len(opts)))
                        cv2.line(disp_temp, (cx,y), (cx,y+h_box), (255,0,0),1)
                    cv2.imshow(window, disp_temp)
                elif event == cv2.EVENT_LBUTTONUP and state['drawing']:
                    state['drawing'] = False
                    x0, y0 = state['ix'], state['iy']
                    x1, y1 = mx, my
                    state['rect'] = (min(x0,x1), min(y0,y1), abs(x1-x0), abs(y1-y0))
            # set callback
            cv2.setMouseCallback(window, mouse_cb)
            # wait for user to finish drawing (rect set on mouse up)
            while state['rect'] is None:
                cv2.waitKey(50)
            cv2.setMouseCallback(window, lambda *args: None)
            x, y, w, h = state['rect']
            if w == 0 or h == 0:
                print("Empty ROI, try again.")
                continue
            # map to full-resolution coords using cal_scale
            xf, yf = int(x/cal_scale), int(y/cal_scale)
            wf, hf = int(w/cal_scale), int(h/cal_scale)
            # final preview on warp and scale back
            overlay = warp.copy()
            # draw boundaries and centers as above
            cv2.line(overlay, (xf,yf),(xf+wf,yf),(255,0,0),1)
            cv2.line(overlay, (xf,yf+hf),(xf+wf,yf+hf),(255,0,0),1)
            cv2.line(overlay, (xf,yf),(xf,yf+hf),(255,0,0),1)
            cv2.line(overlay, (xf+wf,yf),(xf+wf,yf+hf),(255,0,0),1)
            for j in range(len(opts)):
                cx = int(xf + (j+0.5)*(wf/len(opts)))
                cv2.line(overlay, (cx,yf),(cx,yf+hf),(255,0,0),1)
            prev = cv2.resize(overlay, (disp.shape[1], disp.shape[0]))
            cv2.imshow(window, prev)
            print("Press 'y' to accept grid, any other key to discard.")
            k2 = cv2.waitKey(0) & 0xFF
            if k2 == ord('y'):
                config['grids'].append({'x': xf/W, 'y': yf/H, 'w': wf/W, 'h': hf/H})
                cv2.rectangle(disp, (x,y),(x+w,y+h),(0,255,0),2)
                print(f"Grid #{len(config['grids'])} accepted.")
            else:
                print("Grid discarded.")
        elif key == ord('q'):
            break
    cv2.destroyAllWindows()

    # 3) save final config after grids
    with open('grid_config.json','w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved grid_config.json with {len(config['grids'])} grids and preset bubble calibration.")