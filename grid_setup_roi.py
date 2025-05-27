#!/usr/bin/env python3
import cv2, numpy as np, json, argparse, sys

def auto_find_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(cv2.GaussianBlur(gray,(5,5),0),0,255,
                          cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sq=[]
    for c in cnts:
        a=cv2.contourArea(c)
        if not (2000<a<10000): continue
        peri=cv2.arcLength(c,True)
        approx=cv2.approxPolyDP(c,0.02*peri,True)
        if len(approx)==4:
            x,y,w,h=cv2.boundingRect(approx)
            if 0.8<=w/float(h)<=1.2:
                sq.append((x+w/2,y+h/2))
    if len(sq)!=4:
        raise RuntimeError(f"found {len(sq)} markers")
    sq=sorted(sq,key=lambda p:(p[1],p[0]))
    tl,tr,bl,br = sq[0],sq[1],sq[2],sq[3]
    if bl[0]>br[0]: bl,br=br,bl
    return np.array([tl,tr,br,bl],dtype='float32')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('image', help='sample scanned sheet')
    p.add_argument('--cols', type=int, required=True)
    p.add_argument('--rows', type=int, required=True)
    p.add_argument('--options', default='A,B,C,D')
    args=p.parse_args()

    img_full = cv2.imread(args.image)
    if img_full is None:
        print("Cannot open",args.image); sys.exit(1)

    # 1) auto‐warp via corner marks
    try:
        pts = auto_find_corners(img_full)
    except Exception as e:
        print("Auto corner detect failed:",e)
        sys.exit(1)

    (tl,tr,br,bl) = pts
    wA = np.linalg.norm(br-bl); wB=np.linalg.norm(tr-tl)
    hA = np.linalg.norm(tr-br); hB=np.linalg.norm(tl-bl)
    W,H = int(max(wA,wB)), int(max(hA,hB))
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]],dtype='float32')
    M = cv2.getPerspectiveTransform(pts,dst)
    warp = cv2.warpPerspective(img_full,M,(W,H))

    # 2) drag‐to‐select bubble area
    disp = warp.copy()
    win = 'Drag to select GRID, ENTER to confirm'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    r = cv2.selectROI(win, disp, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    x,y,w,h = [int(v) for v in r]
    if w==0 or h==0:
        print("No selection made"); sys.exit(1)

    # 3) normalize and save config
    x1n, y1n = x/W, y/H
    x2n, y2n = (x+w)/W, (y+h)/H
    colw = (x2n - x1n) / args.cols
    cfg = {
        'warp_w': W, 'warp_h': H,
        'cols': args.cols, 'rows': args.rows,
        'options': args.options.split(','),
        'x_offsets': [x1n + i*colw for i in range(args.cols)],
        'y_start': y1n,
        'y_step': (y2n-y1n)/(args.rows-1),
        'col_width': colw
    }
    with open('grid_config.json','w') as f:
        json.dump(cfg, f, indent=2)
    print("Saved grid_config.json:")
    print(json.dumps(cfg,indent=2))