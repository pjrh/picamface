#!/usr/bin/env python

#import numpy as np
import cv2
#import math
import picamera
import picamera.array

# local modules
#from video import create_capture
from common import clock, draw_str

help_message = '''
USAGE: facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
'''

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

if __name__ == '__main__':
    import sys, getopt
    print help_message

    overlay = cv2.imread("./andreaw.png")

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    cascade_fn = args.get('--cascade', "./data/haarcascades/haarcascade_frontalface_alt.xml")
    #nested_fn  = args.get('--nested-cascade', "./data/haarcascades/haarcascade_eye.xml")

    cascade = cv2.CascadeClassifier(cascade_fn)
    #nested = cv2.CascadeClassifier(nested_fn)

    #cam = create_capture(video_src, fallback='synth:bg=../cpp/lena.jpg:noise=0.05')
    picam = picamera.PiCamera()
    picam.resolution = (320,240)
    stream = picamera.array.PiRGBArray(picam)

    while True:
        stream.truncate(0)
        #ret, img = cam.read()
        picam.capture(stream, format='bgr')
        img = stream.array

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        t = clock()
        rects = detect(gray, cascade)
        vis = img.copy()
        #draw_rects(vis, rects, (0, 255, 0))
        for x1, y1, x2, y2 in rects:

            #Scale and offset the overlay
            scalef=0.25

            yoffset=0
            xoffset=0

            x1_big=int(x1-scalef*(x2-x1))-xoffset
            x2_big=int(x2+scalef*(x2-x1))-xoffset
            y1_big=int(y1-scalef*(y2-y1))-yoffset
            y2_big=int(y2+scalef*(y2-y1))-yoffset

            overlay_resized = cv2.resize(overlay, (x2_big-x1_big,y2_big-y1_big))

            #Fix the rectangle edges if they go off the screen
            ymax, xmax = img.shape[:2]
            if x1_big<0:
                x1_bigf=0
            else:
                x1_bigf=x1_big

            if y1_big<0:
                y1_bigf=0
            else:
                y1_bigf=y1_big

            if x2_big>xmax:
                x2_bigf=xmax
            else:
                x2_bigf=x2_big

            if y2_big>ymax:
                y2_bigf=ymax
            else:
                y2_bigf=y2_big

            xmaxoverlay, ymaxoverlay = overlay_resized.shape[:2]
            overlayxmin=0+(x1_bigf-x1_big)
            #overlayxmax=xmaxoverlay-(x2_big-x2_bigf)
            overlayymin=0+(y1_bigf-y1_big)
            #overlayymax=ymaxoverlay-(y2_big-y2_bigf)

            #Extract the relevant bits of the image
            roi = gray[y1_bigf:y2_bigf, x1_bigf:x2_bigf]
            vis_roi = vis[y1_bigf:y2_bigf, x1_bigf:x2_bigf]

            yroi, xroi=vis_roi.shape[:2]
            overlay_resized_roi=overlay_resized[overlayymin:(overlayymin+yroi) , overlayxmin:(overlayxmin+xroi)]

            # Now create a mask of logo and create its inverse mask also
            #img2gray = cv2.cvtColor(overlay_resized,cv2.COLOR_BGR2GRAY)
            #ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            #mask_inv = cv2.bitwise_not(mask)

            # Now black-out the area of logo in ROI
            #img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

            # Take only region of logo from logo image.
            #img2_fg = cv2.bitwise_and(overlay_resized,overlay_resized,mask = mask)

            # Put logo in ROI and modify the main image
            dst = cv2.add(vis_roi,overlay_resized_roi)

            vis[y1_bigf:y2_bigf, x1_bigf:x2_bigf] = dst

            #subrects = detect(roi.copy(), nested)
            #draw_rects(vis_roi, subrects, (255, 0, 0))
        dt = clock() - t

        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
        cv2.imshow('facedetect', vis)

        if 0xFF & cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows()
