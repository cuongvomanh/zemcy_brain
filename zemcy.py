# Python 2/3 compatibility
from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3
DEBUG = False
TEST = True
STANDARD_SIZE = '480x234'
STANDARD_AREA = 480*234
if PY3:
    xrange = range

import numpy as np
import cv2 as cv
import video
import math
# from common import nothing, getsize
sys.path.append('/home/cuong/VNG/National_Identification_Card_Reader/src')
import support_lib
def cut(frame, unit):
    (center_x,center_y), (w,h) = unit
    return frame.copy()[center_y - h//2: center_y + h//2, center_x - w//2: center_x + w//2]
def window_to_unit(window):
    topleft_x, topleft_y, w, h = window
    unit = (topleft_x + int(w/2), topleft_y + int(h/2)), (w, h)
    return unit
def unit_to_rect(unit):
    (center_x, center_y), (w, h) = unit
    topleft_x, topleft_y =  center_x - w//2, center_y - h//2
    rect = topleft_x, topleft_y, w, h
    return rect
def resize_img(img, cvt_area):
    width, height = _img_revolution = img.shape[1::-1]
    area = width*height
    ratio = math.sqrt(1.0*cvt_area/area)
    cvt_width = int(width*ratio)
    cvt_height = int(height*ratio)
    resize_shape = (cvt_width, cvt_height)
    resized_img = cv.resize(img, resize_shape)
    return resized_img
def box_to_box(box, shape):
    height, width = shape
    pts = cv.boxPoints(box)
    pts = np.int0(pts)
    topleft_x, topleft_y, w, h = cv.boundingRect(pts)
    if DEBUG:
        print('topleft_x, topleft_y, w, h = ', topleft_x, topleft_y, w, h)
    if topleft_x < 0:
        topleft_x = 0
        delta_x = abs(topleft_x)
        w -= delta_x
    if topleft_x + w > width:
        delta_x = topleft_x + w - width
        w -= delta_x
    if topleft_y + h > height:
        delta_y = topleft_y + h - height
        h -= delta_y
    if topleft_y < 0:
        topleft_y = 0
        delta_y = abs(topleft_y)
        h -= delta_y
    center_x, center_y, w, h = topleft_x + w//2, topleft_y + h//2, w, h
    unit = (center_x,center_y), (w,h)
    return unit
def floodFill_return_window(img, floodFill_mask, seed_pt, low, high, color = (255, 255, 255), flags = cv.FLOODFILL_FIXED_RANGE):
    cv.floodFill(img, floodFill_mask, seed_pt, (255, 255, 255), (low,)*3, (high,)*3, flags)
    # none_margins_mask = floodFill_mask[1:-1, 1:-1]
    # pixelpoints = cv.findNonZero(none_margins_mask)
    # window = support_lib.points_to_window(pixelpoints)
    mask = cv.inRange(img, np.array([255,255,255]), np.array([255,255,255]))
    pixelpoints = cv.findNonZero(mask)
    window = support_lib.points_to_window(pixelpoints)
    return window
if __name__ == '__main__':
    import sys
    # print(__doc__)

    try:
        fn = sys.argv[1]
    except:
        fn = 0
    cap = video.create_capture(fn)

    # leveln = 6
    # cv.namedWindow('level control')
    # for i in xrange(leveln):
        # cv.createTrackbar('%d'%i, 'level control', 5, 50, nothing)
    ret, frame = cap.read()
    if DEBUG:
        print('frame shape = ', frame.shape)
    while True:
        ret, frame = cap.read()
        cv.imshow('frame', frame)
        frame_resolution = tuple(frame.shape[1::-1])
        w_frame, h_frame = frame_resolution
        x, y, w, h = w_frame//2, h_frame//2, w_frame//2, h_frame//2 
        unit = ((x,y), (w,h))
        if DEBUG and TEST:
            print(unit)
        cut_img = cut(frame, unit)
        resized_cut_img = resize_img(cut_img, STANDARD_AREA)
        cv.imshow('resized_cut_img', resized_cut_img)
        if cv.waitKey(1) == 27:
            break
        TEST = False