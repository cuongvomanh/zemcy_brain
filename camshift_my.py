#!/usr/bin/env python

'''
Camshift tracker
================

This is a demo that shows mean-shift based tracking
You select a color objects such as your face and it tracks it.
This reads from video camera (0 by default, or the camera number the user enters)

http://www.robinhewitt.com/research/track/camshift.html

Usage:
------
    camshift.py [<video source>]

    To initialize tracking, select the object with mouse

Keys:
-----
    ESC   - exit
    b     - toggle back-projected probability visualization
'''

# Python 2/3 compatibility
from __future__ import print_function
import sys
import zemcy
from matplotlib import pyplot as plt
PY3 = sys.version_info[0] == 3
STANDARD_AREA = 32*32
if PY3:
    xrange = range

import numpy as np
import cv2 as cv

# local module
import video
from video import presets


class App(object):
    def __init__(self, video_src):
        self.cam = video.create_capture(video_src, presets['cube'])
        _ret, self.origin_frame = self.cam.read()
        self.frame = self.origin_frame.copy()
        cv.namedWindow('camshift')
        cv.setMouseCallback('camshift', self.onmouse)

        self.selection = None
        self.drag_start = None
        self.show_backproj = False
        self.track_window = None
        self.hist = None

    def onmouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.track_window = None
        if self.drag_start:
            xmin = min(x, self.drag_start[0])
            ymin = min(y, self.drag_start[1])
            xmax = max(x, self.drag_start[0])
            ymax = max(y, self.drag_start[1])
            self.selection = (xmin, ymin, xmax, ymax)
        if event == cv.EVENT_LBUTTONUP:
            self.drag_start = None
            self.track_window = (xmin, ymin, xmax - xmin, ymax - ymin)

    def show_hist(self):
        bin_count = self.hist.shape[0]
        bin_w = 24
        img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
        for i in xrange(bin_count):
            h = int(self.hist[i])
            cv.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
        img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
        cv.imshow('hist', img)
    def draw_hsv(self,flow, seed_pt):
        h, w = flow.shape[:2]
        fx, fy = flow[:,:,0], flow[:,:,1]
        ang = np.arctan2(fy, fx) + np.pi
        v = np.sqrt(fx*fx+fy*fy)
        
        hsv = np.zeros((h, w, 3), np.uint8)
        # hsv[...,0] = ang*(180/np.pi/2)
        hsv[...,0] = 180
        hsv[...,1] = 255
        hsv[...,2] = np.minimum(v*4, 255)
        flags = cv.FLOODFILL_FIXED_RANGE
        
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        # lo = 20
        # hi = 20
        # cv.floodFill(bgr, None, seed_pt, (255, 0, 0), (lo,)*3, (hi,)*3, flags)
        
        # cv.imshow('hsvvvvv', bgr)
        return bgr
    def get_moving_object(self, track_box):
        unit = zemcy.box_to_box(track_box, self.origin_frame.shape[:2])
        (x,y), (w,h) = unit
        if w > 2 and h > 2:
            cut_img = zemcy.cut(self.origin_frame, unit)
            resized_cut_img = zemcy.resize_img(cut_img, STANDARD_AREA)
            return resized_cut_img
        return None
            
    def run(self):
        ret, self.origin_frame = self.cam.read()
        prevgray = cv.cvtColor(self.origin_frame, cv.COLOR_BGR2GRAY)
        track_box = None
        while True:
            _ret, self.origin_frame = self.cam.read()
            # _ret, self.frame = self.cam.read()
            gray = cv.cvtColor(self.origin_frame, cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            prevgray = gray

            # cv.imshow('flow', draw_flow(gray, flow))

            

            vis = self.frame.copy()
            hsv = cv.cvtColor(self.frame, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
            if track_box:
                # (xmin, ymin, xmax, ymax) = self.selection
                (x, y), (w,h), angle = track_box
                (x, y), (w,h) = zemcy.box_to_box(track_box, self.origin_frame.shape[:2])
                # hsv_img = cv.cvtColor(self.frame, cv.COLOR_BGR2HSV)
                # ok = False
                # for xx in xrange(int(x-w/2),int(x+w/2)):
                #     for yy in xrange(int(y-h/2),int(y+h/2)):
                #         if hsv_img[yy][xx][2] > 150:
                #             ok = True
                #             xxx, yyy = xx, yy
                #             break
                #     if ok == True:
                #         break
                # seed_pt = (xxx,yyy)
                seed_pt = x, y

            else:
                seed_pt = (320,240)
            self.frame = self.draw_hsv(flow, seed_pt)
            if self.selection:
                x0, y0, x1, y1 = self.selection
                hsv_roi = hsv[y0:y1, x0:x1]
                mask_roi = mask[y0:y1, x0:x1]
                hist = cv.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
                cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)
                self.hist = hist.reshape(-1)
                
                self.show_hist()

                vis_roi = vis[y0:y1, x0:x1]
                cv.bitwise_not(vis_roi, vis_roi)
                vis[mask == 0] = 0

            if self.hist is None or (self.track_window and self.track_window[2] > 0 and self.track_window[3] > 0):
                self.selection = None
                if self.hist is None:
                    self.hist = np.zeros(16)
                    self.hist[0] = 255
                    self.track_window = (267, 148, 137, 332)
                print('hist')
                print(self.hist)
                prob = cv.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
                prob &= mask
                term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
                track_box, self.track_window = cv.CamShift(prob, self.track_window, term_crit)

                if self.show_backproj:
                    vis[:] = prob[...,np.newaxis]
                try:
                    # print(self.track_window)
                    # print(track_box)
                    cv.ellipse(vis, track_box, (0, 0, 255), 2)
                    object_img = self.get_moving_object(track_box)
                    
                    if object_img is not None:
                        # cv.imshow('track box', object_img)
                        plt.title('Track box')
                        # plt.ion()
                        cv.imshow('cuted img', object_img)
                        cv.waitKey(0)
                        plt.imshow(object_img)
                        plt.pause(0.001)
                        # plt.show(0)
                        # plt.show()
                except:
                    pass
                    # print(track_box)

            cv.imshow('camshift', vis)

            ch = cv.waitKey(5)
            if ch == 27:
                break
            if ch == ord('b'):
                self.show_backproj = not self.show_backproj
        cv.destroyAllWindows()
        plt.show()
        


if __name__ == '__main__':
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0
    print(__doc__)
    App(video_src).run()
