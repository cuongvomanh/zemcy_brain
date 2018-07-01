# Python 2/3 compatibility
from __future__ import print_function
import sys
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import numpy as np
PY3 = sys.version_info[0] == 3
STANDARD_AREA = 32*32
DEBUG = True
if PY3:
    xrange = range

import zemcy
import video

def draw_rectangles(img, rects, color, thinkness):
    image = img.copy()
    for rect in rects:
        bx, by, bw, bh = rect
        cv.rectangle(image, (bx, by), (bx+bw, by+bh), color, thinkness)
    return image
class App(object):
    def __init__(self, video_src):
        self.cam = video.create_capture(video_src, None)
        _ret, self.origin_frame = self.cam.read()
        self.move_degree_frame = self.origin_frame.copy()
        cv.namedWindow('display_frame')
        cv.setMouseCallback('display_frame', self.onmouse)
        cv.setMouseCallback('display_frame', self.onmouse)
        cv.createTrackbar('lo', 'display_frame', 20, 255, self.update_track_window)
        cv.createTrackbar('hi', 'display_frame', 20, 255, self.update_track_window)
        # self.track_window = (267, 148, 137, 332)
        w, h = self._resolution = self.origin_frame.shape[1::-1]
        if DEBUG:
            print('type(w), (h) = ', type(w), (h))
        self.track_window = (w//2 -w//10, h//2 - h//10, w//5, h//5)
        if DEBUG:
            print('self.track_window = ', self.track_window)
        self.center = (w//2, h//2)
        self.TERM_CRIT = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
        hist = np.zeros(16, dtype ='float32')
        hist[0] = 255
        self.ROI_HIST = hist
        self.track_box = None
        self.drag_start = None
        self.floodFill_mask = np.zeros((h+2, w+2), np.uint8)
        self.recently_not_none_track_box = None
        self.focus_window = None
        self.display_frame = self.origin_frame
    def onmouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
        if self.drag_start:
            xmin = min(x, self.drag_start[0])
            ymin = min(y, self.drag_start[1])
            xmax = max(x, self.drag_start[0])
            ymax = max(y, self.drag_start[1])
        if event == cv.EVENT_LBUTTONUP:
            self.drag_start = None
            self.track_window = (xmin, ymin, xmax - xmin, ymax - ymin)
        
    def draw_hsv(self,flow, seed_pt=None):
        h, w = flow.shape[:2]
        fx, fy = flow[:,:,0], flow[:,:,1]
        v = np.sqrt(fx*fx+fy*fy)
        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[...,0] = 180
        hsv[...,1] = 255
        hsv[...,2] = np.minimum(v*4, 255)
        return hsv
    def get_moving_object(self):
        unit = zemcy.box_to_box(self.track_box, self.origin_frame.shape[:2])
        (_center_x,_center_y), (w,h) = unit
        if w > 2 and h > 2:
            cut_img = zemcy.cut(self.origin_frame, unit)

            resized_cut_img = zemcy.resize_img(cut_img, STANDARD_AREA)
            return resized_cut_img
        return None
    def is_something_move(self):
        return self.track_box != ((0.0, 0.0), (0.0, 0.0), 0.0)
    def get_focus_img(self):
        if self.focus_window is None:
            return None
        rect = _x, _y, w, h = self.focus_window
        if w > 2 and h > 2:
            cut_img = zemcy.cut(self.origin_frame, zemcy.rect_to_unit(rect))
            resized_cut_img = zemcy.resize_img(cut_img, STANDARD_AREA)
            return resized_cut_img
        return None
    def update_track_window(self, dummy=None):
        dst = cv.calcBackProject([cv.cvtColor(self.move_degree_frame, cv.COLOR_HSV2BGR) ],[0],self.ROI_HIST,[0,180],1)
        # apply meanshift to get the new location
        cam_shift_mask = cv.inRange(self.move_degree_frame, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        dst &= cam_shift_mask
        self.track_box, self.track_window = cv.CamShift(dst, self.track_window, self.TERM_CRIT)
        if self.is_something_move():
            self.recently_not_none_track_box = self.track_box
        if self.recently_not_none_track_box:
            if DEBUG:
                print('Everything is not moving now!')
            (center_x, center_y), (_,_), _ = self.recently_not_none_track_box
            seed_pt = int(center_x), int(center_y)
            if DEBUG:
                print('center_x, center_y = ',center_x, center_y)
            lo = cv.getTrackbarPos('lo', 'display_frame')
            hi = cv.getTrackbarPos('hi', 'display_frame')
            self.floodFill_mask[:] = 0

            cv.floodFill(self.display_frame, self.floodFill_mask, seed_pt, (255, 255, 255), (lo,)*3, (hi,)*3, cv.FLOODFILL_FIXED_RANGE)
            none_margins_mask = self.floodFill_mask[1:-1, 1:-1]
            pixelpoints = cv.findNonZero(none_margins_mask)
            rect_x,rect_y,rect_w,rect_h = cv.boundingRect(pixelpoints)
            self.focus_window = rect_x, rect_y, rect_w,rect_h
            cv.rectangle(self.display_frame,(rect_x,rect_y),(rect_x+rect_w,rect_y+rect_h),(255,0,0),3)
        if DEBUG:
            print('track_box')
            print(self.track_box)
            print('track_window')
            print(self.track_window)
            pts = cv.boxPoints(self.track_box)
            pts = np.int0(pts)
            track_box_img = self.origin_frame.copy()
            cv.polylines(track_box_img ,[pts],True, 255,2)
            cv.imshow('track_box_img', track_box_img)
    def run(self):
        _ret, self.origin_frame = self.cam.read()
        prevgray = cv.cvtColor(self.origin_frame, cv.COLOR_BGR2GRAY)

        while True:
            _ret, self.origin_frame = self.cam.read()
            gray = cv.cvtColor(self.origin_frame, cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            prevgray = gray

            self.move_degree_frame = self.draw_hsv(flow, None)
            move_degree_bgr_frame = cv.cvtColor(self.move_degree_frame, cv.COLOR_HSV2BGR)
            if DEBUG:
                cv.imshow('move_degree_bgr_frame', move_degree_bgr_frame)
            move_degree_mask = cv.inRange(self.move_degree_frame, np.array((180., 255., 20.)), np.array((180., 255., 255.)))
            # 
            
            self.display_frame = self.origin_frame.copy()          
            self.update_track_window()
            # 
            if DEBUG:
                moving_object_img = self.get_moving_object()
                if moving_object_img is not None:
                    cv.imshow('moving_object', moving_object_img)
            focus_img = self.get_focus_img()
            if focus_img is not None:
                cv.imshow('focus_img',focus_img)
            self.display_frame[move_degree_mask != 0] = move_degree_bgr_frame[move_degree_mask != 0]
            self.display_frame = draw_rectangles(self.display_frame, [self.track_window], (0, 255, 0), 1)
            cv.ellipse(self.display_frame, self.track_box, (0, 0, 255), 1)
            cv.imshow('display_frame', self.display_frame)
            if DEBUG:
                cv.imshow('origin_frame', self.origin_frame)

            ch = cv.waitKey(5)
            if ch == 27:
                break
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

