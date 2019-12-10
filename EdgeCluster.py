'''
 '
 ' @file  EdgeCluster.py
 '
 ' @brief  Edge clustering algorithm
 '
 ' @version  1.0
 '
 ' @date  2019/12/10 20:53:59
 '
 ' @author  Red, 735467224@qq.com
 '
'''

import cv2
import time
import math
import random
import numpy as np

'''
 ' @brief Edge clustering algorithm
 ' there were any indoor scenes that contained irregular objects different from others such as lines and color-rectangles in the floor.
 ' it will find these objects and clustering them.
'''
class EdgeCluster:
    def __init__(self, source):
        '''
        brief: __init__
        :param source: input source, video(source=0) or avi(source='../source.avi')
        '''
        self.source = source
        self.w = 0                  # width of source video/avi
        self.h = 0                  # height of source video/avi
        self.thresElapsed = 0.08    # running elapsed of per frame, if exceed then truncature

        self.idx_block = 0          # current index of block, if truncature it will running from the current index block at the next frame
        self.hisrects = []          # detected rects on the pass
        self.hiscount = []          # times of lose focus for the detected rects on the pass
        self.retainfm = 10          # threshold times of lose focus for a rect, if exceed then destroy

        self.thresc = 150           # lower threshold of Canny at the current frame
        self.secth = 30             # interval of Canny thresholds
        self.threscmin = 30         # lower threshold of Canny at the automation loop

        self.bs = 60                # block size
        self.rs = 20                # resize block size

        self.threslp = 15           # threshold of points' number (in the resize block size)
        self.thresacc = 12          # threshold of accuracy for hough lines
        self.thresln = 0.2          # threshold of percentages for hough lines to points
        self.thresov = 0.2          # threshold of overlap between rects
        self.thresdt = 0.1          # threshold of percentage for cluster area to image size

    def run(self):
        '''
        brief: run the clusterings for source video/avi
        :return: none
        '''
        cap = cv2.VideoCapture(self.source)
        ret, frame = cap.read()
        (self.h, self.w, _) = frame.shape

        while (True):
            ret, frame = cap.read()

            if frame is None:
                break

            start = time.clock()

            isInterrupt, rects = self.makePreRects(frame)

            mergerects = self.makeMergeRects(rects)

            ret = self.makeHistoryRects(mergerects)

            back, area = self.makeClusterBack(isInterrupt)


            frame = frame + back * 192

            self.thresc = max((self.thresc+1) if area > self.thresdt else (self.thresc-1), self.threscmin)

            stop = time.clock()


            fps = int(1.0 / (stop - start))

            cv2.putText(frame, str(fps) + ' fps', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow('EdgeCluster', frame)

            if (cv2.waitKey(1) >= 0):
                break

    def makePreRects(self, frame):
        '''
        brief: make any preselection rects
        :param frame: input frame
        :return: isInterrupt, the running elapsed of per frame is exceed or not
                rects, the preselection rects
        '''
        ratio = self.bs / self.rs
        blockx, blocky = int(self.w/self.bs), int(self.h/self.bs)
        totalnum = blockx * blocky

        dst = cv2.Canny(frame, self.thresc, self.thresc + self.secth)

        isInterrupt, rects, start  = False, [], time.clock()
        for idxb in range(self.idx_block, self.idx_block + totalnum):
            idxb = idxb % totalnum

            x = idxb % blockx * self.bs + int(random.randint(0,1) * self.bs / 2)
            y = int(idxb / blockx) * self.bs + int(random.randint(0,1) * self.bs / 2)
            xs, ys = x + self.bs, y + self.bs

            if xs >= self.w or ys >= self.h:
                continue

            block = cv2.resize(dst[y:ys, x:xs], (self.rs, self.rs))
            points = np.argwhere(block > 8)

            lenp = len(points)
            if lenp < self.threslp:
                continue

            hough = np.zeros(self.thresacc, dtype=np.int)
            for i in range(lenp - 1):
                for j in range(i, lenp):
                    p1, p2 = points[i], points[j]
                    radian = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
                    deg = (int)((radian + np.pi) * self.thresacc / np.pi) % self.thresacc
                    hough[deg] += 1

            line_percent = 1.0 * hough.max() / hough.sum()
            if line_percent > self.thresln:
                continue

            maxidx = np.mean(points, axis=0)
            center = (int(maxidx[0] * ratio + x), int(maxidx[1] * ratio + y))
            rect = (center[0] - self.bs / 2, center[1] - self.bs / 2, self.bs, self.bs)
            rects.append(rect)

            if (time.clock() - start) > self.thresElapsed:
                isInterrupt = True
                self.idx_block = idxb
                break

        return (isInterrupt, rects)

    def makeMergeRects(self, rects):
        '''
        brief: merge preselection rects
        :param rects: the preselection rects
        :return: mergerects, merged rects
        '''
        mergerects = []
        for i in range(len(rects)):
            ismerge = False
            for j in range(len(mergerects)):
                ov = EdgeCluster.overlap(rects[i], mergerects[j])
                if ov > self.thresov:
                    ismerge = True
                    (xj, yj, wj, hj) = mergerects[j]
                    (xi, yi, wi, hi) = rects[i]
                    mergerects[j] = (0.5*(xj+xi), 0.5*(yj+yi), 0.5*(wj+wi), 0.5*(hj+hi))
                    break
            if not ismerge:
                mergerects.append(rects[i])
        return mergerects

    def makeHistoryRects(self, mergerects):
        '''
        brief: merge current rects and history rects
        :param mergerects: merged rects in the current frame
        :return: 0
        '''
        for rect in mergerects:
            ismerge = False
            for i in range(len(self.hisrects)):
                ov = EdgeCluster.overlap(rect, self.hisrects[i])
                if ov>self.thresov:
                    ismerge = True
                    (xj, yj, wj, hj) = self.hisrects[i]
                    (xi, yi, wi, hi) = rect
                    self.hisrects[i] = (0.5*(xj+xi), 0.5*(yj+yi), 0.5*(wj+wi), 0.5*(hj+hi))
                    self.hiscount[i] = 0
                    break
            if not ismerge:
                self.hisrects.append(rect)
                self.hiscount.append(0)

        for i in range(len(self.hisrects)-1,0,-1):
            self.hiscount[i] += 1
            if(self.hiscount[i] > self.retainfm):
                del self.hiscount[i]
                del self.hisrects[i]
        return 0

    def makeClusterBack(self, isInterrupt):
        '''
        brief: make the clustering result
        :param isInterrupt: the running elapsed of per frame is exceed or not
        :return: back, the clustering mask
                area, the clustering area percentage
        '''
        back, circles = np.zeros((self.h, self.w), dtype=np.uint8), []
        for rect in self.hisrects:
            ismerge = False
            (xi, yi, wi, hi) = rect
            centeri, radiusi = (int(xi+wi/2), int(yi+hi/2)), int((wi+hi)/4)
            for j in range(len(circles)):
                centerj, radiusj = circles[j]
                disij = math.sqrt((centeri[0]-centerj[0])**2 + (centeri[1]-centerj[1])**2)
                thresr = int((radiusi + radiusj) * 1.25)
                if disij < thresr:
                    ismerge = True
                    cv2.line(back, centeri, centerj, 255, thresr)
            if not ismerge:
                circles.append([centeri, radiusi])

        back, backsize = cv2.resize(back, (int(self.w/2), int(self.h/2))), self.w * self.h / 4
        area = 1.0 if isInterrupt else (len(np.argwhere(back > 8)) / backsize)
        back = cv2.resize(cv2.Sobel(back, cv2.CV_8UC1, 1, 1), (self.w, self.h))
        return (cv2.merge([back,back,back]), area)

    @staticmethod
    def merge(rect1, rect2):
        '''
        brief: merge two rects
        :param rect1, rect2
        :return: merged rect
        '''
        (x1, y1, w1, h1) = rect1
        (x2, y2, w2, h2) = rect2
        x3, y3 = min(x1, x2), min(y1, y2)
        r, b = max(x1 + w1, x2 + w2), max(y1 + h1, y2 + h2)
        return (x3, y3, r - x3, b - y3)

    @staticmethod
    def intersec(rect1, rect2):
        '''
        brief: intersection of two rects
        :param rect1, rect2
        :return: intersec rect
        '''
        (x1, y1, w1, h1) = rect1
        (x2, y2, w2, h2) = rect2
        x3, y3 = max(x1, x2), max(y1, y2)
        r, b = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        if r <= x3 or b <= y3:
            return (0, 0, 0, 0)
        else:
            return (x3, y3, r - x3, b - y3)

    @staticmethod
    def overlap(rect1, rect2):
        '''
        brief: overlap of two rects
        :param rect1, rect2
        :return: overlap
        '''
        (xi, yi, wi, hi) = EdgeCluster.intersec(rect1, rect2)
        areai = wi * hi
        if areai < 1e-5:
            return 0.
        (xm, ym, wm, hm) = EdgeCluster.merge(rect1, rect2)
        aream = wm * hm
        return 1.0 * areai / aream