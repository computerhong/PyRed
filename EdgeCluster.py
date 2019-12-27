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
import random
import numpy as np

from util import util
from util import VideoSave

'''
 ' @brief Edge clustering algorithm
 ' there were any indoor scenes that contained irregular objects different from others such as lines and color-rectangles in the floor.
 ' it will find these objects and clustering them.
'''
class EdgeCluster(VideoSave):
    def __init__(self, savepath=None):
        '''
        brief: __init__
        :param savepath: savepath='../...avi'   It will save the result video while run(source).
        '''
        VideoSave.__init__(self, savepath)

        self.idx_block = 0          # current index of block, if truncature it will running from the current index block at the next frame
        self.hisrects = []          # detected rects on the pass
        self.hiscount = []          # times of lose focus for the detected rects on the pass
        self.ovlcount = []          # times of touch by the detected rects on the pass
        self.retainov = 3           # threshold times of touch on a hisrect
        self.retainfm = 10          # threshold times of lose focus for a rect, if exceed then destroy
        self.thresElapsed = 0.05    # running elapsed of per frame, if exceed then truncature

        self.thresc = 150           # lower threshold of Canny at the current frame
        self.secth = 30             # interval of Canny thresholds
        self.threscmin = 100        # lower threshold of Canny at the automation loop

        self.rw = 160               # resize width of source image
        self.rh = 120               # resize height of source image
        self.bs = 20                # block size

        self.threslp = 30           # threshold of points' number (in the resize block size)
        self.thresacc = 12          # threshold of accuracy for hough lines
        self.thresln = 0.2          # threshold of percentages for hough lines to points
        self.thresov = 0.2          # threshold of overlap between rects
        self.thresdt = 0.1          # threshold of percentage for cluster area to image size

    def run(self, source):
        '''
        brief: run the clusterings for source video/avi
        :param source: input source, video(source=0) or avi(source='../source.avi')
        :return: none
        '''
        cap = cv2.VideoCapture(source)

        while (True):
            ret, frame = cap.read()

            if frame is None:
                print('The source=' + str(source) + ' is lost or ending.')
                break

            start = time.clock()

            self.srcimg, (self.h, self.w, _), self.count = frame.copy(), frame.shape, (self.count + 1) % 999

            isInterrupt, rects = self.makePreRects(frame)

            mergerects = self.makeMergeRects(rects)

            ret = self.makeHistoryRects(mergerects)

            back, area = self.makeClusterBack(isInterrupt)


            frame = frame + back * 192

            frame[frame > 255] = 255


            self.thresc = max((self.thresc+1) if area > self.thresdt else (self.thresc-1), self.threscmin)

            stop = time.clock()


            fps = int(1.0 / (stop - start))

            cv2.putText(frame, str(fps) + ' fps', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow('EdgeCluster', frame), self.save(frame)

            if (cv2.waitKey(1) >= 0):
                break

        cv2.destroyAllWindows()
        cap.release()
        return

    def makePreRects(self, frame):
        '''
        brief: make any preselection rects
        :param frame: input frame
        :return: isInterrupt, the running elapsed of per frame is exceed or not
                rects, the preselection rects
        '''
        start, isInterrupt, rects, blockx, blocky, ratiox, ratioy = \
            time.clock(), False, [], int(self.rw/self.bs), int(self.rh/self.bs), self.w/self.rw, self.h/self.rh

        dst, totalnum = cv2.Canny(cv2.resize(frame, (self.rw, self.rh)),
                                  self.thresc, self.thresc + self.secth), blockx * blocky

        for idxb in range(self.idx_block, self.idx_block + totalnum):
            idxb = idxb % totalnum

            x = idxb % blockx * self.bs + int(0.5 * random.randint(0,1) * self.bs)
            y = int(idxb / blockx) * self.bs + int(0.5 * random.randint(0,1) * self.bs)
            xs, ys = x + self.bs, y + self.bs

            if xs >= self.rw or ys >= self.rh:
                continue

            points = np.argwhere(dst[y:ys, x:xs] > 8)
            if len(points) < self.threslp:
                continue

            hough, lenp = np.zeros(self.thresacc, dtype=np.int), len(points)
            for i in range(lenp - 1):
                for j in range(i, lenp):
                    p1, p2 = points[i], points[j]
                    radian = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
                    deg = (int)((radian + np.pi) * self.thresacc / np.pi) % self.thresacc
                    hough[deg] += 1

            line_percent = 1.0 * hough.max() / hough.sum()
            if line_percent > self.thresln:
                continue

            lt = np.mean(points, axis=0) - 0.5 * self.bs + [x, y]
            rect = (lt[0] * ratiox, lt[1] * ratioy, self.bs * ratiox, self.bs * ratioy)
            rects.append(rect)

            if (time.clock() - start) > self.thresElapsed:
                isInterrupt, self.idx_block = True, idxb
                break

        return (isInterrupt, rects)

    def makeMergeRects(self, rects):
        '''
        brief: merge preselection rects
        :param rects: the preselection rects
        :return: mergerects, merged rects
        '''
        mergerects = []
        for i, rect in enumerate(rects):
            ismerge = False
            for j, mrect in enumerate(mergerects):
                if util.overlap(rect, mrect) > self.thresov:
                    ismerge, (xj, yj, wj, hj), (xi, yi, wi, hi) = True, mrect, rect
                    mergerects[j] = tuple(np.array([xj+xi, yj+yi, wj+wi, hj+hi])*0.5)
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
        for mrect in mergerects:
            ismerge = False
            for i, hrect in enumerate(self.hisrects):
                if util.overlap(mrect, hrect) > self.thresov:
                    ismerge, (xj, yj, wj, hj), (xi, yi, wi, hi) = True, hrect, mrect
                    self.hisrects[i] = tuple(np.array([xj+xi, yj+yi, wj+wi, hj+hi])*0.5)
                    self.hiscount[i], self.ovlcount[i] = 0, self.ovlcount[i] + 1
                    break
            if not ismerge:
                self.hisrects.append(mrect), self.hiscount.append(0), self.ovlcount.append(0)

        for i in range(len(self.hisrects)-1,0,-1):
            self.hiscount[i] += 1
            if self.hiscount[i] > self.retainfm:
                del self.hisrects[i], self.hiscount[i], self.ovlcount[i]

        return 0

    def makeClusterBack(self, isInterrupt):
        '''
        brief: make the clustering result
        :param isInterrupt: the running elapsed of per frame is exceed or not
        :return: back, the clustering mask
                area, the clustering area percentage
        '''
        back, circles = np.zeros((self.h, self.w), dtype=np.uint8), []
        for i, rect in enumerate(self.hisrects):
            if self.ovlcount[i] < self.retainov:
                continue

            ismerge, (xi, yi, wi, hi) = False, rect
            centeri, radiusi = (int(xi+wi/2), int(yi+hi/2)), int((wi+hi)/4)

            for j, (centerj, radiusj) in enumerate(circles):
                thresr = 1.25 * (radiusi + radiusj)
                if util.distance(centeri, centerj) < thresr:
                    ismerge = True
                    cv2.line(back, centeri, centerj, 255, int(thresr))

            if not ismerge:
                circles.append([centeri, radiusi])

        backrs, backsize = cv2.resize(back, (int(self.w/8), int(self.h/8))), self.w * self.h / 64
        area = 1.0 if isInterrupt else (len(np.argwhere(backrs > 8)) / backsize)
        return (cv2.merge([cv2.Sobel(back, cv2.CV_8UC1, 1, 1)] * 3), area)