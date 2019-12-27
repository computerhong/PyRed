'''
 '
 ' @file  OcclusionDetect.py
 '
 ' @brief  Occlusion detect algorithm
 '
 ' @version  1.0
 '
 ' @date  2019/12/18 20:53:59
 '
 ' @author  Red, hong.huang@iim.ltd
 '
'''

import cv2
import time
import numpy as np

from util import util
from util import VideoSave

'''
 ' @brief Occlusion detect algorithm
 ' there were any outdoor scenes that the camera is occlusion. It will be detected.
'''
class OcclusionDetect(VideoSave):
    def __init__(self, savepath=None):
        '''
        :param savepath: savepath='../...avi'   It will save the result video while run(source).
        brief: __init__
        '''
        VideoSave.__init__(self, savepath)
        self.vframe = None

        self.result = -1                # the result of this detection(init:-1 / normal:0 / occlusion:1)
        self.intervalrs = 15            # the interval of reserve a frame
        self.boundrs = [5, 30]          # the bound of reserves

        self.thresvary = 30             # the threshold of vary in foreground
        self.thresvaryacc = 50          # the threshold of accumulating vary frame nums in foreground
        self.thresmatch = 0.95          # the threshold of background match template
        self.thresvmbn = self.thresvaryacc * 4
                                        # the threshold of accumulating vary of foreground
        self.meanmatch = -1.0           # the mean value of background match template

        self.reductimg = 4              # the reduction of frames
        self.rsize = 64                 # the re-size of frames
        self.rsize2 = self.rsize ** 2   # the area of a resized frame
        self.bsize = 8                  # the blocksize of a resized frame
        self.bwidth = int(self.rsize/self.bsize)
                                        # the block width of a resized frame
        self.bnum = self.bwidth ** 2    # the blocknum of a resized frame

        self.oframes = []               # the reserved resized frames
        self.backgound = None           # the dynamic background
        self.varymask = np.zeros([self.rsize, self.rsize, 3], np.float32)
                                        # the mask of foreground
        self.idxva = 0                  # the current index of varyacc
        self.varyacc = [0] * self.thresvaryacc
                                        # the accumulating vary of foreground


        self.edgeidx = [0, 1, 2, 3, 4, 5, 6, 7,
                        15, 23, 31, 39, 47, 55, 63,
                        62, 61, 60, 59, 58, 57, 56,
                        48, 40, 32, 24, 16, 8, 0, 1, 2, 3]
                                        # the edge indices of a resized frame

    def feed(self, frame):
        '''
        brief: feed a frame and get the result of detection.
                It needs at least 50 frames to init the background (50 = self.intervalrs * self.boundrs[0])
        :param frame: input frame
        :return: the result of this detection
                init:-1
                normal:0
                occlusion:1
        '''
        if frame is None:
            return -1

        self.srcimg, (self.h, self.w, _), self.count = frame, frame.shape, (self.count + 1) % 999

        self.processOne(frame)

        return self.result

    def run(self, source):
        '''
        brief: run the OcclusionDetect for source video/avi
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

            self.processOne(frame)

            stop = time.clock()


            result, resimg, fps = self.result, self.mergeResult(), int(1.0 / (stop - start))

            cv2.putText(resimg, str(fps) + ' fps', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow('OcclusionDetect', resimg), self.save()

            if self.backgound is not None:
                cv2.imshow('background', self.backgound)

            if result == 1:
                cv2.imshow('occlusion', resimg)

            if (cv2.waitKey(1) >= 0):
                break

        if self.videowriter is not None:
            self.videowriter.release()
            self.videowriter =  None

        if self.videowritersrc is not None:
            self.videowritersrc.release()
            self.videowritersrc= None

        cv2.destroyAllWindows()
        cap.release()
        return

    def processOne(self, src):
        '''
        brief: process one frame
        :param frame: input frame
        :return: the result image of this detection
        '''
        if self.result == 1:
            self.oframes, self.varymask, self.meanmatch, self.result, self.varyacc =\
                [], 0.0 * self.varymask, -1.0, -1, [0] * self.thresvaryacc

        ismatch, isvary, varyimg = True, False, np.zeros([self.rsize, self.rsize], np.float32)

        frame, leno = (src / self.reductimg).astype(np.uint8) * self.reductimg, len(self.oframes)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        grayrs = cv2.resize(gray, (self.rsize, self.rsize)).astype(np.float32)

        if(leno >= self.boundrs[0]):

            noframes = np.array(self.oframes).astype(np.uint8)

            back = np.zeros((self.rsize, self.rsize), np.uint8)

            for i in range(self.rsize2):

                x, y = i % self.rsize, int(i/self.rsize)

                back[y, x] = np.argmax(np.bincount(noframes[:, y, x]))

            varyimg, self.backgound = abs(grayrs - back), cv2.resize(back, (self.w, self.h)).astype(np.uint8)


            absgo = cv2.resize(varyimg, (self.bsize, self.bsize))

            isvary = (len(np.argwhere(absgo > self.thresvary)) > 0)

            self.result = 0

        if isvary:

            al, ar = int(self.bwidth/2), self.rsize - int(self.bwidth/2)

            roigrs = grayrs[al:ar, al:ar].astype(np.uint8)

            mtres = cv2.matchTemplate(back, roigrs, cv2.TM_CCOEFF)

            _, max_val, _, _ = cv2.minMaxLoc(mtres)


            degmat = max_val / self.meanmatch

            ismatch, self.meanmatch = (True, max_val) if self.meanmatch < 0 else \
                ((True, 0.5 * (self.meanmatch + max_val)) if degmat > self.thresmatch else (False, self.meanmatch))

            total, maxtotal, self.varymask[:, :, 2], occmask = \
                0, -1, varyimg, cv2.resize(varyimg, (self.bsize, self.bsize)).reshape(self.bnum)[self.edgeidx]

            for i in range(len(occmask)):

                maxtotal, total = max(maxtotal, total), (total+1) if occmask[i] > self.thresvary else max(0, total-1)

            self.varyacc[self.idxva], self.idxva = maxtotal, (self.idxva + 1) % self.thresvaryacc

            self.result = (1 if sum(self.varyacc)>=self.thresvmbn and not ismatch else 0)

        else:
            self.varymask *= 0.5

        if self.count % self.intervalrs == 0:
            self.oframes.append(grayrs)

        if len(self.oframes) > self.boundrs[1]:
            del self.oframes[0]

        return

    def mergeResult(self):
        '''
        brief: merge the source image and the result image of this detection
        :return: the merge image
        '''
        mask, maxval = np.array(self.varymask), min(np.max(self.varymask[:,:,2]), 255)

        cv2.normalize(self.varymask, mask, 0, maxval, cv2.NORM_MINMAX)

        mframe = self.srcimg + cv2.resize(mask, (self.w, self.h))

        mframe[mframe>255] = 255

        return mframe.astype(np.uint8)

    def save(self):
        '''
        brief: save a result video if savepath is not none
        :return: none
        '''
        if self.savepath is None:
            return

        sw, sh, hsw, fps = 1280, 480, 640, 25

        if self.videowriter is None:

            srcpath, fourcc = self.savepath.replace('.avi', '_source.avi'), cv2.VideoWriter_fourcc("X", "V", "I", "D")

            self.videowriter = cv2.VideoWriter(self.savepath, fourcc, fps, (sw, sh))

            self.videowritersrc = cv2.VideoWriter(srcpath, fourcc, fps, (self.w, self.h))

            self.vframe = np.zeros((sh, sw, 3), np.uint8)


        self.vframe[:, :hsw] = util.resize(self.srcimg, (hsw, sh))

        if self.result == 1:

            self.vframe[:, hsw:] = util.resize(self.mergeResult(), (hsw, sh))

        self.videowriter.write(self.vframe)

        self.videowritersrc.write(self.srcimg)