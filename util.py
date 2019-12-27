'''
 '
 ' @file  util.py
 '
 ' @brief  Any util tools
 '
 ' @version  1.0
 '
 ' @date  2019/12/23 20:53:59
 '
 ' @author  Red, 735467224@qq.com
 '
'''

import os
import cv2
import numpy as np

import re
import requests
from urllib import error

class util:
    def __init__(self):
        pass

    @staticmethod
    def listFiles(rootdir):
        '''
        brief: make a list that contains all filenames in the root directory
        :param rootdir: root directory
        :return: filenames list
        '''
        list, files = os.listdir(rootdir), []
        for sublist in list:
            path = os.path.join(rootdir, sublist)
            files.append(path) if os.path.isfile(path) \
                else files.extend(util.listFiles(path))
        return files

    @staticmethod
    def grabImages(keyword, savepath, total=100):
        '''
        brief: grab and save images from url
        :param keyword: word used for search images
        :param savepath: save path (to be made if not existed)
        :param total: total num of grab images
        :return: none
        '''
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        url, pullnum = 'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=' + keyword + '&pn=', 1 + int(total/60)
        for i in range(pullnum):
            try:
                urlt, currentid = url + str(i * 60), i * 60
                result = requests.get(urlt, timeout=10)
            except error.HTTPError as e:
                print('error: ' + str(e))
                continue
            else:
                util.dowmloadPicture(result.text, savepath, currentid, total)
        return

    @staticmethod
    def dowmloadPicture(htmlstr, savepath, currentid, total):
        '''
        brief: grab and save images from url
        :param htmlstr: word used for search images
        :param savepath: save path
        :param currentid: id of downloading image
        :param total: total num of grab images
        :return: none
        '''
        pic_url, pullnum = re.findall('"objURL":"(.*?)",', htmlstr, re.S), min(60, total-currentid+1)
        for i in range(pullnum):
            try:
                idx, pic = i + currentid, requests.get(pic_url[i], timeout=7)
            except BaseException as e:
                print('error: ' + str(e))
                continue
            else:
                savename, progress = savepath + '/' + str(idx) + '.jpg', str(idx) + '/' + str(total)
                with open(savename, 'wb') as f:
                    f.write(pic.content), print(progress, savename)
        return

    @staticmethod
    def resize(image, dsize):
        '''
        brief: resize an image that keep aspect ratio
        :param image: source image
        :param dsize: (width_dst, height_dst)
        :return: resized image
        '''
        (h, w, c), (dw, dh) = image.shape, dsize

        ratiow, ratioh, dst = 1.0 * dw / w, 1.0 * dh / h, np.zeros((dh, dw, c), np.uint8)

        ratio = ratioh if (h * ratiow - dh) > (w * ratioh - dw) else ratiow

        rw, rh, x, y = int(ratio * w), int(ratio * h), int(0.5 * (dw - ratio * w)), int(0.5 * (dh - ratio * h))

        dst[y:y+rh, x:x+rw] = cv2.resize(image, (rw, rh))

        return dst

    @staticmethod
    def merge(rect1, rect2):
        '''
        brief: merge two rects
        :param rect1, rect2
        :return: merged rect
        '''
        (x1, y1, w1, h1), (x2, y2, w2, h2) = rect1, rect2
        x3, y3, r, b = min(x1, x2), min(y1, y2), max(x1 + w1, x2 + w2), max(y1 + h1, y2 + h2)
        return (x3, y3, r - x3, b - y3)

    @staticmethod
    def intersec(rect1, rect2):
        '''
        brief: intersection of two rects
        :param rect1, rect2
        :return: intersec rect
        '''
        (x1, y1, w1, h1), (x2, y2, w2, h2) = rect1, rect2
        x3, y3, r, b = max(x1, x2), max(y1, y2), min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        return (0, 0, 0, 0) if (r <= x3 or b <= y3) else (x3, y3, r - x3, b - y3)

    @staticmethod
    def overlap(rect1, rect2):
        '''
        brief: overlap of two rects
        :param rect1, rect2
        :return: overlap
        '''
        (xi, yi, wi, hi) = util.intersec(rect1, rect2)
        areai = wi * hi
        if areai < 1e-5:
            return 0.

        (xm, ym, wm, hm) = util.merge(rect1, rect2)
        aream = wm * hm
        return 1.0 * areai / aream

    @staticmethod
    def distance(point1, point2):
        return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

    @staticmethod
    def makeVideo(imgname, videoname):
        '''
        brief: make a video
        :param imgname, the source image
        :param videoname, the save video name
        :return: none
        '''
        image, fourcc = cv2.resize(cv2.imread(imgname, -1), (640, 480)), cv2.VideoWriter_fourcc("X", "V", "I", "D")

        videowriter = cv2.VideoWriter(videoname, fourcc, 25, (640, 480))

        ks, src, dst = [0, 5, 0, 9, 13, 0, 21, 0, 0], image.astype(np.float), image.astype(np.float)

        for i in range(len(ks)-1):
            k, k_ = ks[i], ks[i+1]
            src = cv2.GaussianBlur(image, (k, k), 0) if k > 0 else image.astype(np.float)
            dst = cv2.GaussianBlur(image, (k_, k_), 0) if k_ > 0 else image.astype(np.float)

            for j in range(100):
                ratio = j / 100.0

                temp = (1.0 - ratio) * src + ratio * dst
                temp = temp.astype(np.uint8)

                cv2.imshow('video', temp), videowriter.write(temp)
                cv2.waitKey(10)

        videowriter.release()
        return

    @staticmethod
    def modifyVideo(srcvideo, videoname):
        '''
        brief: modify a video
        :param srcvideo, the source video
        :param videoname, the save video name
        :return: none
        '''
        cap, fourcc = cv2.VideoCapture(srcvideo), cv2.VideoWriter_fourcc("X", "V", "I", "D")

        ret, image = cap.read()

        (h, w, _) = image.shape

        videowriter = cv2.VideoWriter(videoname, fourcc, 25, (w, h))

        ks, idx = [0, 5, 0, 9, 13, 0, 21, 0, 0], 0

        while (True):

            ret, image = cap.read()

            if image is None:
                print('The source=' + srcvideo + ' is lost or ending.')
                break

            i = int(idx / 100)

            if i>=len(ks)-1:
                break

            k, k_, ratio = ks[i], ks[i+1], (idx % 100) / 100.0

            src = cv2.GaussianBlur(image, (k, k), 0) if k > 0 else image.astype(np.float)

            dst = cv2.GaussianBlur(image, (k_, k_), 0) if k_ > 0 else image.astype(np.float)

            temp = (1.0 - ratio) * src + ratio * dst

            temp = temp.astype(np.uint8)

            cv2.imshow('video', temp), videowriter.write(temp)

            idx += 1

            if (cv2.waitKey(1) >= 0):
                break

        videowriter.release()
        return

    @staticmethod
    def mergeVideo(image_dir, videoname):
        '''
        brief: make a video from list images
        :param image_dir, directory of list images
        :param videoname, the save video name
        :return: none
        '''
        videowriter = cv2.VideoWriter(videoname, cv2.VideoWriter_fourcc("X", "V", "I", "D"), 20, (640, 480))

        image = np.zeros((480, 640, 3), np.uint8)
        for i in range(800):
            for j in range(4,5):
                filename = image_dir + '/timg' + str(j+1) + '-' + str(i+1+j*800) + '.jpg'
                block = cv2.imread(filename, -1)
                print(filename)

                #x, y = j%5 * 640, int(j/5) * 480
                #image[y:y+480, x:x+640] = block
                image = block
            videowriter.write(image)

        videowriter.release()

class VideoSave:
    def __init__(self, savepath=None):
        '''
        :param savepath: savepath='../...avi'   It will save the result video while run(source).
        brief: __init__
        '''
        self.srcimg = None              # the source image
        self.w = 0                      # the width of source video/avi
        self.h = 0                      # the height of source video/avi
        self.count = 0                  # the count of frames

        self.savepath = savepath        # the video save path
        self.videowriter = None         # the videoWriter
        self.videowritersrc = None      # the source videoWriter

    def __del__(self):
        '''
        brief: __del__
        '''
        if self.videowriter is not None:
            self.videowriter.release()
            self.videowriter = None

        if self.videowritersrc is not None:
            self.videowritersrc.release()
            self.videowritersrc = None

    def save(self, frame):
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


        vframe = np.zeros((sh, sw, 3), np.uint8)

        vframe[:, :hsw], vframe[:, hsw:] = util.resize(self.srcimg, (hsw, sh)), util.resize(frame, (hsw, sh))

        self.videowriter.write(vframe), self.videowritersrc.write(self.srcimg)

        return