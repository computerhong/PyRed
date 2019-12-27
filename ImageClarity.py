'''
 '
 ' @file  ImageClarity.py
 '
 ' @brief  Image clarity calculation
 '
 ' @version  1.0
 '
 ' @date  2019/12/19 20:53:59
 '
 ' @author  Red, hong.huang@iim.ltd
 '
'''
import cv2
import time
import random
import numpy as np

from util import util
from util import VideoSave

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util

'''
 ' @brief Image clarity calculation
 ' A function of calculating image clarity based on DCT
'''
class ImageClarity(VideoSave):
    def __init__(self, savepath=None):
        '''
        brief: __init__
        :param savepath: savepath='../...avi'   It will save the result video while run(source).
        '''
        VideoSave.__init__(self, savepath)

        self.rsize = 256                                # the size of dct
        self.interval = 100                             # save/overwrite the *.pb every 100 batches
        self.temparam = './params_image_clarity'        # the temporary params save path
        self.logpath = './train_log_image_clarity.csv'  # the log path of trainning

        self.sessic = None                              # tf.Session() used for loading *.pb
        self.input_name = 'input'                       # the input node name of *.pb
        self.inputn_name = 'inputn'                     # the inputn node name of *.pb
        self.output_name = 'output'                     # the output node name of *.pb
        self.input_ic = None                            # the input tensor of *.pb
        self.inputn_ic = None                           # the inputn tensor of *.pb
        self.output_ic = None                           # the output tensor of *.pb

        self.dctindices = self.get_dct_idx()            # the indices of dct

    def load(self, pbsavename):
        '''
        brief: load a *.pb
        :param pbsavename: *.pb, the saved pb filename
        :return: none
        '''
        self.sessic = tf.Session()

        with gfile.FastGFile(pbsavename, 'rb') as f:

            graph_def = tf.GraphDef()

            graph_def.ParseFromString(f.read())

            self.sessic.graph.as_default()

            self.input_ic, self.inputn_ic, self.output_ic = tf.import_graph_def(
                graph_def, return_elements = [self.input_name + ':0', self.inputn_name + ':0', self.output_name + ':0'])

        return

    def run(self, source):
        '''
        brief: run the ImageClarity for source video/avi
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

            self.srcimg, (self.h, self.w, _), self.count = frame.copy(), frame.shape, (self.count + 1) % 99999

            clarity = self.getClarity(frame)

            stop = time.clock()


            fps, color = int(1.0 / (stop - start)), (0, 0, 255) if clarity<3.0 else (0, 255, 0)

            cv2.putText(frame, str(fps) + ' fps', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.putText(frame, 'clarity : ' + str(clarity)[:3], (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow('ImageClarity', frame), self.save(frame)

            if (cv2.waitKey(1) >= 0):
                break

        cv2.destroyAllWindows()
        cap.release()
        return

    def write(self, sess, pbsavename):
        '''
        brief: write a *.pb
        :param sess: tf.Session()
        :param pbsavename: *.pb, the saved pb filename
        :return: none
        '''
        constant_graph = graph_util.convert_variables_to_constants(
            sess, sess.graph_def, [self.output_name])

        with tf.gfile.FastGFile(pbsavename, mode='wb') as f:
            f.write(constant_graph.SerializeToString())

        return

    def train(self, labeltxt, pbsavename, restore=False):
        '''
        brief: train a *.pb from the dataset described in labeltxt
        :param labeltxt: dataset, such as per line (filename, clarity): lena.bmp, 5.1
        :param pbsavename: *.pb, the saved pb filename
        :param restore: restore the params or not
        :return: none
        '''
        train_x, train_xn, train_y = self.get_data_train(labeltxt)

        learningrate, batchsize, totalstep = 0.0001, 100, 1000000

        input = tf.placeholder(tf.float32, [None, 256], name=self.input_name)

        inputn = tf.placeholder(tf.float32, [None, 1], name=self.inputn_name)

        inputr = tf.reshape(input, [-1, 16, 16, 1])

        conv1 = tf.layers.conv2d(inputr, 16, 3, 1, 'same', activation=tf.nn.relu)

        pool1 = tf.layers.max_pooling2d(conv1, 2, 2)

        conv2 = tf.layers.conv2d(pool1, 32, 3, 1, 'same', activation=tf.nn.relu)

        pool2 = tf.layers.max_pooling2d(conv2, 2, 2)

        plat1 = tf.layers.dense(tf.reshape(pool2, [-1, 4 * 4 * 32]), 1)

        plat2 = tf.layers.dense(tf.concat([plat1, inputn], 1), 1)

        output = tf.identity(plat2, name=self.output_name)

        labels = tf.placeholder(tf.float32, [None, 1])

        loss = tf.losses.mean_squared_error(labels, output)

        optimizer = tf.train.GradientDescentOptimizer(learningrate).minimize(loss)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        f, steps, losses = open(self.logpath, 'w'), [], []

        with tf.Session() as sess:

            sess.run(init_op)

            if restore:

                tf.train.Saver().restore(sess, self.temparam)


            plt.ion()

            plt.figure(1)

            for i in range(1, totalstep):

                idx1 = [x for x in range(len(train_x))]

                random.shuffle(idx1)

                idx1 = idx1[:batchsize]


                input_x, input_xn, input_y = train_x[idx1], train_xn[idx1], train_y[idx1]

                _, loss_ = sess.run([optimizer, loss],
                                    feed_dict = { input: input_x, inputn: input_xn, labels: input_y })

                steps.append(i), losses.append(loss_)

                print('Step:', i, '| test loss: %.2f' % loss_)


                plt.clf()

                plt.plot(steps, losses)

                plt.pause(0.001)


                f.write(str(i) + ',' + str(loss_) + '\n')

                if i % self.interval == 0:

                    tf.train.Saver().save(sess, self.temparam, write_meta_graph=False), self.write(sess, pbsavename)

            plt.ioff()

        f.close()

        return

    def getClarity(self, image):
        '''
        brief: calculate clarity of an image
        :param image: an image
        :return: clarity
        '''
        if self.sessic is None:
            print('The pb file is not loaded.')
            return None

        grayrs = cv2.cvtColor(cv2.resize(image, (self.rsize, self.rsize)), cv2.COLOR_BGR2GRAY)

        dctgrs, grsm = self.get_dct(grayrs), cv2.medianBlur(grayrs, 3) / 256.0

        if np.isnan(dctgrs[0]):
            return None

        input_x, inputn_x = np.reshape(dctgrs, [1, self.rsize]), np.reshape([np.max(grsm)-np.min(grsm)], [1, 1])

        clarity = self.sessic.run([self.output_ic],
                                  feed_dict = { self.input_ic: input_x, self.inputn_ic: inputn_x })[0][0][0]

        return clarity

    def getClarities(self, image_dir):
        '''
        brief: calculate clarity of every image and make a dict which key=file & value=clarity
        :param image_dir: directory of images
        :return: a clarity dict
        '''
        if self.sessic is None:
            print('The pb file not loaded.')
            return None

        filenames, exts, clarities = util.listFiles(image_dir), ['.jpg', '.bmp', '.png'], dict()

        for i, file in enumerate(filenames):
            if len([0 for ext in exts if ext in file]) < 1:
                continue

            image = cv2.imread(file, -1)
            if image is None:
                continue

            clarities[file] = self.getClarity(image)

        return clarities

    @staticmethod
    def make_data_train(savepath, labeltxt, keyword=None):
        '''
        brief: make a labeltxt from (grab and) save images
        :param savepath: save path (to be made if not existed)
        :param labeltxt: dataset, ex. 'lena.bmp, 5.1'   / (filename, clarity)
        :param keyword: grab images from url if not None
        :return: none
        '''
        if keyword is not None:
            util.grabImages(keyword, savepath)
            return

        filenames, lines, keys, exts, clabase, idx = \
            util.listFiles(savepath), [], ['gauss', 'shrink'], ['.jpg', '.bmp', '.png'], 5.0, 0

        for file in filenames:
            if len([0 for key in keys if key in file])>0 or len([0 for ext in exts if ext in file])<1:
                continue

            img = cv2.imread(file, -1)
            if img is None:
                continue

            filename, line, idx = file.split('/')[-1], file + ',' + str(clabase) + '\n', idx+1
            lines.append(line), print(str(idx) + '/' + str(len(filenames)), file)

            for i in range(1, 11):
                ks, clarity = 1 + i * 2, str(clabase - 0.5 * i)
                blurfile = file.replace(filename, filename.split('.')[0] + '_gauss_' + str(ks) + '.jpg')

                cv2.imwrite(blurfile, cv2.GaussianBlur(img, (ks, ks), 0))
                lines.append(blurfile + ',' + str(clarity) + '\n')

            minv, maxv = np.min(img), np.max(img)

            for i in range(1, 11):
                dish, clarity = (maxv-minv) * i * 0.05, str(clabase - 0.5 * i)
                shrinkfile = file.replace(filename, filename.split('.')[0] + '_shrink_' + str(i) + '.jpg')

                cv2.normalize(img, img, int(minv + dish), int(maxv - dish), cv2.NORM_MINMAX)
                cv2.imwrite(shrinkfile, img), lines.append(shrinkfile + ',' + str(clarity) + '\n')

        with open(labeltxt, 'w') as f:
            f.writelines(lines)

        return

    def get_data_train(self, labeltxt):
        '''
        brief: get datas used for trainning
        :param labeltxt: dataset, such as per line (filename, clarity): lena.bmp, 5.1
        :return: clarity
        '''
        lines, train_x, train_xn, train_y = open(labeltxt, 'r').readlines(), [], [], []

        for idx in range(len(lines)):

            items, progress = lines[idx].split(','), str(idx) + '/' + str(len(lines))

            filename, clarity = items[0], float(items[1])

            image = cv2.imread(filename, -1)

            if image is None:
                print(progress, filename + ' loading error...')
                continue
            else:
                print(progress, filename + ' loading...')

            grayrs = cv2.cvtColor(cv2.resize(image, (self.rsize, self.rsize)), cv2.COLOR_BGR2GRAY)

            dctgrs, grsm = self.get_dct(grayrs), cv2.medianBlur(grayrs, 3) / 256.0

            train_x.append(dctgrs), train_xn.append([np.max(grsm)-np.min(grsm)]), train_y.append([clarity])

        return (np.array(train_x), np.array(train_xn), np.array(train_y))

    def get_dct(self, image):
        '''
        brief: calculate reduced dct of an image
        :param image: an image
        :return: dct(reduced)
        '''
        dctimg, farr = cv2.dct(image.astype(np.float32)), np.zeros(self.rsize, np.float)

        for i in range(self.rsize):
            farr[i] = np.mean(np.abs(dctimg[self.dctindices[i]]))

        return -np.log(farr/sum(farr)) if sum(farr)>0 else farr

    def get_dct_idx(self):
        '''
        brief: calculate dct indices
        :return: dct indices dict
        '''
        size, disdict = self.rsize, dict()

        for y in range(size):
            for x in range(size):

                dis = int(np.sqrt(x**2 + y**2) - 0.5)
                if dis >= size:
                    continue

                if dis not in disdict.keys():
                    disdict[dis] = []
                disdict[dis].append([y, x])

        del disdict[0][0]
        for i in range(size):
            disdict[i] = np.array(disdict[i])

        return  disdict