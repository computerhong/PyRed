'''
 '
 ' @file  Hungary.py
 '
 ' @brief  Hungarian Optimal Matching Algorithm
 '
 ' @version  1.0
 '
 ' @date  2019/10/15 20:53:59
 '
 ' @author  Red, 735467224@qq.com
 '
'''

import copy
import numpy as np

'''
 ' @brief any structure declarations
'''
class szres:
    def __init__(self, rowsum_= None, colsum_= None, orowsum_= None):
        self.rowsum = copy.deepcopy(rowsum_)
        self.colsum = copy.deepcopy(colsum_)
        self.orowsum = copy.deepcopy(orowsum_)

class mkres:
    def __init__(self, rowmark_= None, colmark_= None):
        self.rowmark = copy.deepcopy(rowmark_)
        self.colmark = copy.deepcopy(colmark_)

class ercas:
    def __init__(self, mat_= None, rowsum_= None, mk_= None):
        self.mat = copy.deepcopy(mat_)
        self.rowsum = copy.deepcopy(rowsum_)
        self.mk = copy.deepcopy(mk_)

class ezres:
    def __init__(self, mat_= None, erzeros_= None):
        self.mat = copy.deepcopy(mat_)
        self.erzeros = copy.deepcopy(erzeros_)

'''
 ' @brief Hungarian Optimal Matching Algorithm
'''
class Hungary:
    def __init__(self, cost):
        '''
        brief: __init__
        :param cost: input cost matrix
        '''
        self.costM = copy.deepcopy(cost)
        self.indices = []

    def Invoke(self):
        '''
        brief: run the matchings for input cost matrix
        :return: none
        '''
        self.indices = []
        costN = self.normRowsCols(self.costM)

        ez = ezres()
        while True:

            sz = self.getSingleZeros(costN)

            mk = self.markRowsCols(sz)

            er = ercas(costN, sz.rowsum, mk)

            (reb, ez) = self.setEndRowZeros(er)
            if reb:
                break

            costN = copy.deepcopy(ez.mat)

        self.indices = self.zerosAlloc(ez.erzeros, self.indices, 0)


    def getIndices(self):
        '''
        brief: get an example of input cost matrix
        :return: array, matching indices
        '''
        return np.array(self.indices)

    @staticmethod
    def getExpInput():
        '''
        brief: get an example of input cost matrix
        :return: array, cost matrix
        '''
        exp_arr = np.array([[12,  7,  9,  7,  9],
                            [ 8,  9,  6,  6,  6],
                            [ 7, 17, 12, 14,  9],
                            [15, 14,  6,  6, 10],
                            [ 4, 10,  7, 10,  9]],
                            dtype=np.float)

        return exp_arr


    def normRowsCols(self, in_):
        '''
        brief: cost matrix minimum normalization
        :param in_: input cost matrix
        :return: array, normalized cost matrix
        '''
        out = copy.deepcopy(in_)
        (h, w) = in_.shape

        for i in range(h):
            imin = min(out[i, :])
            out[i, :] -= imin

        for j in range(w):
            jmin = min(out[:, j])
            out[:, j] -= jmin

        return out

    def markRowsCols(self, in_):
        '''
        brief: try marking zeros with the least lines
        :param in_: input single zeros matrix
        :return: mkres, marked zeros matrix
        '''
        h = len(in_.rowsum)
        w = len(in_.colsum)

        marking = False
        rowmark = np.array([False] * h)
        colmark = np.array([False] * w)

        for i in range(h):
            if 0 == len(in_.rowsum[i]):
                rowmark[i] = True
                marking = True

        while marking:
            marking = False
            for i in range(h):
                if rowmark[i]:
                    ors = in_.orowsum[i]
                    for j in range(len(ors)):
                        x = ors[j]
                        if not colmark[x]:
                            colmark[x] = True
                            marking = True

            for j in range(w):
                if colmark[j]:
                    cs = in_.colsum[j]
                    for i in range(len(cs)):
                        y = cs[i]
                        if not rowmark[y]:
                            rowmark[y] = True
                            marking = True

        out = mkres(rowmark,colmark)
        return out

    def getSingleZeros(self, in_):
        '''
        brief: try matching to get single zeros
        :param in_: input normalized cost matrix
        :return: szres, single zeros matrix
        '''
        (h, w) = in_.shape

        rowsum = [[] for i in range(h)]
        colsum = [[] for i in range(w)]
        for i in range(h):
            for j in range(w):
                if in_[i,j] < 1e-9:
                    rowsum[i].append(j)
                    colsum[j].append(i)

        orowsum = copy.deepcopy(rowsum)
        singlezero = True
        while singlezero:
            singlezero = False
            for i in range(len(rowsum)):
                rcs = rowsum[i]
                if 1 == len(rcs):
                    x = rcs[0]
                    singlezero = (1 != len(colsum[x]))

                    for j in range(len(colsum[x])):
                        y = colsum[x][j]
                        if y != i:
                            rowsum[y].remove(x)

                    colsum[x] = [i]

            for j in range(len(colsum)):
                rcs = colsum[j]
                if 1 == len(rcs):
                    y = rcs[0]
                    singlezero = (1 != len(rowsum[y]))

                    for i in range(len(rowsum[y])):
                        x = rowsum[y][i]
                        if x != j:
                            colsum[x].remove(y)

                    rowsum[y] = [j]

        out = szres(rowsum, np.array(colsum), np.array(orowsum))
        return out

    def setEndRowZeros(self, in_):
        '''
        brief: try marking other zeros to get the final matrix
        :param in_: input marked zeros matrix
        :return: (bool, ezres), marked zeros final matrix
        '''
        h = len(in_.mk.rowmark)
        w = len(in_.mk.colmark)
        line = 0

        for i in range(h):
            if not in_.mk.rowmark[i] or in_.mk.colmark[i]:
                line += 1

        out = ezres(None,[])
        out.mat = copy.deepcopy(in_.mat)
        if line < w:
            minval = 99999
            for i in range(h):
                for j in range(w):
                    iscorvered = (not in_.mk.rowmark[i] or in_.mk.colmark[j])
                    if not iscorvered:
                        value = in_.mat[i, j]
                        minval = min(minval, value)

            for i in range(h):
                for j in range(w):
                    if in_.mk.rowmark[i]:
                        out.mat[i, j] -= minval
                    if in_.mk.colmark[j]:
                        out.mat[i, j] += minval
        else:
            for i in range(h):
                vps = []
                rs = in_.rowsum[i]
                for j in range(len(rs)):
                    x = rs[j]
                    y = i
                    vps.append([x, y])
                out.erzeros.append(vps)

            return (True, out)

        return (False, out)


    def zerosAlloc(self, in_, out, iter):
        '''
        brief: get the result of matchings from the marked zeros final matrix
        :param in_: input marked zeros final matrix
        :param iter: obj for iteration
        :return: array, result of matchings
        '''
        if iter >= len(in_):
            return  out

        zeros = copy.deepcopy(in_[iter])
        for j in range(len(zeros)):
            zero = zeros[j]
            x = zero[0]

            size = len(out)
            isalloc = False
            for k in range(size):
                if x == out[k]:
                    isalloc = True
                    break

            if not isalloc:
                out.append(x)
                out = self.zerosAlloc(in_, out, iter+1)

        if len(out) > 0 and len(out) < len(in_):
            del(out[-1])

        return out