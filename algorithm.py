'''
 '
 ' @file  algorithm.py
 '
 ' @brief  Any algorithmic implementations
 '
 ' @version  1.0
 '
 ' @date  2019/9/25 20:53:59
 '
 ' @author  Red, 735467224@qq.com
 '
'''

from Hungary import Hungary

class algorithm:
    def __init__(self):
        pass

    @staticmethod
    def Examples():
        '''
        brief: run any examples for this algorithm
        :return: none
        '''

        ####################################################################
        ########################## getPolygonArea ##########################
        points = []
        points.append((0, 0))
        points.append((0, 3))
        points.append((4, 0))

        area = algorithm.getPolygonArea(points)
        print("Exp.1 Polygon Area = " + str(area) + ", expect(6).")
        ####################################################################


        ####################################################################
        ############################# Hungary ##############################
        exp_cost = Hungary.getExpInput()

        indices = algorithm.getCostMatchings(exp_cost)
        print("Exp.2 Hungary Result is", indices, ", expect(1,2,4,3,0).")
        ####################################################################


        ####################################################################
        ###                         (next Exp.)                          ###
        ####################################################################


        ####################################################################
        ############################## Cacel ###############################
        print()
        print("All examples for algorithm were completed.")
        ####################################################################

    @staticmethod
    def getPolygonArea(points):
        '''
        brief: calculate the Polygon Area with vertex coordinates
        refer: https://blog.csdn.net/qq_38862691/article/details/87886871

        :param points: list, input vertex coordinates
        :return: float, polygon area
        '''
        sizep = len(points)
        if sizep<3:
            return 0.0

        area = points[-1][0] * points[0][1] - points[0][0] * points[-1][1]
        for i in range(1, sizep):
            v = i - 1
            area += (points[v][0] * points[i][1])
            area -= (points[i][0] * points[v][1])

        return abs(0.5 * area)

    @staticmethod
    def getCostMatchings(cost):
        '''
        brief: solve the minimum weighted bipartite matching problem by using Hungary Algorithm
        refer: https://wenku.baidu.com/view/20428d2cba0d4a7303763a8c.html

        :param cost: matching cost matrix
        :return: array, matching indices
        '''
        hgy = Hungary(cost)
        hgy.Invoke()

        indices = hgy.getIndices()
        return indices