# -*- coding:utf-8 -*-
# @Author: IEeya
# Lab8：平面点集 CVT 的 Lloyd 算法
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay, delaunay_plot_2d

n = 30


# 在正方形中随机生成点
def generate_data():
    # 生成随机点
    # np.random.seed(0)  # 设置随机种子以确保结果可重复
    random_points = np.random.rand(n, 2) * 2 - 1  # 在(-1, 1)正方形内生成随机点

    # 正方形的四个顶点
    square_points = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])

    # 合并随机点和正方形点
    generate_points = np.vstack((random_points, square_points))
    # 生成所有点
    return generate_points


def outsideBox(point):
    return point[0] * point[0] > 1 or point[1] * point[1] > 1


def line_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    # 计算向量AB和CD
    AB = (x2 - x1, y2 - y1)
    CD = (x4 - x3, y4 - y3)

    # 计算向量AC和AD
    AC = (x3 - x1, y3 - y1)
    AD = (x4 - x1, y4 - y1)

    # 计算叉积
    cross1 = AB[0] * AC[1] - AB[1] * AC[0]
    cross2 = AB[0] * AD[1] - AB[1] * AD[0]

    # 如果叉积相乘小于0，表示线段CD与线段AB相交
    if cross1 * cross2 < 0:
        # 计算向量CA和CB
        CA = (x1 - x3, y1 - y3)
        CB = (x2 - x3, y2 - y3)

        # 计算叉积
        cross3 = CD[0] * CA[1] - CD[1] * CA[0]
        cross4 = CD[0] * CB[1] - CD[1] * CB[0]

        # 如果叉积相乘小于0，表示线段AB与线段CD相交
        if cross3 * cross4 < 0:
            # 计算交点坐标
            t = cross3 / (cross3 - cross4)
            intersection_x = x1 + AB[0] * t
            intersection_y = y1 + AB[1] * t

            # 检查交点是否在两条线段上
            if (min(x1, x2) <= intersection_x <= max(x1, x2) and
                    min(y1, y2) <= intersection_y <= max(y1, y2) and
                    min(x3, x4) <= intersection_x <= max(x3, x4) and
                    min(y3, y4) <= intersection_y <= max(y3, y4)):
                return np.array([intersection_x, intersection_y])

    # 如果没有交点，返回None
    return None


iteration = 10


# Lloyd算法，在每一轮迭代中更新points点到对应的Delaunay多边形的中心
# 注意，包中的算法并不保证边界点，因此要自己定义边界
def Lloyd(raw_points):
    temp_points = raw_points
    vor = Voronoi(temp_points)
    tri = Delaunay(temp_points)

    # 可视化
    voronoi_plot_2d(vor)
    delaunay_plot_2d(tri)
    # 开始迭代
    for i in range(0, iteration):
        # 创建Voronoi图
        vor = Voronoi(temp_points)
        all_points = np.array(vor.vertices)

        # 对每一个Regions进行操作
        for region_id, region in enumerate(vor.regions):
            if len(region) > 0 and -1 not in region:
                temp_region = []
                # 检测多出来的边
                for i in region:
                    if outsideBox(all_points[i]):
                        # 加入边界交叉点
                        interaction_points = [sublist[0] if sublist[1] == i
                                              else sublist[1]
                                              for sublist in vor.ridge_vertices if i in sublist]
                        interaction_points = [x for x in interaction_points if
                                              x != -1 and not outsideBox(all_points[x])]
                        for inter_point_index in interaction_points:
                            point1 = all_points[i]
                            point2 = all_points[inter_point_index]
                            point_inter = line_intersection(point1, point2, [-1, -1], [-1, 1])
                            if point_inter is None:
                                point_inter = line_intersection(point1, point2, [-1, 1], [1, 1])
                                if point_inter is None:
                                    point_inter = line_intersection(point1, point2, [1, 1], [1, -1])
                                    if point_inter is None:
                                        point_inter = line_intersection(point1, point2, [1, -1], [-1, -1])
                                        if point_inter is None:
                                            continue
                            temp_region.append(point_inter)
                    else:
                        temp_region.append(all_points[i])

                midPoint_x = np.mean([region[0] for region in temp_region])
                midPoint_y = np.mean([region[1] for region in temp_region])
                point_id = vor.point_region[region_id - 1] - 1
                if point_id < n:
                    temp_points[point_id] = [midPoint_x, midPoint_y]
    vor = Voronoi(temp_points)
    tri = Delaunay(temp_points)

    # 可视化
    voronoi_plot_2d(vor)
    delaunay_plot_2d(tri)

    plt.show()


if __name__ == "__main__":
    points = generate_data()
    Lloyd(points)
