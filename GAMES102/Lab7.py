# -*- coding:utf-8 -*-
# @Author: IEeya
# Lab7：全局法生成极小曲面 & 曲面参数化
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 加载OBJ模型
obj = "HW6_models/Nefertiti_face.obj"  # 替换为你的OBJ文件路径
# obj = "HW6_models/Balls.obj"  # 替换为你的OBJ文件路径
vertices = []
faces = []
# 创建一个空字典用于存储顶点的邻域信息
vertex_neighbors = {}
# 存储每个点的邻域点的索引
neighbor_points = {}
# 存储每个点的边界情况
boundary_points = []

pi = 3.1415


# 如果一个三角形的一条边不与其他三角形共享，它就是边界三角形，对应的两个顶点就是边界

def load_data():
    with open(obj, "r") as file:
        for line in file:
            if line.startswith("v "):
                vertex = list(map(float, line.split()[1:]))
                vertices.append(np.array(vertex))
            elif line.startswith("f "):
                face = [int(vertex.split("/")[0]) - 1 for vertex in line.split()[1:]]
                faces.append(np.array(face))
    # 遍历每个三角形
    for triangle_index, triangle in enumerate(faces):
        for vertex_index in triangle:
            # 如果顶点索引不在字典中，创建一个新的项
            if vertex_index not in vertex_neighbors:
                vertex_neighbors[vertex_index] = [triangle_index]
            else:
                # 如果顶点索引已经存在，将当前三角形索引添加到邻域列表中
                vertex_neighbors[vertex_index].append(triangle_index)
    for vertex_index, neighbor_triangles in vertex_neighbors.items():
        neighbors = set()  # 存储邻域点的集合
        for triangle_index in neighbor_triangles:
            for vertex in faces[triangle_index]:
                if vertex != vertex_index:
                    neighbors.add(vertex)
        neighbor_points[vertex_index] = neighbors
    #   找出边界点
    # 找到所有边界边
    boundary_edges = set()
    for i, triangle in enumerate(faces):
        for j in range(3):
            edge = tuple(sorted([triangle[j], triangle[(j + 1) % 3]]))
            if edge in boundary_edges:
                boundary_edges.remove(edge)
            else:
                boundary_edges.add(edge)

    # 找到所有边界点
    temp_boundary_points = set()
    for edge in boundary_edges:
        temp_boundary_points.update(edge)

    boundary_points.append(list(temp_boundary_points))

    draw(vertices)


# 绘制球体的三角网格
def draw(result):
    polygons = [[result[index] for index in face] for face in faces]
    fig = plt.figure()
    # 创建一个3D坐标轴
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(Poly3DCollection(polygons, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)


def draw_2d(result):
    polygons = [[result[index] for index in face] for face in faces]
    fig = plt.figure()
    # 创建一个2D坐标轴（不需要projection参数）
    ax = fig.add_subplot(222)
    ax.add_collection(PolyCollection(polygons, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)


def global_calculation():
    n = len(vertices)
    #    构建矩阵
    L = np.zeros(shape=(n, n), dtype=float)
    Y = np.zeros(shape=(n, 3), dtype=float)
    for i in range(0, n):
        L[i][i] = 1
    #     对于所有的顶点，更新weight
    for vertex_index, neighbor_triangles in vertex_neighbors.items():
        # 仅针对内部节点更新
        if vertex_index not in boundary_points[0]:
            total_cot = 0
            # 遍历所有的邻接边
            for neighbor_point_index in neighbor_points[vertex_index]:
                # 三角形
                for triangle_index in neighbor_triangles:
                    if neighbor_point_index in faces[triangle_index]:
                        total_cot += cot_dist(vertex_index, neighbor_point_index, triangle_index)
            for neighbor_point_index in neighbor_points[vertex_index]:
                temp_cot = 0
                # 三角形
                for triangle_index in neighbor_triangles:
                    if neighbor_point_index in faces[triangle_index]:
                        temp_cot += cot_dist(vertex_index, neighbor_point_index, triangle_index)
                L[vertex_index][neighbor_point_index] = -temp_cot / total_cot
        else:
            Y[vertex_index] = vertices[vertex_index]
    draw(np.linalg.inv(L) @ Y)


def cot_dist(p2_index, p3_index, triangle_index):
    p1_index = np.sum(faces[triangle_index]) - p3_index - p2_index
    p1 = vertices[p1_index]
    p2 = vertices[p2_index]
    p3 = vertices[p3_index]

    # 计算边的向量
    edge_AB = p2 - p1
    edge_AC = p3 - p1

    # 计算cot值
    dot_product = np.dot(edge_AB, edge_AC)
    cross_product = np.linalg.norm(np.cross(edge_AB, edge_AC))
    cot_value = dot_product / cross_product
    return cot_value


def dist(p1_index, p2_index):
    p1 = vertices[p1_index]
    p2 = vertices[p2_index]
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2


def parameterization():

    index_list = boundary_points[0]
    polar_coordinates = []
    for index in index_list:
        point = vertices[index]
        vector = np.array(point)
        polar_radius = np.linalg.norm(vector)  # 极径
        polar_angle = np.arctan2(vector[1], vector[0])  # 极角
        polar_coordinates.append((index, polar_radius, polar_angle))

    # 按照极角对索引列表进行排序
    sorted_index_list = [index for index, _, _ in sorted(polar_coordinates, key=lambda x: x[2])]

    result = [[] for i in range(0, len(vertices))]
    # 边界固定, 每边安排n/4个点
    n = len(vertices)
    temp_n = len(boundary_points[0])
    # 映射在一个半径为1的圆形中
    for i, vertex_index in enumerate(sorted_index_list):
        result[vertex_index] = [-math.cos(i * 2 * pi / temp_n), -math.sin(i * 2 * pi / temp_n)]

    # 构建方程组
    L = np.zeros(shape=(n, n), dtype=float)
    Y = np.zeros(shape=(n, 2), dtype=float)
    for i in range(0, n):
        L[i][i] = 1
    for vertex_index, neighbor_triangles in vertex_neighbors.items():
        # 仅针对内部节点更新
        if vertex_index not in boundary_points[0]:
            total_cot = 0
            # 遍历所有的邻接边
            for neighbor_point_index in neighbor_points[vertex_index]:
                # 三角形
                for triangle_index in neighbor_triangles:
                    if neighbor_point_index in faces[triangle_index]:
                        total_cot += cot_dist(vertex_index, neighbor_point_index, triangle_index)
            for neighbor_point_index in neighbor_points[vertex_index]:
                temp_cot = 0
                # 三角形
                for triangle_index in neighbor_triangles:
                    if neighbor_point_index in faces[triangle_index]:
                        temp_cot += cot_dist(vertex_index, neighbor_point_index, triangle_index)
                L[vertex_index][neighbor_point_index] = -temp_cot / total_cot
        else:
            Y[vertex_index] = result[vertex_index]

    draw_2d(np.linalg.inv(L) @ Y)


if __name__ == "__main__":
    load_data()
    global_calculation()
    parameterization()
    plt.show()