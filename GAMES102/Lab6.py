# -*- coding:utf-8 -*-
# @Author: IEeya
# Lab6：Laplace法生成极小曲面
import math
import matplotlib.pyplot as plt
import numpy as np
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
# 步长
alpha = 0.1


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


# 绘制球体的三角网格
def draw():
    polygons = [[vertices[index] for index in face] for face in faces]
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

    plt.show()


def iteration():
    # 简单的边界判定法，邻域三角形个数较少（在这里 <= 3）
    count_time = 50
    for i in range(0, count_time):
        if i % 10 == 0:
            print("iteration: " + str(i))
        # 开始一轮新的迭代
        for vertex_index, neighbor_triangles in vertex_neighbors.items():
            # 仅针对内部节点更新
            if vertex_index not in boundary_points[0]:
                center_point = vertices[vertex_index]
                A = 0
                T = 0

                # 遍历所有的邻接边
                for neighbor_point_index in neighbor_points[vertex_index]:
                    neighbor_point = vertices[neighbor_point_index]
                    total_cot = 0
                    # 三角形
                    for triangle_index in neighbor_triangles:
                        if neighbor_point_index in faces[triangle_index]:
                            total_cot += cot_dist(vertex_index, neighbor_point_index, triangle_index)
                    # # Laplace-Beltrami算子定义面积
                    A += 0.125 * dist_square(neighbor_point, center_point) * total_cot
                    T += total_cot * (-center_point + neighbor_point)
                vertices[vertex_index] += alpha * T * 0.25 / A



def dist_square(p1, p2):
    return math.sqrt(dist(p1, p2))


def dist(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2


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


if __name__ == "__main__":
    load_data()
    iteration()
    draw()
