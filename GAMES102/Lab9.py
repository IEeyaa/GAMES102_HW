# -*- coding:utf-8 -*-
# @Author: IEeya
# Lab9：基于论文《Surface Simplification Using Quadric Error Metrics》的曲面简化
# 论文链接：http://mgarland.org/files/papers/quadrics.pdf
"""
    论文思想简介：
        曲面简化的过程主要体现在每一轮迭代的过程中，需要去减少一些点（自然，也要去除和这些点连接的边）
        QEM（Quadric Error Metrics）主要集中于讲述了如何去找到这些需要被去掉的点
            1.v1-v2是原网格中的一条边 (相连)
            2.||v1-v2|| < threshold (不相连接但是靠的很近)
        Q矩阵的定义是一个顶点周围所有平面的Kp(fundamental error quadric K)矩阵的和
            Kp = [ a^2 ab ac ad ]
                 [ ab b^2 bc bd ]
                 [ ac bc c^2 cd ]
                 [ ad bd cd d^2 ]
        对于所有的候选点对，新的点会是两个点的加权平均，而这个“权”则由cost的最小化决定【移动的同时尽可能少的改变图形外观】
            cost = VT (Q1 + Q2) V

            V_new = [ q11 q12 q13 q14 ]-1 * [ 0]
                    [ q21 q22 q23 q24 ]     [ 0]
                    [ q31 q32 q33 q34 ]     [ 0]
                    [   0   0   0   1 ]     [ 1]
    显然，每一轮iteration，点会减少一半 :}

    当然，这个算法有一个致命的缺陷：对于边界的判断势必导致大部分的面片都会产生严重的扭曲（即最后的许多面片都会集中在边缘）
"""
import heapq

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 加载OBJ模型
obj = "HW6_models/Nefertiti_face.obj"  # 替换为你的OBJ文件路径
# obj = "HW6_models/Balls.obj"  # 替换为你的OBJ文件路径
vertices = []
edges = []
faces = []
Q_all = []
# 创建一个空字典用于存储顶点的邻域信息
vertex_neighbors = {}
# 存储每个点的邻域点的索引
neighbor_points = {}
# 存储所有的边界
boundary_edges = set()
# 存储边界点情况
boundary_points = []
# 步长
alpha = 0.1


class Edge:
    def __init__(self, p1_index, p2_index):
        self.points = [p1_index, p2_index]
        self.new_point_base, self.new_point = self.count_new_point()
        self.cost = self.count_cost()

    def count_new_point(self):
        Y = np.zeros(shape=(4, 1), dtype=float)
        Y[3][0] = 1
        try:
            # 尝试计算逆矩阵
            temp_Q = Q_all[self.points[0]] + Q_all[self.points[1]]
            temp_Q[3][0] = 0
            temp_Q[3][1] = 0
            temp_Q[3][2] = 0
            temp_Q[3][3] = 1
            v = np.linalg.inv(temp_Q) @ Y
            v2 = np.array([v[0][0], v[1][0], v[2][0]])
        except np.linalg.LinAlgError:
            print("当前矩阵不可逆")
            v2 = (vertices[self.points[0]] + vertices[self.points[1]]) / 2
            v = np.append(v2, 1)
        return v, v2

    def count_cost(self):
        if (self.points[0], self.points[1]) in boundary_edges:
            return 0.1
        elif self.points[0] in boundary_points[0] or self.points[1] in boundary_points[0]:
            return 0.05
        dot1 = np.transpose(self.new_point_base) @ (Q_all[self.points[0]] + Q_all[self.points[1]])
        dot2 = dot1 @ self.new_point_base
        return dot2[0][0]

    def __lt__(self, other):
        return self.cost < other.cost


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
    for triangle in faces:
        for i in range(3):
            vertex1 = triangle[i]
            vertex2 = triangle[(i + 1) % 3]
            if vertex1 < vertex2:
                edge = [vertex1, vertex2]
            else:
                edge = [vertex2, vertex1]
            if edge not in edges:
                edges.append(edge)
    # 找出边界点
    # 找到所有边界边
    temp_boundary_points = set()
    for i, triangle in enumerate(faces):
        for j in range(3):
            edge = tuple(sorted([triangle[j], triangle[(j + 1) % 3]]))
            if edge in boundary_edges:
                boundary_edges.remove(edge)
            else:
                boundary_edges.add(edge)
    for edge in boundary_edges:
        temp_boundary_points.update(edge)

    boundary_points.append(list(temp_boundary_points))


# 绘制球体的三角网格
def draw():
    polygons = [[vertices[index] for index in face] for face in faces if face is not None]
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


def iteration():
    # 简单的边界判定法，邻域三角形个数较少（在这里 <= 3）
    count_time = 1
    count_max = 0
    Y = np.zeros(shape=(4, 1), dtype=float)
    Y[3][0] = 1
    # 创建一个优先队列
    pq = []
    for i in range(0, count_time):
        # 计算所有点的Q
        for point_index in range(0, len(vertices)):
            Q_all.append(get_Q(point_index))
        # 边对收敛
        for edge in edges:
            point1_index = edge[0]
            point2_index = edge[1]
            eg = Edge(point1_index, point2_index)
            pq.append(eg)

        # 启动！
        heapq.heapify(pq)
        count = 0
        while not len(pq) == 0 and count < count_max:
            count += 1
            top_edge = heapq.heappop(pq)
            point1_index = top_edge.points[0]
            point2_index = top_edge.points[1]
            if vertices[point1_index] is None or vertices[point2_index] is None:
                print("error!")
            new_point = top_edge.new_point

            # 更新点
            vertices[point1_index] = new_point
            vertices[point2_index] = None

            # 更新三角形
            for face_index in range(0, len(faces)):
                if faces[face_index] is None:
                    continue
                if point1_index in faces[face_index] and point2_index in faces[face_index]:
                    faces[face_index] = None
                elif point2_index in faces[face_index]:
                    faces[face_index] = [point1_index if element == point2_index else element for element in faces[face_index]]

            # 更新优先队列
            modify_q = []
            vertex_neighbors[point1_index].extend(vertex_neighbors[point2_index])

            Q_all[point1_index] = get_Q(point1_index)
            for item_pq in pq:
                all_points = item_pq.points
                # 均包含, 删除这条边
                if point1_index in all_points and point2_index in all_points:
                    continue
                # 只有p_1, 更新所有的cost
                elif point1_index in all_points:
                    modify_q.append(Edge(item_pq.points[0], item_pq.points[1]))
                # 只有p_2，将其都变为p_1
                elif point2_index in all_points:
                    temp_points = all_points
                    if temp_points[0] == point2_index:
                        temp_points[0] = point1_index
                    else:
                        temp_points[1] = point1_index
                    if temp_points[0] < temp_points[1]:
                        modify_q.append(Edge(temp_points[0], temp_points[1]))
                    else:
                        modify_q.append(Edge(temp_points[1], temp_points[0]))
                # 其它
                elif point1_index is not None and point2_index is not None:
                    modify_q.append(item_pq)
            pq = modify_q
            heapq.heapify(modify_q)


def get_Q(p1_index):
    Q = np.zeros(shape=(4, 4), dtype=float)
    for triangle in vertex_neighbors[p1_index]:
        if faces[triangle] is None:
            continue
        p1, p2, p3 = faces[triangle]
        p = get_plane_factor(p1, p2, p3)
        Q += p.transpose() @ p
    return Q


def get_plane_factor(p1, p2, p3):
    # 三个点的坐标
    point1 = np.array(vertices[p1])
    point2 = np.array(vertices[p2])
    point3 = np.array(vertices[p3])

    # 计算两个向量A和B
    A = point2 - point1
    B = point3 - point1

    # 计算法向量N
    N = np.cross(A, B)
    # 获取平面方程系数
    A, B, C = N
    D = -np.dot(N, point1)

    return np.array([[A, B, C, D]])


if __name__ == "__main__":
    load_data()
    draw()
    iteration()
    vertices_count = 0
    face_count = 0
    for item in vertices:
        if item is not None:
            vertices_count += 1
    for item in faces:
        if item is not None:
            face_count += 1
    print("vertices last is : " + str(vertices_count))
    print("faces last is : " + str(face_count))
    draw()
    plt.show()

