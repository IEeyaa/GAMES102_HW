# -*- coding:utf-8 -*-
# @Author: IEeya
# Lab10：RBF 基于点云的曲面重建

"""
    kitten 参数
    # 可调节参数
    alpha = 0.5
    # 整体缩放比例尺
    update_index = 1.0
    # Marching Cubes 范围
    MC = np.mgrid[-30:40:2, 0:100:2, -30:40:2]
    # 采样比例（如果是10则采样原点云中1/10的点数）
    cluster_number = 1
    # 法向量相关参数
    radius = 5.0
    max_nn = 10
    # 法向量估计方式: 1确定相机，2根据周围点，3确定轴线
    methods = 2

    arma04 参数
    # 外部定义参数
    alpha = 0.5
    # 整体缩放比例尺
    update_index = 10
    # Marching Cubes 范围
    MC = np.mgrid[-15:15:0.5, -15:15:0.5, -15:15:0.5]
    # 采样比例（如果是10则采样原点云中1/10的点数）
    cluster_number = 2
    # 法向量相关参数
    radius = 5.0
    max_nn = 10
    # 法向量估计方式: 1确定相机，2根据周围点，3确定轴线
    methods = 2
"""
import random

import matplotlib.pyplot as plt
import mcubes
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import open3d as o3d

vertices = []
vertices_temp = []
faces = []
normals_all = []
normals = []
weights = []


"""
    本实验核心的一些参数，请慎重调节
"""

# 点云数据
obj = "HW10_models/Arma_04.obj"  # 替换为你的OBJ文件路径
# obj = "HW10_models/kitten_04.obj"  # 替换为你的OBJ文件路径
# obj = "HW10_models/dragon_04.obj"  # 替换为你的OBJ文件路径

# 外部定义参数
alpha = 0.5
# 整体缩放比例尺
update_index = 10
# Marching Cubes 范围
MC = np.mgrid[-15:15:0.5, -15:15:0.5, -15:15:0.5]
# 采样比例（如果是10则采样原点云中1/10的点数）
cluster_number = 2
# 法向量相关参数
radius = 5.0
max_nn = 10
# 法向量估计方式: 1确定相机，2根据周围点，3确定轴线
methods = 3


def load_data():
    with open(obj, "r") as file:
        for line in file:
            if line.startswith("v "):
                vertex = list(map(float, line.split()[1:]))
                vertices.append(np.array(vertex) * update_index)


# 球面
def load_test_data():
    # 生成1000个均匀分布在球面上的点
    n = 1000
    phi = np.random.uniform(0, np.pi, n)  # 极角范围 [0, pi]
    theta = np.random.uniform(0, 2 * np.pi, n)  # 方位角范围 [0, 2*pi]
    r = 5.0  # 球的半径

    # 将球坐标转换为笛卡尔坐标
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    v_temp = list(zip(x, y, z))
    for item in v_temp:
        vertices.append(np.array([item[0], item[1], item[2]]))


# 绘制球体的三角网格
def draw():
    # polygons = [[vertices[index] for index in face] for face in faces]
    fig = plt.figure(dpi=120)
    # 创建一个3D坐标轴
    ax = fig.add_subplot(111, projection='3d')
    # ax.add_collection3d(Poly3DCollection(polygons, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

    # 分离点坐标的x、y、z分量
    x = [point[0] for point in vertices]
    y = [point[1] for point in vertices]
    z = [point[2] for point in vertices]

    # 绘制点
    ax.scatter(x, y, z, c='b', marker='.', s=2, linewidth=0, alpha=1, cmap='spectral')

    x = [point[0] for point in normals]
    y = [point[1] for point in normals]
    z = [point[2] for point in normals]

    # 绘制点
    ax.scatter(x, y, z, c='r', marker='*', s=2, linewidth=0, alpha=1, cmap='spectral')

    n = int(len(vertices_temp[0]) / 2)
    # 绘制法向量方向向量
    for i in range(n):
        x_start, y_start, z_start = vertices_temp[0][i]
        x_end, y_end, z_end = vertices_temp[0][i + n]  # 终点为顶点坐标加上法向量

        ax.quiver(x_start, y_start, z_start, x_end - x_start, y_end - y_start, z_end - z_start, color='r', pivot='tail',
                  linewidth=1.0)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


# 绘制重建网格
def final_draw(v, t):
    # 绘制最终结果
    polygons = [[v[index] for index in face] for face in t]
    fig = plt.figure()
    # 创建一个3D坐标轴
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(Poly3DCollection(polygons, facecolors='cyan', linewidths=.2, edgecolors='r', alpha=.25))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    ax.set_zlim(0, 30)


# 生成法向量
def create_normal():
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))

    if methods == 1:
        tot = np.mean(vertices, axis=0)
        # tot = np.array([-1.5, -1.5, 0])
        # 中心
        pcd.orient_normals_towards_camera_location(tot)
    elif methods == 2:
        # 根据周围点的情况进行方向判断
        pcd.orient_normals_consistent_tangent_plane(3)
    else:
        # 沿坐标轴
        pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([1.0, 0, 0]))
    for i, item in enumerate(pcd.normals):
        normals_all.append(vertices[i] - alpha * np.asarray(item))


# 重建
def reconstruction():
    #   PCL实现法向量估计
    point_total_number = len(vertices)

    num_to_select = point_total_number // cluster_number
    #   生成不重复的随机索引
    vertices_indexes = random.sample(range(point_total_number), num_to_select)
    vertices_indexes.sort()
    create_normal()

    for item in vertices_indexes:
        normals.append(normals_all[item])

    print("normal ready")

    # 建立（超级大的）RBF矩阵
    temp_vertices = [vertices[i] for i in vertices_indexes]
    temp_vertices.extend(normals)
    vertices_temp.append(temp_vertices)

    n = len(vertices_temp[0])

    K = np.zeros(shape=(n, n), dtype=float)
    Y = np.zeros(shape=(n, 1), dtype=float)

    # 外部点
    for i in range(int(n / 2), n):
        Y[i] = alpha

    for i in range(0, n):
        for j in range(0, n):
            K[i][j] = RBF_major(i, j)

    result = np.linalg.inv(K) @ Y
    print("calculation ready")
    weights.append([item[0] for item in result])
    # ff = lambda x, y, z: x ** 2 + y ** 2 + z ** 2 - 25
    ff = lambda x, y, z: sum(
        weights[0][i] * RBF_major_result(x, y, z, i) for i in range(len(vertices_temp[0]))
    )
    X, Y, Z = MC
    u = ff(X, Y, Z)
    obj_vertices, obj_faces = mcubes.marching_cubes(u, 0)
    print("marching cube ready")

    final_draw(obj_vertices, obj_faces)


# RBF函数
def RBF_major(point_1_index, point_2_index):
    # return np.exp(-pow(np.linalg.norm(vertices_temp[0][point_1_index] - vertices_temp[0][point_2_index]), 2) / 2)
    return np.sqrt(((vertices_temp[0][point_1_index][0] - vertices_temp[0][point_2_index][0]) ** 2 +
                    (vertices_temp[0][point_1_index][1] - vertices_temp[0][point_2_index][1]) ** 2 +
                    (vertices_temp[0][point_1_index][2] - vertices_temp[0][point_2_index][2]) ** 2)) ** 3


# 使用r^3作为RBF基函数
def RBF_major_result(x, y, z, point_2_index):
    # res = np.exp(-pow(np.sqrt(((x - vertices_temp[0][point_2_index][0]) ** 2 +
    #                            (y - vertices_temp[0][point_2_index][1]) ** 2 +
    #                            (z - vertices_temp[0][point_2_index][2]) ** 2)), 2) / 2)
    # return res
    return np.sqrt(((x - vertices_temp[0][point_2_index][0]) ** 2 +
                    (y - vertices_temp[0][point_2_index][1]) ** 2 +
                    (z - vertices_temp[0][point_2_index][2]) ** 2)) ** 3


if __name__ == "__main__":
    load_data()
    # load_test_data()
    reconstruction()
    draw()
    plt.show()
