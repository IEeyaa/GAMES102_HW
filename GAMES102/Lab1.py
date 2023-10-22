# -*- coding:utf-8 -*-
# @Author: IEeya
# Lab1：四种方式的曲线拟合
import numpy as np
import matplotlib.pyplot as plt

# 采样点
points_x = [1, 2, 3, 4, 6, 7, 10, 12]
surface_range_y = range(1, 10)  # range of surface
num = 8
n = num - 1
points_y = np.random.choice(surface_range_y, num)

# 绘制图像的区域
final_x = [i for i in np.arange(1, 12.2, 0.2)]

X = np.linspace(-5, 5, 100)
Y = np.sin(X) + np.random.normal(0, 0.1, 100)
# 绘制图像的区域
test_x = [i for i in np.arange(-5, 5, 0.2)]

# 高斯函数的方差Sigma
sigma = 2
# 岭回归的干扰系数lambda
lambda_number = 3


# 第j项拉格朗日基函数
def lagrange_base(x, i, x_list):
    tempMulUp = 1
    tempMulDown = 1
    for j in range(0, len(x_list)):
        if i != j:
            tempMulDown *= x_list[i] - x_list[j]
            tempMulUp *= x - x_list[j]
    return tempMulUp / tempMulDown


# 拉格朗日插值
def Lagrange(x, x_list, y_list):
    final = []
    for item in x:
        number_sum = 0
        for i in range(0, len(x_list)):
            number_sum += y_list[i] * lagrange_base(item, i, x_list)
        final.append(number_sum)
    return final


def gauss_base(x):
    return np.exp(-pow(x, 2) / (2 * sigma ** 2))


# 高斯插值
def Gauss(x, x_list, y_list):
    A = np.mat([[gauss_base(abs(x_list[i] - x_list[j])) for j in range(len(x_list))] for i in range(len(x_list))])
    Y = y_list
    X = np.dot(np.linalg.inv(A), Y)
    final = []
    X = X.tolist()
    for item in x:
        number_sum = 0
        for i in range(0, len(x_list)):
            number_sum += X[0][i] * gauss_base(item - x_list[i])
        final.append(number_sum)
    return final


# 最小二乘法
def Binary_min(x, x_list, y_list):
    X = np.mat([[item ** j for j in range(n)] for item in x_list])
    Y = y_list
    XT = np.transpose(X)
    XTX_inv = np.linalg.inv(np.dot(XT, X))
    XTX_inv_XT = np.dot(XTX_inv, XT)
    B = np.dot(XTX_inv_XT, Y)
    B = B.tolist()
    print(B)
    final = []
    for item in x:
        number_sum = B[0][0]  # Initialize with the constant term
        for i in range(1, n):
            number_sum += B[0][i] * (item ** i)
        final.append(number_sum)

    return final


# 岭回归
def Ridge_min(x, x_list, y_list):
    X = np.mat([[item ** j for j in range(n)] for item in x_list])
    Y = y_list
    XT = np.transpose(X)
    XTX = np.dot(XT, X) + lambda_number * np.eye(n)
    XTX_inv = np.linalg.inv(XTX)
    XTX_inv_XT = np.dot(XTX_inv, XT)
    B = np.dot(XTX_inv_XT, Y)
    B = B.tolist()

    final = []
    for item in x:
        number_sum = B[0][0]  # Initialize with the constant term
        for i in range(1, n):
            number_sum += B[0][i] * (item ** i)
        final.append(number_sum)

    return final


if __name__ == '__main__':
    final_y_gauss = Gauss(final_x, points_x, points_y)
    final_y_lagrange = Lagrange(final_x, points_x, points_y)
    final_y_binary = Binary_min(final_x, points_x, points_y)
    final_y_ridge = Ridge_min(final_x, points_x, points_y)

    plt.scatter(points_x, points_y, color="orange")
    #  高斯
    plt.plot(final_x, final_y_gauss, color="red")
    #  拉格朗日
    plt.plot(final_x, final_y_lagrange, color="blue")
    #  最小二乘法
    plt.plot(final_x, final_y_binary, color="green")
    #   岭回归
    plt.plot(final_x, final_y_ridge, color="yellow")

    plt.show()
