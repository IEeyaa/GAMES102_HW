# -*- coding:utf-8 -*-
# @Author: IEeya
# Lab3：参数拟合
import numpy as np
import matplotlib.pyplot as plt

# 采样点
surface_range_x = range(1, 10)  # range of surface
surface_range_y = range(1, 10)  # range of surface
num = 10
points_x = np.random.choice(surface_range_x, num)
points_y = np.random.choice(surface_range_y, num)

# 岭回归系数
n = num - 1
lambda_number = 0.1


# 参数采样
# 均匀参数化
def Equidistant():
    points_t = [i for i in range(1, len(points_x) + 1)]
    return points_t


# 弦长参数化
def Chordal():
    points_t = []
    t = 1
    points_t.append(t)
    for i in range(1, len(points_x)):
        t += np.sqrt((points_x[i] - points_x[i - 1]) ** 2 + (points_y[i] - points_y[i - 1]) ** 2)
        points_t.append(t)
    return points_t


# 中心参数化
def Centripetal():
    points_t = []
    t = 1
    points_t.append(t)
    for i in range(1, len(points_x)):
        t += np.sqrt(np.sqrt((points_x[i] - points_x[i - 1]) ** 2 + (points_y[i] - points_y[i - 1]) ** 2))
        points_t.append(t)
    return points_t


sigma = 1


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


# 最小二乘法
def Binary_min(x, x_list, y_list, n):
    X = np.mat([[item ** j for j in range(n)] for item in x_list])
    Y = y_list
    XT = np.transpose(X)
    XTX_inv = np.linalg.inv(np.dot(XT, X))
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
    final_t_equidistant = Equidistant()
    points_t_test = [i for i in np.arange(1, final_t_equidistant[-1] + 0.1, 0.1)]
    final_x_equidistant = Gauss(points_t_test, final_t_equidistant, points_x)
    final_y_equidistant = Gauss(points_t_test, final_t_equidistant, points_y)

    final_t_chordal = Chordal()
    points_t_test = [i for i in np.arange(1, final_t_chordal[-1] + 0.1, 0.1)]
    final_x_chordal = Gauss(points_t_test, final_t_chordal, points_x)
    final_y_chordal = Gauss(points_t_test, final_t_chordal, points_y)

    final_t_centripetal = Centripetal()
    points_t_test = [i for i in np.arange(1, final_t_centripetal[-1] + 0.1, 0.1)]
    final_x_centripetal = Gauss(points_t_test, final_t_centripetal, points_x)
    final_y_centripetal = Gauss(points_t_test, final_t_centripetal, points_y)

    plt.scatter(points_x, points_y, color="orange")
    plt.plot(final_x_equidistant, final_y_equidistant, color="red")
    plt.plot(final_x_chordal, final_y_chordal, color="blue")
    plt.plot(final_x_centripetal, final_y_centripetal, color="green")

    plt.show()
