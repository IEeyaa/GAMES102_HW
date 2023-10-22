# -*- coding:utf-8 -*-
# @Author: IEeya
# Lab4：三次样条曲线

import matplotlib.pyplot as plt
import numpy as np


class ClickPlot:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.points = []

        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)

        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        if event.inaxes is not None:
            x, y = event.xdata, event.ydata
            x = round(x, 1)  # 将 x 坐标保留一位小数
            y = round(y, 1)  # 将 y 坐标保留一位小数
            self.points.append((x, y))
            self.ax.cla()  # 清除当前图形
            self.ax.set_xlim(0, 10)
            self.ax.set_ylim(0, 10)
            self.ax.plot([p[0] for p in self.points], [p[1] for p in self.points], 'ro')  # 绘制所有点

            if len(self.points) > 2:
                self.update_curve()

            self.fig.canvas.draw()
    #   生成对应的三次样条函数

    def update_curve(self):
        values = []
        n = len(self.points) - 1
        x = np.zeros(shape=(4 * n, 4 * n), dtype=float)
        # 生成y值
        for i in range(1, len(self.points)):
            values.append(0)
            values.append(self.points[i-1][1])
            values.append(self.points[i][1])
            values.append(0)
        # 将列表转换为NumPy数组
        y = np.array(values)

        # 生成大矩阵
        # 头
        now_x = self.points[0][0]
        x[0][0] = 6 * now_x
        x[0][1] = 2
        x[1][0] = now_x ** 3
        x[1][1] = now_x ** 2
        x[1][2] = now_x
        x[1][3] = 1
        for i in range(2, 4 * n - 4, 4):
            now_x = self.points[int(i / 4) + 1][0]
            # row1
            x[i][i - 2] = now_x ** 3
            x[i][i - 1] = now_x ** 2
            x[i][i] = now_x
            x[i][i + 1] = 1
            # row2
            x[i + 1][i - 2] = 3 * now_x ** 2
            x[i + 1][i - 1] = 2 * now_x
            x[i + 1][i] = 1
            x[i + 1][i + 2] = -3 * now_x ** 2
            x[i + 1][i + 3] = -2 * now_x
            x[i + 1][i + 4] = -1
            # row3
            x[i + 2][i - 2] = 6 * now_x
            x[i + 2][i - 1] = 2
            x[i + 2][i + 2] = -6 * now_x
            x[i + 2][i + 3] = -2
            # row4
            x[i + 3][i + 2] = now_x ** 3
            x[i + 3][i + 3] = now_x ** 2
            x[i + 3][i + 4] = now_x
            x[i + 3][i + 5] = 1
        # 尾
        now_x = self.points[n][0]
        x[4 * n - 2][4 * n - 4] = now_x ** 3
        x[4 * n - 2][4 * n - 3] = now_x ** 2
        x[4 * n - 2][4 * n - 2] = now_x
        x[4 * n - 2][4 * n - 1] = 1
        x[4 * n - 1][4 * n - 2] = 6 * now_x
        x[4 * n - 1][4 * n - 1] = 2

        result = np.linalg.inv(x) @ y

        final_x = []
        final_y = []

        for i in range(0, len(result), 4):
            # 生成X轴上的一系列点
            x = np.linspace(self.points[int(i / 4)][0], self.points[int(i / 4) + 1][0], 100)  # 从起始点到结束点生成100个点
            final_x.append(x)
            # 计算Y轴上的值
            a, b, c, d = result[i:i + 4]
            y = a * x ** 3 + b * x ** 2 + c * x + d
            final_y.append(y)

        # 绘制多条曲线
        for x, y in zip(final_x, final_y):
            self.ax.plot(x, y)



if __name__ == "__main__":
    click_plot = ClickPlot()
    plt.show()