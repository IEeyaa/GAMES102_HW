# -*- coding:utf-8 -*-
# @Author: IEeya
# Lab5：Chaikin细分

import matplotlib.pyplot as plt

alpha = 0.5


class ClickPlot:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.points = []

        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)

        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def on_click(self, event):
        if event.inaxes is not None:
            x, y = event.xdata, event.ydata
            x = round(x, 1)  # 将 x 坐标保留一位小数
            y = round(y, 1)  # 将 y 坐标保留一位小数
            self.points.append((x, y))
            self.draw_points()


    def on_key_press(self, event):
        if event.key == 'j':
            # 在按下"k"键时执行特定函数，这里以打印点的坐标为例
            self.update_chaikin()
        if event.key == "k":
            self.update_chaikin_3()
        if event.key == "l":
            self.update_cluster()

    def update_chaikin(self):
        new_points = []

        index = 0
        new_points.append((self.points[index][0] * 0.75 + self.points[index + 1][0] * 0.25,
                           self.points[index][1] * 0.75 + self.points[index + 1][1] * 0.25))
        for i in range(1, len(self.points) - 1):
            new_points.append((self.points[i-1][0] * 0.25 + self.points[i][0] * 0.75,
                               self.points[i-1][1] * 0.25 + self.points[i][1] * 0.75))
            new_points.append((self.points[i][0] * 0.75 + self.points[i+1][0] * 0.25,
                               self.points[i][1] * 0.75 + self.points[i+1][1] * 0.25))

        index = len(self.points) - 1
        new_points.append((self.points[index - 1][0] * 0.25 + self.points[index][0] * 0.75,
                           self.points[index - 1][1] * 0.25 + self.points[index][1] * 0.75))
        self.points = new_points
        self.draw_points()

    def update_chaikin_3(self):
        new_points = []
        index = 0
        new_points.append((self.points[index][0] * 0.75 + self.points[index+1][0] * 0.25,
                           self.points[index][1] * 0.75 + self.points[index+1][1] * 0.25))
        for i in range(1, len(self.points) - 1):
            new_points.append((self.points[i-1][0] * 0.125 + self.points[i][0] * 0.75 + self.points[i+1][0] * 0.125,
                               self.points[i-1][1] * 0.125 + self.points[i][1] * 0.75 + self.points[i+1][1] * 0.125))
            new_points.append((self.points[i][0] * 0.5 + self.points[i+1][0] * 0.5,
                               self.points[i][1] * 0.5 + self.points[i+1][1] * 0.5))

        index = len(self.points) - 1
        new_points.append((self.points[index - 1][0] * 0.25 + self.points[index][0] * 0.75,
                           self.points[index - 1][1] * 0.25 + self.points[index][1] * 0.75))
        self.points = new_points
        self.draw_points()

    def update_cluster(self):
        new_points = []
        n = len(self.points)
        for i in range(0, len(self.points)):
            new_points.append((self.points[i][0], self.points[i][1]))

            qr_x = self.points[i % n][0] * 0.5 + self.points[(i+1) % n][0] * 0.5
            mr_x = self.points[(i-1) % n][0] * 0.5 + self.points[(i+2) % n][0] * 0.5
            qr_y = self.points[i % n][1] * 0.5 + self.points[(i+1) % n][1] * 0.5
            mr_y = self.points[(i-1) % n][1] * 0.5 + self.points[(i+2) % n][1] * 0.5

            temp_x = qr_x + alpha * (qr_x - mr_x)
            temp_y = qr_y + alpha * (qr_y - mr_y)

            new_points.append((temp_x, temp_y))
        self.points = new_points
        self.draw_points()

    def draw_points(self):
        self.ax.cla()  # 清除当前图形
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.plot([p[0] for p in self.points], [p[1] for p in self.points])  # 绘制所有点
        self.fig.canvas.draw()



if __name__ == "__main__":
    click_plot = ClickPlot()
    plt.show()