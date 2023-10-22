# -*- coding:utf-8 -*-
# @Author: IEeya
# Lab2：基于tensorflow的RBF神经网络拟合
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 生成一些示例数据
# 采样点
X = np.linspace(-5, 5, 100)
Y = np.sin(X) + np.random.normal(0, 0.1, 100)
# 绘制图像的区域
test_x = [i for i in np.arange(-5, 5, 0.2)]
# 聚类中心数量
num_centers = 4
# 训练次数
ep = 5000


# 定义RBF神经网络模型
class RBFModel(tf.keras.Model):
    def __init__(self, num_of_centers):
        super(RBFModel, self).__init__()
        self.num_centers = num_of_centers
        # 要学习的部分
        self.centers = tf.Variable(tf.random.normal(shape=(num_of_centers,)))
        self.beta = tf.Variable(1.0, dtype=tf.float32)

    def call(self, inputs):
        x = tf.expand_dims(inputs, axis=-1)
        c = tf.expand_dims(self.centers, axis=0)
        return tf.reduce_sum(tf.exp(-self.beta * tf.square(x - c) / 2), axis=1)


def RBF_generate():
    # 顺序模型
    model = tf.keras.Sequential()

    # 输入层
    model.add(tf.keras.layers.Input(shape=(1,)))

    # 隐藏层
    layer1 = RBFModel(num_centers)
    model.add(layer1)
    # ax+b层, dense代表数据维度
    layer2 = tf.keras.layers.Dense(1)
    model.add(layer2)

    # 编译模型, 梯度下降+最小二乘法
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 数据
    X_arr = np.array(X, dtype=np.float32)
    Y_arr = np.array(Y, dtype=np.float32)
    max_x = max(X_arr)
    max_y = max(Y_arr)
    X_arr /= max_x
    Y_arr /= max_y

    # 训练模型
    model.fit(X_arr, Y_arr, epochs=ep, verbose=2)

    test_x_arr = np.array(test_x, dtype=np.float32)
    test_x_arr /= max_x
    # 使用模型进行预测
    pred_y = model.predict(test_x_arr)
    pred_y *= max_y
    return pred_y


if __name__ == '__main__':
    pred_y = RBF_generate()
    # 绘制拟合结果
    plt.scatter(X, Y, label='Actual')
    plt.plot(test_x, pred_y, label='Predicted', color='red')
    plt.legend()
    plt.show()
