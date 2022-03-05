from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

# 在同一画布画多张子图
# fig, a = plt.subplots(2,2)
# x = np.arange(1,5)
# #绘制平方函数
# a[0][0].plot(x,x*x)
# a[0][0].set_title('square')
# a[0][0].legend('squre')
# #绘制平方根图像
# a[0][1].plot(x,np.sqrt(x))
# a[0][1].set_title('square root')
# #绘制指数函数
# a[1][0].plot(x,np.exp(x))
# a[1][0].set_title('exp')
# #绘制对数函数
# a[1][1].plot(x,np.log10(x))
# a[1][1].set_title('log')
# plt.show()
accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
labels = [1,2]
predictions = np.argmax([[0.1, 0.5, 0.4],[0.7,0.3,0.2]],axis=1)
res = tf.cast(tf.equal(labels, predictions), dtype=tf.float32)
print(res)
res = tf.reduce_mean(res)
print(res)