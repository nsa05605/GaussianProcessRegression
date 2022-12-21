import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# 3차원 그래프를 그리기 위한 Axes3D 모듈 임포트

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')

x = np.arange(0, 10, 0.1)
y = np.sin(x)
z = np.cos(x)

# 그래프를 plot으로 그리기
#ax.plot(x,y,z)

# 그래프를 산점도로 그리기
#ax.scatter(x, y, z, color = 'r', alpha = 0.5)
#ax.scatter(x, z, y, color = 'g', alpha = 0.5) # 다른 그래프를 동시에 그리기도 가능

# 그래프를 3D 평면으로 그리기
x_m, y_m = np.meshgrid(x, y)
z = x_m + 5 * y_m
ax.plot_surface(x, y, z, cmap="brg_r")

plt.show()