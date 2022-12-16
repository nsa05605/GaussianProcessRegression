import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# 3차원 그래프를 그리기 위한 Axes3D 모듈 임포트

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')

x = np.arange(0, 10, 0.1)
y = np.sin(x)
z = np.cos(x)
ax.plot(x,y,z)

plt.show()

ax.scatter(x,y,z,color='r',alpha=0.5)
plt.show()