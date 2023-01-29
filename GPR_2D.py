import math

import numpy as np
import matplotlib.pyplot as plt
import GPy
import random
import cv2

from pprint import pprint

### 2차원 방사선 데이터 만들기
# X축, Y축을 -5 ~ 5 사이의 범위로 만들고, 0인 지점의 방사선 세기를 50으로 설정하고
# 나머지 부분의 강도는 이전과 마찬가지로 역제곱 법칙으로 계산

X = np.arange(-5, 5.01, 0.01)
Y = np.arange(-5, 5.01, 0.01)
xv, yv = np.meshgrid(X, Y)
m_grid = np.dstack((xv.ravel(), yv.ravel()))[0]
#print(m_grid.shape) # (1002001, 2)

# 방사선 물질이 위치할 것이라 가정한 중앙 부분의 좌표
#print(m_grid[501000])   # (0,0)

values = np.zeros(1002001, float).reshape(-1, 1)
values[501000] = 50

cnt=1
while(True):
    difference = math.sqrt((m_grid[501000][0] - m_grid[501000-cnt][0])**2 + (m_grid[501000][1] - m_grid[501000-cnt][1])**2)
    distance = math.sqrt((difference**2 + 1))
    values[501000-cnt][0] = 50 / (distance ** 2)
    values[501000+cnt][0] = values[501000-cnt][0]
    if (cnt == 501000):
        break
    cnt += 1
    
# 관측값이 위치할 위치
# 지금은 (-5,-5) -> (5,5) 로 이동하는 형태인데, 변경 필요함
measure_xy = np.zeros(42).reshape(-1 ,2)
measure_values = np.zeros(21, float).reshape(-1, 1)
print(measure_xy.shape) # (21, 1)
for i in range(21):
    measure_xy[i][0] = m_grid[50100*i][0]
    measure_xy[i][1] = m_grid[50100*i][1]
    measure_values[i][0] = values[50100*i][0]

print(measure_xy.shape)
print(measure_values.shape)

kernel = GPy.kern.Matern32(input_dim=2, variance=1.0, lengthscale=1.0, ARD=False)

# GPRegression의 likelihood를 Poisson으로 변경함
model = GPy.models.GPRegression(X=measure_xy, Y=measure_values, kernel=kernel)
print(model)
model.optimize(messages=True)
print(model)

pred_point = m_grid

f_mean, f_var = model._raw_predict(pred_point)
f_mean = np.exp(f_mean)

#model.plot()
plt.scatter(x=pred_point[:,0], y=pred_point[:,1], c=f_mean, cmap='jet')
plt.show()

plt.scatter(x=m_grid[:,0], y=m_grid[:,1], c=values, cmap='jet')
plt.show()