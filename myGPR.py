import numpy as np
import pandas
import matplotlib.pyplot as plt
import GPy
import yaml
import cv2
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

#directory = "C:/Users/rail/PycharmProjects/GaussianProcessRegression/LUNL/"    # 노트북
directory = "C:/Users/jihun/PycharmProjects/GaussianProcessRegression/LUNL/"    # 연구실
input_file = "LUNL_radiation_data.txt"
output_file = "LUNL_2Drad_GPR.txt"

data_raw = np.array(pandas.read_csv(directory + input_file, delimiter=',', header=0, usecols=['x','y','z','counts']))

X = np.array(data_raw[:,0:3])       # x, y, z
Y = np.array([data_raw[:, 3]]).T    # counts
print("Data Ready")


# Gaussian Process Regression
# A poisson distribution is used
poisson_likelihood = GPy.likelihoods.Poisson()
laplace_inf = GPy.inference.latent_function_inference.Laplace()

k1 = GPy.kern.RBF(input_dim=3, variance=0.3, lengthscale=0.2, ARD=False)
k2 = GPy.kern.Bias(input_dim=3, variance=0.3)   # noise
kernel = k1 + k2
print("Kernel Initialized")

m = GPy.core.GP(X=X, Y=Y, likelihood=poisson_likelihood, inference_method=laplace_inf, kernel=kernel)
print("Pre-optimization : ")
print(m)
m.optimize(messages=True, max_f_eval=1000)
print("Optimized :")
print(m)

# mapFile Loading
# 기존에는 2D 지도라 Occupancy Grid Map이 들어왔는데, 지금은 3D 지도라 궁극적으로 OctoMap이 들어와야 할듯
# 아직 지도는 없어서 지도 없이 진행
mapFile = "LUNL_SLAM_map.pgm"
metaFile = "LUNL_SLAM_map.yaml"

FREESPACE = 254
OCCUPIED = 0
UNKNOWN = 205

print("Reading ROS SLAM Map")

try:
    map = cv2.imread(directory+mapFile, -1)
except cv2.error as ecv:
    print(ecv)

print("Map Width (x) : ", map.shape[1], " , Map Height (y) : ", map.shape[0])

with open(directory + metaFile, 'r') as stream:
    try:
        metadata = yaml.safe_load(stream)
    except yaml.YAMLERROR as exc:
        print(exc)

print("Map Metadata : ")
for key in metadata.keys():
    print(key, metadata[key])

freespaceOnly = True;

if freespaceOnly:
    cells = np.where(map == FREESPACE)
    grid = np.array(list(zip(cells[1], (map.shape[0] - cells[0]))))
    grid = grid*metadata["resolution"]
    grid[:, 0] = grid[:, 0] + metadata["origin"][0]
    grid[:, 1] = grid[:, 1] + metadata["origin"][1]
    grid[np.where((grid[:,0]<=2.5) & (grid[:, 1]<=0.0)), :]
else:
    xnew = np.arange(0, map.shape[1])*metadata["resolution"] + metadata["origin"][0]
    ynew = np.arange(0, map.shape[0])*metadata["resolution"] + metadata["origin"][1]
    xv, yv = np.meshgrid(xnew, ynew)
    grid = np.dstack((xv.ravel(), yv.ravel()))[0]

pred_point = grid
print("Generated Grid of Points : " + str(grid.shape[0]))
print("Sampling at Grid Points")

print(pred_point)

### 현재 p(기존의 pred_point)가 z축으로 하나의 값만 갖기 때문에 이를 확장할 필요가 있음
### resolution이 0.05이기 때문에, 0.0부터 1.0까지 0.05의 간격으로 z축을 만들 예정
### numpy 확장 함수 알아보기

'''
### 일단 되는지 확인하기 위해 수작업
z = np.ones((24871, 1))
z *= 0.30
p1 = np.hstack([pred_point, z])
print(p1.shape)

z = np.ones((24871, 1))
z *= 0.35
p2 = np.hstack([pred_point, z])
print(p2.shape)

z = np.ones((24871, 1))
z *= 0.40
p3 = np.hstack([pred_point, z])
print(p3.shape)

z = np.ones((24871, 1))
z *= 0.45
p4 = np.hstack([pred_point, z])
print(p4.shape)

z = np.ones((24871, 1))
z *= 0.50
p5 = np.hstack([pred_point, z])
print(p5.shape)

p = np.vstack((p1,p2,p3,p4,p5))
print(p.shape)
'''


### 3차원 구현 코드
### 아직은 man's power가 필요해서 자동으로 z축 확장하는 방법도 고안이 필요함
### 추가로 현재 resolution이 0.05로 설정해놔서 지도 생성하는데 시간이 꽤 소요됨
### resolution을 0.10 정도로 줄여서 실행해보기
### resolution 문제가 아니라 grid 생성하는 부분을 찾아봐야 할듯
### 추가로 현재 결과를 보면 0에 가까운 공간까지 모두 진한 파랑색으로 칠해서 지도를 이해하기 어려운 단점이 있음
z = np.zeros((24871, 1))
p = np.hstack([pred_point, z])

for i in range(1, 21):
    ### p_i를 만들어서 vstack에 넣기
    ### 최종 결과 p.shape은 (522291, 3)이 나와야 함
    z = np.ones((24871, 1))
    z *= 0.05*i
    p1 = np.hstack([pred_point, z])
    p = np.vstack((p, p1))
print(p.shape)



'''
### 3차원으로 확장하면 여기에서 차원 관련 오류
### 해결
f_mean, f_var = m._raw_predict(p)
f_mean = np.exp(f_mean)

# print(f_mean)
print(max(f_mean))
print(min(f_mean))


### sklearn이랑 결합해서 하는 방법이 있을지 생각
### 없다면 sklearn으로 구현해야 할듯

### 어쩌면 grid에 z축 좌표가 없어서 오류가 발생할 수도
### grid 타입 확인해보고
### z축 임의로 넣어서 결과 보기

# print(pred_point)
# print(len(pred_point))
# print(pred_point.shape)
# print(type(pred_point)) ## class 'numpy.ndarray
#                         ## ndarray는 NumPy의 N차원 배열 객체

xs = p[:, 0]    # x
ys = p[:, 1]    # y
zs = p[:, 2]    # z
color = f_mean

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, zs, c=color, marker='s', s=1, cmap='jet')
plt.xlim([-4,4])
plt.ylim([-8,0])
ax.set_zlim([0,8])
plt.show()

fig2 = plt.figure(figsize=(6,10))
ax2 = fig2.add_subplot(111, projection='3d')
x = np.array(data_raw[:, 0])
y = np.array(data_raw[:, 1])
z = np.array(data_raw[:, 2])
ax2.set_xlim([-5,3])
ax2.set_ylim([-8,0])
ax2.set_zlim([0,8])
ax2.scatter(x,y,z,c=Y, cmap='jet')
plt.show()
'''