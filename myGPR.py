import numpy as np
import pandas
import matplotlib.pyplot as plt
import GPy
import yaml
import cv2
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

directory = "C:/Users/rail/PycharmProjects/GaussianProcessRegression/LUNL/"
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

k1 = GPy.kern.RBF(input_dim=3, variance=0.5, lengthscale=0.8, ARD=False)
k2 = GPy.kern.Bias(input_dim=3, variance=0.6)   # noise
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

### 3차원으로 확장하면 여기에서 차원 관련 오류
#f_mean, f_var = m._raw_predict(pred_point)
#f_mean = np.exp(f_mean)

### sklearn이랑 결합해서 하는 방법이 있을지 생각
### 없다면 sklearn으로 구현해야 할듯

### 어쩌면 grid에 z축 좌표가 없어서 오류가 발생할 수도
### grid 타입 확인해보고
### z축 임의로 넣어서 결과 보기

kernel = 1 * RBF(length_scale=1.0)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gp.fit(X,Y)
print(gp.kernel_)

mean_prediction, std_prediction = gp.predict(pred_point, return_std=True)

xs = X[:, 0]    # x
ys = X[:, 1]    # y
zs = X[:, 2]    # z
color = Y

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, zs, c=color, marker='o', s=15, cmap='jet')

plt.show()