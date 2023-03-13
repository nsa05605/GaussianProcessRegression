import numpy as np
import pandas as pd

# LUNL 데이터셋 샘플링을 통해 진행했던 결과와 원본의 오차를 확인하기 위한 코드
# index가 동일하기 때문에 좌표를 통해 비교하면 될듯

### LUNL
# # 각 파일의 경로
# directory     = "C:/Users/jihun/PycharmProjects/GaussianProcessRegression/LUNL/"
# gt_file       = "LUNL_radiation_data.txt"
# predict_file1 = "compare_Method3_Invsq_LUNL_2Drad_GPR.txt"
# predict_file2 = "compare_Method3_AddMatExp_LUNL_2Drad_GPR.txt"
# predict_file3 = "compare_Method3_MulMatExp_LUNL_2Drad_GPR.txt"
# predict_file4 = "compare_Method3_Mat32_LUNL_2Drad_GPR.txt"

### JSI
# 각 파일의 경로
directory     = "C:/Users/jihun/PycharmProjects/GaussianProcessRegression/JSI/"
gt_file       = "JSI_radiation_data.txt"
predict_file1 = "compare_Method3_Invsq_JSI_2Drad_GPR.txt"
predict_file2 = "compare_Method3_AddMatExp_JSI_2Drad_GPR.txt"
predict_file3 = "compare_Method3_MulMatExp_JSI_2Drad_GPR.txt"
predict_file4 = "compare_Method3_Mat32_JSI_2Drad_GPR.txt"

# 각 파일에서 x, y, measurements 데이터를 가져옴
data_gt  = np.array(pd.read_csv(directory+gt_file, delimiter=',', header=0, usecols=['x','y','counts']))
data_Inv = np.array(pd.read_csv(directory+predict_file1, delimiter=',', header=0, usecols=['x','y','mean']))
data_Add = np.array(pd.read_csv(directory+predict_file2, delimiter=',', header=0, usecols=['x','y','mean']))
data_Mul = np.array(pd.read_csv(directory+predict_file3, delimiter=',', header=0, usecols=['x','y','mean']))
data_Mat = np.array(pd.read_csv(directory+predict_file4, delimiter=',', header=0, usecols=['x','y','mean']))

# 실제 측정값(gt)와 예측값을 가져옴
Y_gt  = np.array(data_gt[:,2])
Y_Inv = np.array(data_Inv[:,2])
Y_Add = np.array(data_Add[:,2])
Y_Mul = np.array(data_Mul[:,2])
Y_Mat = np.array(data_Mat[:,2])

Y_gt_re  = np.zeros(3000)
Y_Inv_re = np.zeros(3000)
Y_Add_re = np.zeros(3000)
Y_Mul_re = np.zeros(3000)
Y_Mat_re = np.zeros(3000)

cnt = 0
for i in range(len(Y_gt)):
    if(Y_gt[i] > 4):
        Y_gt_re[cnt] = Y_gt[i]
        Y_Inv_re[cnt] = Y_Inv[i]
        Y_Add_re[cnt] = Y_Add[i]
        Y_Mul_re[cnt] = Y_Mul[i]
        Y_Mat_re[cnt] = Y_Mat[i]
        cnt += 1
# print(cnt)
# print(Y_gt_re.shape)
Y_gt_re  = Y_gt_re[:cnt]
Y_Inv_re = Y_Inv_re[:cnt]
Y_Add_re = Y_Add_re[:cnt]
Y_Mul_re = Y_Mul_re[:cnt]
Y_Mat_re = Y_Mat_re[:cnt]

# 전체 데이터에 대해 평균 제곱 오차(MSE)를 계산
error_Inv, error_Add, error_Mul, error_Mat = 0, 0, 0, 0
for i in range(len(Y_gt)):
    error_Inv += (Y_gt[i] - Y_Inv[i])**2
    error_Add += (Y_gt[i] - Y_Add[i])**2
    error_Mul += (Y_gt[i] - Y_Mul[i])**2
    error_Mat += (Y_gt[i] - Y_Mat[i])**2

error_Inv /= len(Y_gt)
error_Add /= len(Y_gt)
error_Mul /= len(Y_gt)
error_Mat /= len(Y_gt)

print("<전체 데이터에 대해 계산>")
print("error_Inv : {}".format(error_Inv))
print("error_Add : {}".format(error_Add))
print("error_Mul : {}".format(error_Mul))
print("error_Mat : {}".format(error_Mat))


# GT가 3 이상인 데이터에 대해 평균 제곱 오차(MSE)를 계산
error_Inv, error_Add, error_Mul, error_Mat = 0, 0, 0, 0
for i in range(len(Y_gt_re)):
    error_Inv += (Y_gt_re[i] - Y_Inv_re[i])**2
    error_Add += (Y_gt_re[i] - Y_Add_re[i])**2
    error_Mul += (Y_gt_re[i] - Y_Mul_re[i])**2
    error_Mat += (Y_gt_re[i] - Y_Mat_re[i])**2

error_Inv /= len(Y_gt_re)
error_Add /= len(Y_gt_re)
error_Mul /= len(Y_gt_re)
error_Mat /= len(Y_gt_re)

print()
print("<GT가 5 이상인 데이터에 대해 계산>")
print("error_Inv : {}".format(error_Inv))
print("error_Add : {}".format(error_Add))
print("error_Mul : {}".format(error_Mul))
print("error_Mat : {}".format(error_Mat))