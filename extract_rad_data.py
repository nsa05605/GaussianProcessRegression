### 드론으로 촬영한 데이터셋을 사용하기 위해 timestamp를 맞춘 데이터를 생성

import pandas as pd
import os

'''
### measurement 변환 ###

f = open("docs/measurements.txt", 'r')

l_time = []
l_measure = []
curr_sec = 43   # measurements.txt 파일이 44, 43.95초에서 시작하기 때문에 시작을 43으로 설정함
measurements = 0
cnt = 1

while True:
    pred_sec = curr_sec
    line = f.readline()
    if not line: break
    #print(line)

    # print(int(line[0:2]))               # 분
    # print(int(line[3:5]))               # 초
    # 아마 int(line[6:8]) 하면 0.00초 단위로 나올듯
    # print(int(line.split(',')[-1]))     # 측정값

    curr_sec = int(line[3:5])        # 현재의 초를 기억(단위는 초)

    if pred_sec == curr_sec:    # 이전 step과 초 단위가 같으면
        measurements += int(line.split(',')[-1])    # measurements 변수에 현재의 강도를 더해줌
        cnt += 1                                    # cnt도 증가
    else:                       # 이전 step과 초 단위가 달라지면
        result = measurements / cnt                 # 초 단위가 같은 측정 값들의 평균을 계산
        time = int(line[0:2])*60 + int(line[3:5])   # 시간을 맞춰줌(1분을 60초로 만들어서 초 단위의 시간)
        l_time.append(time)
        l_measure.append(round(result))             # 시간과 측정 값들의 평균을 각각 append
        measurements = 0
        cnt = 0                                     # 변수 초기화

f.close()

print(l_time)
print(l_measure)

ff = open("result_measurement.txt", 'w')
for i in range(350):
    data = str(l_time[i])
    ff.write(data)
    data = ', '
    ff.write(data)
    data = str(l_measure[i])
    ff.write(data)
    data = '\n'
    ff.write(data)
ff.close()
'''


### pose 추출 ###
f = open("pose.txt", 'r')
m = open("result_measurement.txt", 'r')

### 시간과 좌표 담을 list
l_time = []
l_x = []
l_y = []
l_z = []

curr_sec = 48   # 얘도 마찬가지로 pose 파일의 시작이 45, 48.83이라 48로 설정

while True:
    pred_sec = curr_sec
    line = f.readline()
    if not line: break

    # print(line)
    # print(int(line[14:16]))  # 분
    # print(int(line[17:19]))  # 초
    # print(int(line[20:22]))  # 0.x 초
    # print(line.split((','))[5])  # x
    # print(line.split((','))[6])  # y
    # print(line.split((','))[7])  # z

    curr_sec  = int(line[17:19]) # 현재 초를 기억
    curr_ssec = int(line[20:22]) # .xx 초

    if curr_sec == pred_sec:
        continue
    else:   # 이전과 초 단위가 달라지는 순간
        time = int(line[14:16])*60 + int(line[17:19])
        l_time.append(time)
        l_x.append(float(line.split((','))[5]))
        l_y.append(float(line.split((','))[6]))
        l_z.append(float(line.split((','))[7]))

# print(len(l_time))  # time
# print(len(l_x))     # x
# print(len(l_y))     # y
# print(len(l_z))     # z

### pose에 없는 시간도 있어서
### pose의 time을 기준으로 result_measurment.txt에 있으면 해당 측정값을 사용

ff = open("radiation_data.txt", 'w')
for i in range(119):
    while True:
        line_m = m.readline()
        if int(l_time[i]) == int(line_m.split((','))[0]):
            print(i)
            data = str(l_time[i])
            ff.write(data)
            data = ', '
            ff.write(data)
            data = str(l_x[i])
            ff.write(data)
            data = ', '
            ff.write(data)
            data = str(l_y[i])
            ff.write(data)
            data = ', '
            ff.write(data)
            data = str(l_z[i])
            ff.write(data)
            data = ','
            ff.write(data)
            data = str(line_m.split((','))[1])
            ff.write(data)
            break

ff.close()