### 드론으로 촬영한 데이터셋을 사용하기 위해 timestamp를 맞춘 데이터를 생성

import pandas as pd
import os


### measurement 변환 ###

f = open("docs/measurements.txt", 'r')

l_time = []
l_measure = []
curr_sec = 43   # measurements.txt 파일이 44, 43.95초에서 시작하기 때문에 시작을 43으로 설정함
measurements = 1
cnt = 1

while True:
    pred_sec = curr_sec
    line = f.readline()
    if not line: break
    #print(line)

    # print(int(line[0:2]))               # 분
    # print(int(line[3:5]))               # 초
    # print(int(line[6:8]))               # 0.xx 초
    # print(int(line.split(',')[-1]))     # 측정값

    curr_sec  = int(line[3:5])      # 현재의 초를 기억(단위는 0.xx초)
    curr_ssec = int(line[6:7])      # 0.x 초


    ### 여기에 0.2초 간격으로 통합하는 방법을 넣기 ###
    # curr_ssec // 0.2 가 0, 1, 2, 3, 4 로 나오는지 확인해보고 -> Yes
    # 그거에 따라 확인해보자
    print("curr_ssec : {}".format(curr_ssec))
    print("curr_ssec // 2 : {}".format(curr_ssec//2))

    temp_pred = temp_curr
    temp_curr = curr_ssec//2
    if temp_pred == temp_curr:  # 이전과 같은 경우
        measurements += int(line.split(',')[-1])
        cnt += 1
    else:
        print("check")

    # if (curr_sec == pred_sec and curr_ssec >= 5) or (curr_ssec < 5):
    if (curr_ssec != 6):        # 해당 데이터셋은 measurement가 0.1초 간격이기 때문에 그냥 6으로 설정해줌
        print("curr_sec: {}, curr_ssec : {}".format(curr_sec, curr_ssec))
        measurements += int(line.split(',')[-1])    # measurements 변수에 현재의 강도를 더해주고
        cnt += 1                                    # cnt도 증가
    else:                       # curr_ssec==6 인 상황인데, 1초 간격을 나누는 기준
        print("check")
        print("curr_sec: {}, curr_ssec : {}".format(curr_sec, curr_ssec))
        result = measurements / cnt                 # 초 단위가 같은 측정 값들의 평균을 계산
        time = int(line[0:2])*60 + int(line[3:5])   # 시간을 맞춰줌(1분을 60초로 만들어서 초 단위의 시간)
        l_time.append(time)
        l_measure.append(round(result))             # 시간과 측정 값들의 평균을 각각 append
        measurements = int(line.split(',')[-1])
        cnt = 0                                     # 변수 초기화

f.close()

print(l_time)
print(l_measure)
'''
ff = open("result_measurement_1.0.txt", 'w')
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

'''
### pose 추출 ###
f = open("docs/pose.txt", 'r')
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

print(len(l_time))  # time
print(len(l_x))     # x
print(len(l_y))     # y
print(len(l_z))     # z

### pose에 없는 시간도 있어서
### pose의 time을 기준으로 result_measurment.txt에 있으면 해당 측정값을 사용

ff = open("docs/radiation_data_1.0.txt", 'w')
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
'''