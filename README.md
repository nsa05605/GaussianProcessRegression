
# GaussianProcessRegression
### Andrew West의 "Use of Gaussian Process Regression for radiation mapping of a nuclear reactor with a mobile robot" 논문의 코드 참고

- 가우시안 프로세스 라이브러리인 GPy를 사용한 2차원 지도 작성

- 추후 3차원 지도 작성으로 확장할 계획

-------------

### 2022/12/16
- sklearn의 Gaussian Process를 사용하기 위해 코드 분석을 진행

### 2022/12/21
- 3차원 방사선 지도 제작
  - myGPR.py : LUNL dataset 기준
  - myGPR_JSI.py : JSI dataset 기준인데 데이터셋이 커서 진행 중 오류 발생

### 2022/12/22
- 3차원 방사선 지도 제작
  - 드론으로 제작한 데이터셋(https://github.com/ntnu-arl/radiation-source-localization-dataset) 을 활용해서 진행
  - bag 파일 내부에서 pose와 measurements의 timestamp가 다른 문제가 있어서, extract_rad_data.py 파일을 통해 데이터 처리
    - measurements.txt 파일에서도 초 단위로 측정 값을 모으고, 평균을 내서 해당 초의 대표 값으로 설정
    - pose.txt 파일이 더 적은 범위의 timestamp를 갖고 있기 때문에 기준으로 삼고 00초에 가장 가까운 time을 추출
    - 결과 pose에 있는 시간과 result_measurement.txt에 있는 시간이 동일한 경우, 해당 위치의 측정 값으로 사용
  - myGPR_drone.py : 드론으로 제작한 방사선 데이터셋 기준

### 2023/03/13
- 중간에 비어 있는 기간에는 OctoMap에 시각화하는 부분과 방사선 데이터에 적합한 커널을 찾는 작업을 진행함
- 제작한 InvSquare 커널을 사용하여 예측한 값과 실제 데이터와 비교를 진행
- GPR_Analysis_LUNL_sampling.py 파일과 GPR_Analysis_JSI_sampling.py 파일

### 2023/03/14
- 이전에 진행했던 2차원 샘플링에 이어 3차원 데이터에 대해 샘플링을 하기 위해 데이터를 1초, 0.2초 간격으로 통합하는 작업을 진행 중
- extract_rad_data_1.0.py 파일과 extract_rad_data_0.2.py 파일