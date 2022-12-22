
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
  - 드론으로 제작한 데이터셋(https://github.com/ntnu-arl/radiation-source-localization-dataset)을 활용해서 진행
  - bag 파일 내부에서 pose와 measurements의 timestamp가 다른 문제가 있어서, extract_rad_data.py 파일을 통해 데이터 처리
    - measurements.txt 파일에서도 초 단위로 측정 값을 모으고, 평균을 내서 해당 초의 대표 값으로 설정
    - pose.txt 파일이 더 적은 범위의 timestamp를 갖고 있기 때문에 기준으로 삼고 00초에 가장 가까운 time을 추출
    - 결과 pose에 있는 시간과 result_measurement.txt에 있는 시간이 동일한 경우, 해당 위치의 측정 값으로 사용
  - myGPR_drone.py : 드론으로 제작한 방사선 데이터셋 기준