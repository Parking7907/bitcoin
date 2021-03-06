# Bitcoin Trading

>Dacon 인공지능 비트 트레이더 경진대회 시즌 2 (연간 대회) <br/>
>https://dacon.io/competitions/official/235712/overview/description/

---

시계열 문제라는 해석  
현재 Run 가능 Code : Arima.py (Dacon Baseline)

Base line Code 바탕

Task : 1380분(23시간)동안의 Information을 바탕으로 다음 120분간의 가격 추이 예측

---

# 풀이 방식

(1) Regression

(2) 시계열 모듈 활용

(3) 추가적인 보조지표로 Trading(RSI, MA, MACD 등)

---

# Data 분석

*현재 주어지는 시계열 Data : 
  + Sample_id, time, Coin_index, Open(시가), High(고가), Low(저가), Close(종가)
    - 일반적인 주식 거래량과 비슷
  + Volume(거래량), quote_av, trades, tb_base_av, tb_quote_av
    - 일반적이지 않은 Coin에서의 차이점 "TAKER BUY"라는 개념 

개인적인 Try : (3), 기본적인 데이터를 바탕으로 RSI, MACD 등 계산하여 푸는 시도

: lstm_preprocess.py Code로 
MA5, MA20, MA60, MA120등을 구하려는 시도 (But, 현재의 문제점은 일별 변화량을 추적하는것이 아니라 "분별 변화량"을 추적하는 상황

__Problem :__ 현재 주어진 가상화폐들은 "연속적이지 않은 데이터", 즉 Sample_id에 따른 Coin_index 값을 같은것을 이어도 연속적이지 않음.
즉, 특정 기간동안의 결과값을 랜덤 배치했음(우리가 보는 차트처럼 연속적이지 않는다는 문제점) : 차트 분석적인 기법으로 풀기가 어려움(23시간 안에서만 차트 분석 가능)


---


# Dependency
### 1. pandas 1.2.3

### 2. python 3.8.8

### 3. matplotlib 3.3.4

### 4. pytorch 1.8.0+cu111

(Not used, but codes like LSTM may use deep learning later, Server is Based on GTX 3090)

libtiff                   4.2.0                h3942068_0<br/>
libuuid                   1.0.3                h1bed415_2<br/>
libwebp-base              1.2.0                h27cfd23_0<br/>
libxcb                    1.14                 h7b6447c_0<br/>
libxml2                   2.9.10               hb55368b_3<br/>
lunarcalendar             0.0.9                    pypi_0    pypi<br/>
lxml                      4.6.2                    pypi_0    pypi<br/>
lz4-c                     1.9.3                h2531618_0<br/>
matplotlib                3.3.4            py38h06a4308_0<br/>
matplotlib-base           3.3.4            py38h62a2d02_0<br/>
mkl                       2020.2                      256<br/>
mkl-service               2.3.0            py38he904b0f_0<br/>
mkl_fft                   1.3.0            py38h54f3939_0<br/>
mkl_random                1.1.1            py38h0573a6f_0<br/>
ncurses                   6.2                  he6710b0_1<br/>
numpy                     1.20.1                   pypi_0    pypi<br/>
numpy-base                1.19.2           py38hfa32c7d_0<br/>
olefile                   0.46                       py_0<br/>
openssl                   1.1.1j               h27cfd23_0<br/>
pandas                    1.2.3            py38ha9443f7_0<br/>
pandas-datareader         0.9.0                    pypi_0    pypi<br/>
patsy                     0.5.1                    pypi_0    pypi<br/>
pcre                      8.44                 he6710b0_0<br/>
pillow                    8.1.2            py38he98fc37_0<br/>
pip                       21.0.1           py38h06a4308_0<br/>
pmdarima                  1.8.0                    pypi_0    pypi<br/>
pymeeus                   0.5.10                   pypi_0    pypi<br/>
pyparsing                 2.4.7              pyhd3eb1b0_0<br/>
pyqt                      5.9.2            py38h05f1152_4<br/>
pystan                    2.19.1.1                 pypi_0    pypi<br/>
python                    3.8.8                hdb3f193_4<br/>
python-dateutil           2.8.1              pyhd3eb1b0_0<br/>
pytz                      2021.1             pyhd3eb1b0_0<br/>
qt                        5.9.7                h5867ecd_1<br/>
readline                  8.1                  h27cfd23_0<br/>
requests                  2.25.1                   pypi_0    pypi<br/>
scikit-learn              0.24.1                   pypi_0    pypi<br/>
scipy                     1.6.1                    pypi_0    pypi<br/>
setuptools                52.0.0           py38h06a4308_0<br/>
setuptools-git            1.2                      pypi_0    pypi<br/>
sip                       4.19.13          py38he6710b0_0<br/>
six                       1.15.0           py38h06a4308_0<br/>
sqlite                    3.35.0               hdfb4753_0<br/>
statsmodels               0.12.2                   pypi_0    pypi<br/>
threadpoolctl             2.1.0                    pypi_0    pypi<br/>
tk                        8.6.10               hbc83047_0<br/>
torch                     1.8.0+cu111              pypi_0    pypi<br/>
torchaudio                0.8.0                    pypi_0    pypi<br/>
torchvision               0.9.0+cu111              pypi_0    pypi<br/>
tornado                   6.1              py38h27cfd23_0  
tqdm                      4.59.0                   pypi_0    pypi  
typing-extensions         3.7.4.3                  pypi_0    pypi  
urllib3                   1.26.4                   pypi_0    pypi  
wheel                     0.36.2             pyhd3eb1b0_0  
xz                        5.2.5                h7b6447c_0  
zlib                      1.2.11               h7b6447c_3  
zstd                      1.4.5                h9ceee32_0  
