# bitcoin

Dacon 인공지능 비트 트레이더 경진대회 시즌 2 (연간 대회)
https://dacon.io/competitions/official/235712/overview/description/

현재 Running 가능 Code : Arima.py (Dacon Baseline)

Base line Code 바탕

Task : 1380분(23시간)동안의 Information을 바탕으로 다음 120분간의 가격 추이 예측

(1) Regression

(2) 시계열 모듈 활용

(3) 추가적인 보조지표로 Trading(RSI, MACD 등)

Data 분석

현재 Data : Sample_id, time, Coin_index, Open(시가), High(고가), Low(저가), Close(종가), Volume(거래량), quote_av, trades, tb_base_av, tb_quote_av

개인적인 Try : (3), 기본적인 데이터를 바탕으로 RSI, MACD 등 계산하여 푸는 시도

Problem : 현재 "연속적이지 않은 데이터", 즉 Sample_id에 따른 Coin_index 값을 같은것을 이어도 연속적이지 않음.
즉, 특정 기간동안의 결과값을 랜덤 배치했음(우리가 보는 차트처럼 연속적이지 않는다는 문제점) : 차트 분석적인 기법으로 풀기가 어려움(23시간 안에서만 차트 분석 가능)


