import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import plotly.graph_objects as go # ✨ Plotly 라이브러리 추가

# 1. 페이지 설정
st.set_page_config(page_title="AI 종목 정밀 진단 + 차트", layout="centered")

# 2. 데이터 및 종목명 수집 함수 (차트 데이터도 함께 가져옴)
@st.cache_data(ttl=3600)
def get_ai_analysis_with_chart(symbol):
    try:
        # 종목 정보 가져오기 (종목명 추출용)
        ticker_info = yf.Ticker(symbol)
        stock_name = ticker_info.info.get('longName', symbol)
        
        # 데이터 수집 (최근 1년 데이터로 차트 효율화)
        df = yf.download(symbol, period='1y', interval='1d', multi_level_index=False)
        if df.empty: return None
        
        # 보조지표 생성 (AI 분석용)
        df['Return'] = df['Close'].pct_change()
        df['Vol_Change'] = df['Volume'].pct_change()
        
        delta = df['Close'].diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0; down[down > 0] = 0
        roll_up = up.rolling(14).mean(); roll_down = down.abs().rolling(14).mean()
        df['RSI'] = 100.0 - (100.0 / (1.0 + roll_up / roll_down.replace(0, np.nan)))
        
        # 학습 및 예측
