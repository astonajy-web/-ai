import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

# 1. 페이지 설정
st.set_page_config(page_title="AI 종목 정밀 진단", layout="centered")

# 2. 데이터 및 종목명 수집 함수
@st.cache_data(ttl=3600)
def get_ai_analysis(symbol):
    try:
        # 종목 정보 가져오기 (종목명 추출용)
        ticker_info = yf.Ticker(symbol)
        # 한국 종목명 또는 긴 이름 가져오기
        stock_name = ticker_info.info.get('longName', symbol)
        
        # 데이터 수집 (2023년부터)
        df = yf.download(symbol, start='2023-01-01', multi_level_index=False)
        if df.empty: return None
        
        # 보조지표 생성
        df['Return'] = df['Close'].pct_change()
        df['Vol_Change'] = df['Volume'].pct_change()
        
        delta = df['Close'].diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0; down[down > 0] = 0
        roll_up = up.rolling(14).mean(); roll_down = down.abs().rolling(14).mean()
        df['RSI'] = 100.0 - (100.0 / (1.0 + roll_up / roll_down.replace(0, np.nan)))
        
        # 학습 및 예측
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df = df.dropna()
        features = ['Close', 'Return', 'RSI', 'Vol_Change']
        X = df[features]
        y = df['Target']
        
        model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05).fit(X, y)
        prob = float(model.predict_proba(X.tail(1))[0][1])
        
        recent = df.tail(20)
        return {
            "name": stock_name,
            "p": float(df['Close'].iloc[-1]),
            "sup": float(recent['Low'].min()),
            "res": float(recent['High'].max()),
            "prob": prob
        }
    except:
        return None

# 3. UI 화면
st.title("🚨 AI 종목 정밀 진단")

# 입력창
ticker = st.text_input("종목 코드 (예: 408920.KQ)", value="408920.KQ").upper()
my_price = st.number_input("내 평단가", value=3500)

if st.button("AI 정밀 분석 실행"):
    with st.spinner('데이터와 종목 정보를 가져오는 중...'):
        data = get_ai_analysis(ticker)
        
        if data:
            st.divider()
            # ✨ 종목명 크게 표시
            st.header(f"📌 {data['name']}") 
            
            c1, c2, c3 = st.columns(3)
            c1.metric("현재가", f"{data['p']:,.0f}원")
            c2.metric("매수적정", f"{data['sup']:,.0f}원")
            c3.metric("매도적정", f"{data['res']:,.0f}원")
            
            st.write(f"### 🔮 내일 상승 확률: **{data['prob']:.1%}**")
            st.progress(max(0.0, min(1.0, data['prob'])))
            
            # 맞춤 전략
            loss_rate = (data['p'] - my_price) / my_price if my_price > 0 else 0
            if data['prob'] < 0.3:
                st.error(f"🔴 AI 하락 경고: {data['name']}은(는) 현재 추가 하락 위험이 큽니다.")
            elif data['prob'] > 0.6:
                st.success(f"🟢 AI 반등 예측: {data['name']}의 저점 매수 기회를 노려보세요.")
            else:
                st.warning(f"🟡 보합세: {data['name']}의 추세가 결정될 때까지 기다리세요.")
        else:
            st.error("종목명을 찾을 수 없거나 데이터 오류가 발생했습니다.")
