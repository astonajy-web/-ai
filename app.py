import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

# 1. 페이지 설정
st.set_page_config(page_title="AI 종목 구조대 Pro", layout="centered")

# 2. 정밀 분석 함수 (이전의 높은 정확도 로직 복구)
@st.cache_data(ttl=3600)
def get_ai_analysis(symbol):
    try:
        # 2023년부터 데이터를 읽어와서 '맥락'을 파악 (정확도의 핵심)
        df = yf.download(symbol, start='2023-01-01', multi_level_index=False)
        if df.empty: return None
        
        # 지표 생성 (RSI, 변동성 등 정밀 지표 포함)
        df['Return'] = df['Close'].pct_change()
        df['Vol_Change'] = df['Volume'].pct_change()
        
        delta = df['Close'].diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0; down[down > 0] = 0
        roll_up = up.rolling(14).mean(); roll_down = down.abs().rolling(14).mean()
        df['RSI'] = 100.0 - (100.0 / (1.0 + roll_up / roll_down.replace(0, np.nan)))
        
        # 타겟 설정 및 학습
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df = df.dropna()
        
        # 정밀 분석용 특징들
        features = ['Close', 'Return', 'RSI', 'Vol_Change']
        X = df[features]
        y = df['Target']
        
        # XGBoost 모델 가동
        model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05).fit(X, y)
        prob = model.predict_proba(X.tail(1))[0][1]
        
        # 지지/저항가 계산
        recent = df.tail(20)
        return {
            "p": float(df['Close'].iloc[-1]),
            "sup": float(recent['Low'].min()),
            "res": float(recent['High'].max()),
            "prob": prob,
            "rsi": float(df['RSI'].iloc[-1])
        }
    except:
        return None

# 3. 화면 구성
st.title("🚨 AI 종목 정밀 진단")
ticker = st.text_input("종목 코드 (예: 408920.KQ)", value="408920.KQ").upper()
my_price = st.number_input("내 평단가", value=3500)

if st.button("AI 정밀 분석 실행"):
    with st.spinner('정밀 엔진(XGBoost) 가동 중...'):
        data = get_ai_analysis(ticker)
        
        if data:
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("현재가", f"{data['p']:,.0f}원")
            c2.metric("매수적정", f"{data['sup']:,.0f}원")
            c3.metric("매도적정", f"{data['res']:,.0f}원")
            
            # 상승 확률 표시
            st.write(f"### 🔮 내일 상승 확률: **{data['prob']:.1%}**")
            st.progress(data['prob'])
            
            # 맞춤 조언
            loss_rate = (data['p'] - my_price) / my_price if my_price > 0 else 0
            
            if data['prob'] < 0.3:
                st.error(f"🔴 **위험:** AI가 하락 신호를 강하게 보내고 있습니다. 평단 {my_price:,}원 도달까지 시간이 필요해 보입니다. 추가 매수는 금물입니다.")
            elif data['prob'] > 0.6:
                st.success(f"🟢 **기회:** 반등 신호가 포착되었습니다. {data['sup']:,.0f}원 근처에서 물타기를 고려하세요.")
            else:
                st.warning("🟡 **관망:** 현재는 방향성이 뚜렷하지 않습니다. 지지선을 지키는지 지켜보세요.")
        else:
            st.error("데이터 수집에 실패했습니다.")
