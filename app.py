import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

# 1. 페이지 설정
st.set_page_config(page_title="AI 종목 정밀 진단", layout="centered")

# 2. 분석 함수
@st.cache_data(ttl=3600)
def get_ai_analysis(symbol):
    try:
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
        
        # 타겟 설정 및 학습
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df = df.dropna()
        
        features = ['Close', 'Return', 'RSI', 'Vol_Change']
        X = df[features]
        y = df['Target']
        
        # 모델 학습
        model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05).fit(X, y)
        
        # 확률 추출 (에러 방지를 위해 확실하게 float으로 변환)
        prob_array = model.predict_proba(X.tail(1))[0][1]
        prob = float(prob_array) # 이 부분이 에러를 해결합니다.
        
        # 지지/저항가
        recent = df.tail(20)
        return {
            "p": float(df['Close'].iloc[-1]),
            "sup": float(recent['Low'].min()),
            "res": float(recent['High'].max()),
            "prob": prob
        }
    except Exception as e:
        return None

# 3. 화면 UI
st.title("🚨 AI 종목 정밀 진단")

ticker = st.text_input("종목 코드 (예: 408920.KQ)", value="408920.KQ").upper()
my_price = st.number_input("내 평단가", value=3500)

if st.button("AI 정밀 분석 실행"):
    with st.spinner('정밀 엔진 가동 중...'):
        data = get_ai_analysis(ticker)
        
        if data:
            st.divider()
            # 숫자 출력
            c1, c2, c3 = st.columns(3)
            c1.metric("현재가", f"{data['p']:,.0f}원")
            c2.metric("매수적정", f"{data['sup']:,.0f}원")
            c3.metric("매도적정", f"{data['res']:,.0f}원")
            
            # 상승 확률 (에러 발생했던 지점 수정 완료)
            st.write(f"### 🔮 내일 상승 확률: **{data['prob']:.1%}**")
            
            # st.progress는 0.0 ~ 1.0 사이의 float만 받습니다.
            clamped_prob = max(0.0, min(1.0, data['prob']))
            st.progress(clamped_prob)
            
            # 전략 리포트
            loss_rate = (data['p'] - my_price) / my_price if my_price > 0 else 0
            if data['prob'] < 0.3:
                st.error(f"🔴 AI 하락 경고: 현재 관망이 유리합니다. 추가 매수 금지.")
            elif data['prob'] > 0.6:
                st.success(f"🟢 AI 반등 예측: 지지선({data['sup']:,.0f}원) 근처에서 대응하세요.")
            else:
                st.warning("🟡 보합세: 방향성이 나올 때까지 대기하세요.")
        else:
            st.error("데이터를 가져오는 중 오류가 발생했습니다. 종목 코드를 확인하세요.")
