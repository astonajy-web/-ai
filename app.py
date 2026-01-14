import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier # XGBoost보다 로딩이 빠름

# 1. 페이지 설정 (최상단)
st.set_page_config(page_title="초고속 AI 진단", layout="centered")

# 2. 데이터 수집 및 분석 함수 (캐싱 적용으로 속도 업)
@st.cache_data(ttl=3600) # 1시간 동안 결과 기억
def fast_analyze(symbol):
    try:
        # 데이터 수집 (최근 1년치로 제한하여 속도 향상)
        df = yf.download(symbol, period='1y', interval='1d', multi_level_index=False)
        if df.empty: return None
        
        # 지지/저항 계산 (벡터 연산으로 고속 처리)
        recent_20 = df.tail(20)
        support = float(recent_20['Low'].min())
        resistance = float(recent_20['High'].max())
        current_p = float(df['Close'].iloc[-1])
        
        # 가벼운 모델 학습
        df['Return'] = df['Close'].pct_change()
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        train_df = df.dropna()
        
        X = train_df[['Close', 'Return']].values
        y = train_df['Target'].values
        
        # 가벼운 RandomForest 사용 (서버 부담 최소화)
        model = RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42)
        model.fit(X, y)
        
        # 마지막 행으로 예측
        last_features = np.array([[current_p, df['Return'].iloc[-1]]])
        prob = model.predict_proba(last_features)[0][1]
        
        return {
            "current_p": current_p,
            "support": support,
            "resistance": resistance,
            "prob": prob
        }
    except Exception as e:
        return None

# --- UI 부분 ---
st.title("⚡ 초고속 AI 투자 진단")

ticker = st.text_input("종목 코드 (예: 408920.KQ)", value="408920.KQ").upper()
my_price = st.number_input("내 평단가", value=0)

if st.button("즉시 분석"): # 버튼을 눌렀을 때만 실행되게 하여 불필요한 재계산 방지
    with st.spinner('AI가 1초 만에 분석 중...'):
        res = fast_analyze(ticker)
        
        if res:
            st.divider()
            # 주요 수치 가로 배치
            cols = st.columns(3)
            cols[0].metric("현재가", f"{res['current_p']:,.0f}")
            cols[1].metric("매수적정", f"{res['support']:,.0f}")
            cols[2].metric("매도적정", f"{res['resistance']:,.0f}")
            
            # 게이지 형태의 확률 표시
            st.progress(res['prob'])
            st.write(f"🔮 AI 상승 확신도: **{res['prob']:.1%}**")
            
            # 전략 리포트
            st.subheader("💡 행동 지침")
            if my_price > 0:
                loss_rate = (res['current_p'] - my_price) / my_price
                if res['prob'] > 0.6 and loss_rate < -0.05:
                    st.success("💎 **물타기 적기:** 확률이 높고 현재 바닥권입니다.")
                elif res['prob'] < 0.4:
                    st.error("✋ **관망 요망:** 에너지가 부족합니다. 더 기다리세요.")
                else:
                    st.info("⚖️ **보유 유지:** 큰 움직임 전까지 대기하세요.")
            
            # 가격대 도달 알림
            if res['current_p'] <= res['support'] * 1.02:
                st.warning(f"🎯 지지선({res['support']:,.0f}원) 근처입니다! 반등 확인 후 진입 고려.")
        else:
            st.error("종목을 찾을 수 없습니다.")

st.divider()
st.caption("데이터 제공: Yahoo Finance / 분석 모델: 초경량 RF Classifier")
