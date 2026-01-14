import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

st.set_page_config(page_title="AI 투자 적정가 진단기", layout="centered")

st.title("🎯 AI 매수/매도 적정가 진단")

# 1. 입력 섹션
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("종목 코드 (예: 408920.KQ)", value="408920.KQ")
    with col2:
        my_price = st.number_input("내 평단가 (0이면 신규진입)", value=0)

@st.cache_data(ttl=3600)
def get_full_analysis(symbol):
    try:
        df = yf.download(symbol, start='2024-01-01', multi_level_index=False)
        if df.empty: return None
        
        # 지지/저항 계산 (최근 20일 기준)
        recent_df = df.tail(20)
        support = recent_df['Low'].min()   # 최근 최저점 = 매수 적정가
        resistance = recent_df['High'].max() # 최근 최고점 = 매도 적정가
        current_p = df['Close'].iloc[-1]
        
        # AI 모델 (내일 상승 확률)
        df['Return'] = df['Close'].pct_change()
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df = df.dropna()
        X = df[['Close', 'Return']]
        y = df['Target']
        model = XGBClassifier().fit(X, y)
        prob = model.predict_proba(X.tail(1))[0][1]
        
        return {
            "current_p": current_p,
            "support": support,
            "resistance": resistance,
            "prob": prob
        }
    except:
        return None

if ticker:
    res = get_full_analysis(ticker)
    
    if res:
        # 주요 수치 브리핑
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("현재 가격", f"{res['current_p']:,.0f}원")
        c2.metric("매수 적정(지지)", f"{res['support']:,.0f}원", delta="바닥권", delta_color="normal")
        c3.metric("매도 적정(저항)", f"{res['resistance']:,.0f}원", delta="목표가", delta_color="inverse")

        # AI 판단 대형 카드
        st.subheader("🤖 AI 종합 투자 가이드")
        
        # 1. 가격 전략 (Price Strategy)
        if res['current_p'] <= res['support'] * 1.03: # 바닥에서 3% 이내일 때
            st.success(f"📍 **지금이 매수 적기!** 현재 가격이 바닥권({res['support']:,.0f}원)에 매우 근접했습니다.")
        elif res['current_p'] >= res['resistance'] * 0.97: # 천장에서 3% 이내일 때
            st.error(f"📍 **지금은 매도 타이밍!** 천장({res['resistance']:,.0f}원) 근처입니다. 익절을 고려하세요.")
        else:
            st.info("📍 **중간 지대입니다.** 서두르지 말고 지지선까지 눌릴 때를 기다리거나, 돌파를 확인하세요.")

        # 2. 내 계좌 맞춤 전략
        if my_price > 0:
            loss_rate = (res['current_p'] - my_price) / my_price
            st.markdown(f"---")
            st.markdown(f"**내 계좌 현황:** 수익률 {loss_rate:.2%}")
            
            if loss_rate < -0.1 and res['prob'] > 0.55:
                st.warning(f"💡 **구조 신호:** 평단을 낮추고 싶다면 {res['support']:,.0f}원 근처에서 추가 매수하세요.")
            elif loss_rate > 0.05:
                st.balloons()
                st.success(f"💰 **수익 관리:** {res['resistance']:,.0f}원 도달 시 전량 또는 분할 매도를 추천합니다.")
        
        st.divider()
        st.write(f"📊 **AI 분석 데이터:** 상승 확률 {res['prob']:.1%} | 최근 20일 변동폭 기준")
    else:
        st.error("데이터를 가져오지 못했습니다. 코드를 확인해 주세요.")
