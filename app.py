import streamlit as st
import yfinance as yf
from xgboost import XGBClassifier
import pandas as pd

# 모바일 화면 설정
st.set_page_config(page_title="메쎄이상 구조대", layout="centered")

st.title("🚨 AI 주식 구조대")
st.subheader("메쎄이상 (408920.KQ)")

# 1. 사용자 입력 (모바일에서 터치하기 쉽게)
my_price = st.number_input("내 평단가를 입력하세요", value=3500)
st.divider()

# 2. AI 분석 로직
@st.cache_data(ttl=3600) # 1시간마다 데이터 갱신
def get_analysis():
    df = yf.download('408920.KQ', start='2023-01-01', multi_level_index=False)
    # ... (여기에 우리가 만든 정확도 55% 모델 로직이 들어갑니다) ...
    current_p = df['Close'].iloc[-1]
    # 가상의 확률 계산 (실제 모델 결과값 연결)
    prob = 0.31 # 예시값
    return current_p, prob

current_p, prob = get_analysis()
loss_rate = (current_p - my_price) / my_price

# 3. 모바일용 대형 카드 출력
col1, col2 = st.columns(2)
col1.metric("현재가", f"{current_p:,}원", f"{loss_rate:.2%}", delta_color="inverse")
col2.metric("상승 확률", f"{prob*100:.1f}%")

st.divider()

# 4. 직관적인 행동 지침
if prob < 0.4:
    st.error(f"❌ 지금은 '관망' 하세요! (하락 위험 높음)")
    st.warning("평단가 3,500원까지는 인내심이 필요합니다.")
elif prob > 0.6:
    st.success(f"✅ '추가 매수' 적기입니다! (반등 확률 높음)")
else:
    st.info(f"🟡 현재는 '보유' 구간입니다.")