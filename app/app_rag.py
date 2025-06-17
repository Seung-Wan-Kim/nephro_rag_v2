import streamlit as st
import pandas as pd

st.set_page_config(page_title="Nephrology RAG System", layout="wide")

st.title("🧪 신장내과 질환 예측 시스템 (Nephro RAG)")
st.markdown("**주요 혈액검사 수치를 입력하고 아래에 질문을 입력해 주세요.**")

# ---- 수치 입력 섹션 ----
st.header("1. 혈액검사 수치 입력")

columns = st.columns(4)
test_items = [
    "BUN", "Creatinine", "eGFR", "Na", "K",
    "Cl", "HCO3-", "Ca", "Phosphorus (P)", "Albumin",
    "Total protein", "Hemoglobin", "PTH", "Vitamin D",
    "ALP", "LDH", "Lactate", "Uric Acid", "Glucose", "CRP"
]

input_values = {}
for i, item in enumerate(test_items):
    with columns[i % 4]:
        input_values[item] = st.number_input(label=item, step=0.1, format="%.2f", key=item)

# ---- 자연어 질의 섹션 ----
st.header("2. 질의 입력")
query = st.text_input("아래에 질문을 입력하세요 (예: 이 수치들은 어떤 질병과 관련 있나요?)")

# ---- 응답 버튼 및 출력 ----
if st.button("질문하기"):
    st.subheader("입력한 검사 수치 요약:")
    df = pd.DataFrame.from_dict(input_values, orient='index', columns=['Value'])
    st.dataframe(df.style.format("{:.2f}"))

    st.subheader("📌 RAG 시스템 응답 예시:")
    st.markdown("> 예: 이 수치는 만성콩팥병(CKD)에서 흔히 관찰되는 패턴입니다. 특히 Creatinine과 eGFR 수치가 CKD G3 수준에 해당할 수 있습니다.")

    st.info("※ 실제 응답은 문서 기반 RAG 시스템 결과로 자동 생성됩니다.")

