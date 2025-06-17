import streamlit as st

# 검사 항목 리스트 (공통 및 중요 항목 기준)
lab_tests = [
    "Creatinine", "BUN", "eGFR", "Na", "K",
    "Cl", "CO2(HCO3-)", "Ca", "Phosphorus(IP)",
    "Albumin", "Total Protein", "LDH", "CRP",
    "Hb", "Hematocrit", "WBC", "Platelet",
    "Glucose", "Uric Acid", "Magnesium"
]

st.set_page_config(page_title="Nephrology RAG System", layout="wide")

st.title("🩺 신장내과 질환 진단 RAG 시스템")
st.markdown("주요 혈액검사 수치를 입력하고 질문을 입력해 주세요.")

# 혈액검사 수치 입력 (5열 고정)
st.subheader("🧪 혈액검사 수치 입력")
cols = st.columns(5)
input_values = {}
for i, test in enumerate(lab_tests):
    with cols[i % 5]:
        value = st.text_input(f"{test}", key=test)
        input_values[test] = value

# 자연어 질문 입력
st.markdown("---")
st.subheader("💬 질의응답")
query = st.text_input("질문을 입력하세요 (예: '이 수치로 AKI 가능성이 있나요?')", key="query")

if st.button("🔍 질의하기"):
    st.write("✅ 입력한 검사 수치:")
    st.json(input_values)

    st.write("✅ 질문 내용:")
    st.write(query)

    st.info("※ 현재는 프론트엔드 UI 시연 단계입니다. 질의응답 기능은 추후 연결됩니다.")
