import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
import os

# ---- 설정 ----
VECTOR_STORE_DIR = "vector_store_aki_ko"  # 나중에 다섯 질병군 다 연결할 경우 확장 가능
MODEL_NAME = "jhgan/ko-sbert-nli"

# ---- FAISS 로드 함수 ----
@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    return FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)

# ---- Streamlit 앱 시작 ----
st.set_page_config(page_title="Nephrology Diagnosis Assistant", layout="wide")
st.title("🧪 신장내과 혈액검사 기반 질의응답 시스템")

st.markdown("""
#### 🔹 혈액검사 수치를 입력한 뒤, 진단이나 검사 해석에 대한 질문을 입력해 주세요.
- 검사 수치는 선택적으로 입력하셔도 됩니다.
- 아래에 제시된 항목은 신장질환 관련 핵심 항목들입니다.
""")

# ---- 검사 수치 입력 ----
st.subheader("1️⃣ 혈액검사 수치 입력 (선택 입력)")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    bun = st.text_input("BUN (mg/dL)")
    creatinine = st.text_input("Creatinine (mg/dL)")
    eGFR = st.text_input("eGFR (mL/min/1.73m²)")
    potassium = st.text_input("K (mmol/L)")
with col2:
    sodium = st.text_input("Na (mmol/L)")
    chloride = st.text_input("Cl (mmol/L)")
    CO2 = st.text_input("CO₂ (mEq/L)")
    calcium = st.text_input("Ca (mg/dL)")
with col3:
    phosphorus = st.text_input("Phosphorus (mg/dL)")
    albumin = st.text_input("Albumin (g/dL)")
    hemoglobin = st.text_input("Hemoglobin (g/dL)")
    PTH = st.text_input("PTH (pg/mL)")
with col4:
    vitamin_d = st.text_input("Vitamin D (ng/mL)")
    ALP = st.text_input("ALP (IU/L)")
    LDH = st.text_input("LDH (IU/L)")
    lactate = st.text_input("Lactate (mmol/L)")
with col5:
    proteinuria = st.text_input("단백뇨 (mg/gCr)")
    uric_acid = st.text_input("Uric Acid (mg/dL)")
    glucose = st.text_input("Glucose (mg/dL)")
    WBC = st.text_input("WBC (x10³/µL)")

# ---- 질문 입력 ----
st.subheader("2️⃣ 질문 입력")
user_question = st.text_area("궁금한 점을 입력하세요.", height=100)

# ---- 벡터스토어 및 응답 생성 ----
if st.button("질문하기"):
    if not user_question.strip():
        st.warning("질문을 입력해주세요.")
    else:
        with st.spinner("질문을 처리 중입니다..."):
            vector_store = load_vector_store()
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

            llm = ChatOpenAI(model_name="gpt-4", temperature=0.3)
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )

            result = qa(user_question)
            answer = result["result"]
            sources = result.get("source_documents", [])

        st.subheader("🧠 답변")
        st.write(answer)

        if sources:
            st.markdown("---")
            st.subheader("📄 참고 문서")
            for i, doc in enumerate(sources):
                st.markdown(f"**문서 {i+1}:** `{os.path.basename(doc.metadata['source'])}`")
