import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os

# ----------------------------
# 설정
# ----------------------------
st.set_page_config(page_title="Nephrology RAG", layout="wide")
st.title("🩺 신장내과 질환 문서 기반 질의응답")

# ----------------------------
# 사용자 입력
# ----------------------------
disease_group = st.selectbox(
    "질병군을 선택하세요:",
    ("AKI", "CKD", "Nephrotic Syndrome", "Glomerulonephritis", "Electrolyte Disorders")
)

user_question = st.text_input("질문을 입력하세요:", placeholder="예: 이 환자의 AKI 치료는 어떻게 진행해야 하나요?")

# ----------------------------
# 벡터스토어 경로 설정
# ----------------------------
vectorstore_path_map = {
    "AKI": "vector_store_aki_ko",
    "CKD": "vector_store_ckd_ko",
    "Nephrotic Syndrome": "vector_store_ns_ko",
    "Glomerulonephritis": "vector_store_gn_ko",
    "Electrolyte Disorders": "vector_store_el_ko",
}

# ----------------------------
# 검색 및 응답 처리 함수
# ----------------------------
def generate_answer(question, vectorstore_dir):
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")
    vectordb = FAISS.load_local(vectorstore_dir, embeddings, allow_dangerous_deserialization=True)

    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0.3),
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
    )

    result = qa_chain({"query": question})
    return result["result"]

# ----------------------------
# 질의 버튼
# ----------------------------
if st.button("질문하기"):
    if not user_question:
        st.warning("질문을 입력해주세요.")
    else:
        with st.spinner("질문을 처리 중입니다..."):
            vector_path = vectorstore_path_map[disease_group]
            try:
                answer = generate_answer(user_question, vector_path)
                st.success("📘 답변:")
                st.write(answer)
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")
