from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os
import pickle

def main():
    data_path = "docx_ko/Electrolyte disorders"  # 사용자가 파일을 저장한 폴더 경로로 수정
    if not os.path.exists(data_path):
        raise ValueError("Electrolyte disorders 요약문서 디렉토리를 찾을 수 없습니다.")

    # 문서 불러오기
    documents = []
    for file_name in os.listdir(data_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(data_path, file_name)
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())

    # 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # KoSBERT 임베딩 모델
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")

    # 벡터스토어 생성
    db = FAISS.from_documents(docs, embeddings)

    # 저장
    db.save_local("vector_store_el_ko")
    with open("vector_store_el_ko/index.pkl", "wb") as f:
        pickle.dump(docs, f)

if __name__ == "__main__":
    main()
