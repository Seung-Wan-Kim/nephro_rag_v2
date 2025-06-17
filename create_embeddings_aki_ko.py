from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from pathlib import Path

# 설정
DATA_PATH = "docx_ko/aki"  # AKI 문서가 들어있는 폴더
SAVE_PATH = "vector_store_aki_ko"  # 저장할 벡터 저장소 위치
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# 1. 문서 불러오기
def load_documents_from_folder(folder_path):
    documents = []
    for file_path in Path(folder_path).glob("*.txt"):
        loader = TextLoader(str(file_path), encoding="utf-8")
        documents.extend(loader.load())
    return documents

# 2. 텍스트 분할
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)

# 3. 임베딩 모델 설정
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="jhgan/ko-sbert-sts",
        model_kwargs={"device": "cpu"}
    )

# 4. 메인 실행
def main():
    print("Loading documents...")
    documents = load_documents_from_folder(DATA_PATH)

    if not documents:
        raise ValueError("No documents found for AKI embedding.")

    print("Splitting documents...")
    split_docs = split_documents(documents)

    print("Creating FAISS vector store...")
    embedding_model = get_embedding_model()
    vectorstore = FAISS.from_documents(split_docs, embedding_model)

    print(f"Saving vector store to {SAVE_PATH} ...")
    vectorstore.save_local(SAVE_PATH)

    print("✅ AKI embedding complete!")

if __name__ == "__main__":
    main()
