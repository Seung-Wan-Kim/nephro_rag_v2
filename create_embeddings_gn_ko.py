from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def main():
    data_path = "docx_ko/Glumerulonephritis"
    persist_path = "vector_store_gn_ko"

    if not os.path.exists(data_path):
        raise ValueError("Glomerulonephritis 요약문서 디렉토리를 찾을 수 없습니다.")

    # 텍스트 로딩
    documents = []
    for file in os.listdir(data_path):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(data_path, file), encoding="utf-8")
            documents.extend(loader.load())

    # 텍스트 분할 (512 token 기준)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)

    # Ko-SBERT 임베딩 모델 사용
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")

    # FAISS 인덱스 생성
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(persist_path)

    print(f"임베딩 완료: {persist_path} 에 저장되었습니다.")

if __name__ == "__main__":
    main()
