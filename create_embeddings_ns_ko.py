from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os
import pickle

def load_documents(data_path):
    documents = []
    for filename in os.listdir(data_path):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(data_path, filename), encoding='utf-8')
            documents.append(loader.load()[0])
    return documents

def main():
    data_path = "docx_ko/Nephrotic syndrome"
    if not os.path.isdir(data_path):
        raise ValueError("Nephrotic Syndrome 요약문서 디렉토리를 찾을 수 없습니다.")

    print("Loading documents...")
    raw_documents = load_documents(data_path)

    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    documents = text_splitter.split_documents(raw_documents)

    print("Creating FAISS vector store...")
    embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")
    db = FAISS.from_documents(documents, embedding_model)

    save_path = "vector_store_ns_ko"
    db.save_local(save_path)
    with open(os.path.join(save_path, "index.pkl"), "wb") as f:
        pickle.dump(db, f)

    print("Nephrotic Syndrome 벡터 스토어 저장 완료.")

if __name__ == "__main__":
    main()
