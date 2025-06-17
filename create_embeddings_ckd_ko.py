import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import pickle

def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            path = os.path.join(directory, filename)
            loader = TextLoader(path, encoding='utf-8')
            documents.extend(loader.load())
    return documents

def main():
    data_path = "docx_ko/ckd"
    save_path = "vector_store_ckd_ko"
    if not os.path.exists(data_path):
        raise ValueError("CKD 요약문서 디렉토리를 찾을 수 없습니다.")

    print("Loading documents...")
    documents = load_documents(data_path)
    if not documents:
        raise ValueError("No documents found for CKD embedding.")

    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    print("Loading Ko-SBERT model...")
    embeddings = HuggingFaceEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")

    print("Creating FAISS index...")
    db = FAISS.from_documents(docs, embeddings)

    print("Saving FAISS index...")
    db.save_local(save_path)
    with open(os.path.join(save_path, "index.pkl"), "wb") as f:
        pickle.dump(docs, f)

    print("✅ CKD 임베딩 완료")

if __name__ == "__main__":
    main()
