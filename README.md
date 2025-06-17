# Nephrology RAG v2

이 프로젝트는 신장내과 주요 질병군(AKI, CKD, Nephrotic Syndrome, Glomerulonephritis, Electrolyte Disorders)에 대한 질의응답 시스템입니다. 문서 기반 임베딩과 Ko-SBERT 모델을 활용한 RAG(Retrieval-Augmented Generation) 방식으로 구성되어 있습니다.

## 📁 주요 구성

- `app/app_rag.py`: Streamlit 기반 UI
- `create_embeddings_*.py`: 각 질병군별 임베딩 생성 스크립트
- `docs_ko/`: 질병군별 한글 요약문서
- `vector_store_*/`: FAISS 기반 벡터 저장소

## 🧪 사용 방법

```bash
pip install -r requirements.txt
streamlit run app/app_rag.py
