"""OpenAI Embeddings와 ChromaDB를 사용하는 화성시 민원 RAG 챗봇."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import chromadb
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"
MANIFEST = DB_DIR / "index_manifest.json"
COLLECTION = "hwaseong_civil_documents"
NO_ANSWER = "자료에서 확인할 수 없습니다"
EMBEDDING_MODEL = "text-embedding-3-small"
ANSWER_MODEL = "gpt-4.1-mini"
# Korean questions can be phrased differently from ordinance terminology
# (for example, "유기동물 발견" vs "유실ㆍ유기동물 발견 신고").
# The answer model still rejects out-of-scope context, so use a less strict
# retrieval gate and let it evaluate the supplied evidence.
MIN_SIMILARITY = 0.30  # cosine similarity threshold

# Load a local .env file when present; environment variables still take priority.
load_dotenv(BASE_DIR / ".env")


def api_key() -> str | None:
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("OPENAI_API_KEY")
    except FileNotFoundError:
        return None


def fingerprint() -> str:
    digest = hashlib.sha256()
    for file in sorted(DATA_DIR.glob("*.txt")):
        digest.update(file.name.encode("utf-8"))
        digest.update(file.read_bytes())
    return digest.hexdigest()


def split_text(text: str, size: int = 1000, overlap: int = 180) -> list[str]:
    """Paragraph chunks, preserving FAQ Q/A blocks where possible."""
    results: list[str] = []
    for section in re.split(r"(?=^Q\d+\.\s)", text.strip(), flags=re.MULTILINE):
        section = section.strip()
        if not section:
            continue
        if len(section) <= size:
            results.append(section)
            continue
        current = ""
        for paragraph in (p.strip() for p in re.split(r"\n\s*\n", section) if p.strip()):
            candidate = f"{current}\n\n{paragraph}".strip()
            if current and len(candidate) > size:
                results.append(current)
                current = f"{current[-overlap:]}\n\n{paragraph}"
            else:
                current = candidate
        if current:
            results.append(current)
    return results


def embed(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def document_type(text: str) -> str:
    found = re.search(r"^문서유형\s*:\s*(.+)$", text, re.MULTILINE)
    return found.group(1).strip() if found else "미분류"


def rebuild_index(client: OpenAI) -> int:
    documents, metadatas, ids = [], [], []
    for file in sorted(DATA_DIR.glob("*.txt")):
        text = file.read_text(encoding="utf-8")
        for number, chunk in enumerate(split_text(text), 1):
            documents.append(chunk)
            metadatas.append({"source": file.name, "document_type": document_type(text), "chunk_number": number})
            ids.append(f"{file.stem}-{number}")
    if not documents:
        raise RuntimeError("data 폴더에서 TXT 문서를 찾을 수 없습니다.")

    DB_DIR.mkdir(exist_ok=True)
    db = chromadb.PersistentClient(path=str(DB_DIR))
    # The first run has no collection yet. ChromaDB raises its own
    # InvalidCollectionException (not always ValueError) in that case.
    try:
        db.get_collection(COLLECTION)
    except Exception:
        collection_exists = False
    else:
        collection_exists = True
    if collection_exists:
        db.delete_collection(COLLECTION)
    collection = db.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    for start in range(0, len(documents), 64):
        end = start + 64
        collection.add(ids=ids[start:end], documents=documents[start:end], metadatas=metadatas[start:end],
                       embeddings=embed(client, documents[start:end]))
    MANIFEST.write_text(json.dumps({"fingerprint": fingerprint(), "chunks": len(documents)}, ensure_ascii=False), encoding="utf-8")
    return len(documents)


def index_current() -> bool:
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))["fingerprint"] == fingerprint()
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return False


def retrieve(client: OpenAI, question: str) -> list[dict]:
    collection = chromadb.PersistentClient(path=str(DB_DIR)).get_collection(COLLECTION)
    result = collection.query(query_embeddings=embed(client, [question]), n_results=3,
                              include=["documents", "metadatas", "distances"])
    return [{"text": text, "metadata": metadata, "similarity": 1 - float(distance)}
            for text, metadata, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])]


def answer(client: OpenAI, question: str, sources: list[dict]) -> str:
    context = "\n\n".join(f"[출처: {s['metadata']['source']}]\n{s['text']}" for s in sources)
    response = client.responses.create(
        model=ANSWER_MODEL, store=False,
        instructions=("당신은 화성시 민원 안내 챗봇입니다. 제공된 근거 자료만으로 간결한 한국어 답변을 작성하세요. "
                      "근거 밖의 사실, 추측, 법률 해석, 최신 정보는 추가하지 마세요. 근거가 불충분하면 정확히 "
                      f"'{NO_ANSWER}'만 답하세요."),
        input=f"질문: {question}\n\n근거 자료:\n{context}",
    )
    text = response.output_text.strip()
    # The model may phrase the fallback sentence slightly differently.  Normalize
    # it so the UI never presents unrelated search results as answer sources.
    if not text or "확인할 수 없습니다" in text:
        return NO_ANSWER
    return text


st.set_page_config(page_title="화성시 민원 챗봇", page_icon="🏛️", layout="centered")
st.title("🏛️ 화성시 민원 챗봇")
st.caption("화성시 조례·민원 FAQ·민원처리지침을 벡터 검색해 답변합니다.")

key = api_key()
if not key:
    st.error("OPENAI_API_KEY를 환경 변수 또는 Streamlit secrets에 설정해 주세요.")
    st.code('$env:OPENAI_API_KEY="여기에_API_키"', language="powershell")
    st.stop()
client = OpenAI(api_key=key)

with st.sidebar:
    st.subheader("문서 관리")
    if index_current():
        st.success("문서 인덱스가 최신입니다.")
    else:
        st.warning("문서 인덱스가 없거나 data 폴더가 변경되었습니다.")
    if st.button("문서 인덱스 새로 만들기", use_container_width=True):
        try:
            with st.spinner("문서를 벡터 DB에 저장하고 있습니다..."):
                count = rebuild_index(client)
            st.success(f"{count}개 청크를 저장했습니다.")
        except Exception as error:
            st.error(f"인덱싱에 실패했습니다: {error}")

if not MANIFEST.exists():
    st.info("왼쪽 메뉴에서 ‘문서 인덱스 새로 만들기’를 먼저 눌러 주세요.")

with st.form("question_form"):
    question = st.text_input("질문을 입력하세요", placeholder="예: 대형 폐기물은 어떻게 버리나요?")
    submitted = st.form_submit_button("질문하기", type="primary")

if submitted:
    if not question.strip():
        st.warning("질문을 입력해 주세요.")
    elif not index_current():
        st.warning("먼저 문서 인덱스를 새로 만들어 주세요.")
    else:
        try:
            with st.spinner("관련 자료를 찾고 있습니다..."):
                sources = [s for s in retrieve(client, question) if s["similarity"] >= MIN_SIMILARITY]
                response = answer(client, question, sources) if sources else NO_ANSWER
                if response == NO_ANSWER:
                    sources = []
            st.subheader("답변")
            st.write(response)
            if sources:
                st.subheader("출처")
                for n, source in enumerate(sources, 1):
                    metadata = source["metadata"]
                    with st.expander(f"{n}. {metadata['source']} · 유사도 {source['similarity']:.0%}", expanded=n == 1):
                        st.caption(f"문서유형: {metadata['document_type']} | 청크: {metadata['chunk_number']}")
                        st.write(source["text"])
        except Exception as error:
            st.error(f"처리 중 오류가 발생했습니다: {error}")
