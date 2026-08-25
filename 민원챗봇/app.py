import hashlib
import json
import os
import re
from pathlib import Path

import chromadb
import streamlit as st
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
MANIFEST_PATH = CHROMA_DIR / "manifest.json"
COLLECTION_NAME = "hwaseong_civil_documents"
NO_ANSWER = "자료에서 확인할 수 없습니다"

load_dotenv(BASE_DIR / ".env")

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
TOP_K = 5
MAX_DISTANCE = 0.55  # cosine distance: 작을수록 질문과 더 유사함
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def read_text(path: Path) -> str:
    """UTF-8을 우선 사용하고, 국내 공공문서에서 흔한 인코딩도 지원한다."""
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"문서 인코딩을 읽을 수 없습니다: {path.name}")


def split_text(text: str) -> list[str]:
    """문단 경계를 최대한 보존하면서 겹치는 청크를 만든다."""
    normalized = re.sub(r"\r\n?", "\n", text).strip()
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", normalized) if part.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > CHUNK_SIZE:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(paragraph):
                chunk = paragraph[start : start + CHUNK_SIZE].strip()
                if chunk:
                    chunks.append(chunk)
                start += CHUNK_SIZE - CHUNK_OVERLAP
            continue

        candidate = f"{current}\n\n{paragraph}".strip()
        if len(candidate) <= CHUNK_SIZE:
            current = candidate
        else:
            if current:
                chunks.append(current)
                overlap = current[-CHUNK_OVERLAP:]
                current = f"{overlap}\n\n{paragraph}".strip()
            else:
                current = paragraph

    if current:
        chunks.append(current)
    return chunks


def document_fingerprint(files: list[Path]) -> str:
    digest = hashlib.sha256()
    digest.update(EMBEDDING_MODEL.encode("utf-8"))
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    embeddings: list[list[float]] = []
    batch_size = 100
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        embeddings.extend(item.embedding for item in response.data)
    return embeddings


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


@st.cache_resource(show_spinner=False)
def prepare_vector_db(fingerprint: str):
    del fingerprint  # 캐시 키로 사용되며, 실제 값은 아래 manifest 비교에 사용하지 않는다.
    files = sorted(DATA_DIR.glob("*.txt"))
    if not files:
        raise FileNotFoundError("data 폴더에 .txt 문서가 없습니다.")

    current_fingerprint = document_fingerprint(files)
    openai_client = OpenAI()
    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    manifest = load_manifest()
    rebuild = manifest.get("fingerprint") != current_fingerprint

    if rebuild:
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    if rebuild or collection.count() == 0:
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        for path in files:
            for index, chunk in enumerate(split_text(read_text(path))):
                ids.append(f"{path.stem}-{index}")
                documents.append(chunk)
                metadatas.append({"source": path.name, "chunk": index})

        if not documents:
            raise ValueError("벡터 DB에 저장할 문서 내용이 없습니다.")

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embed_texts(openai_client, documents),
        )
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(
            json.dumps(
                {
                    "fingerprint": current_fingerprint,
                    "embedding_model": EMBEDDING_MODEL,
                    "files": [path.name for path in files],
                    "chunks": len(documents),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return openai_client, collection


def retrieve(client: OpenAI, collection, question: str) -> list[dict]:
    query_embedding = embed_texts(client, [question])[0]
    count = min(TOP_K, collection.count())
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=count,
        include=["documents", "metadatas", "distances"],
    )

    matches = []
    for document, metadata, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        if distance <= MAX_DISTANCE:
            matches.append(
                {"text": document, "source": metadata["source"], "distance": distance}
            )
    return matches


def answer_question(client: OpenAI, question: str, matches: list[dict]) -> str:
    if not matches:
        return NO_ANSWER

    context = "\n\n".join(
        f"[문서 {index} | 출처: {match['source']}]\n{match['text']}"
        for index, match in enumerate(matches, start=1)
    )
    response = client.responses.create(
        model=CHAT_MODEL,
        instructions=(
            "당신은 화성시 민원 안내 챗봇입니다. 반드시 제공된 문서 내용만 근거로 "
            "한국어로 간결하고 정확하게 답하세요. 문서에 없는 사실을 추측하거나 일반 지식을 "
            "추가하지 마세요. 질문에 답할 충분한 근거가 문서에 없으면 다른 설명 없이 정확히 "
            f"'{NO_ANSWER}'라고만 답하세요. 답변 본문에 출처 목록을 만들지 마세요."
        ),
        input=f"<question>\n{question}\n</question>\n\n<context>\n{context}\n</context>",
        max_output_tokens=600,
    )
    answer = response.output_text.strip()
    if not answer or NO_ANSWER in answer:
        return NO_ANSWER
    return answer


def main() -> None:
    st.set_page_config(page_title="화성시 민원챗봇", page_icon="🏛️", layout="centered")
    st.title("🏛️ 화성시 민원챗봇")
    st.caption("화성시 민원 문서를 검색해 문서에 근거한 답변과 출처를 제공합니다.")

    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일 또는 환경변수를 확인하세요.")
        st.stop()

    files = sorted(DATA_DIR.glob("*.txt"))
    if not files:
        st.error("data 폴더에 .txt 문서가 없습니다.")
        st.stop()

    try:
        fingerprint = document_fingerprint(files)
        with st.spinner("문서를 확인하고 벡터 DB를 준비하고 있습니다..."):
            client, collection = prepare_vector_db(fingerprint)
    except Exception as exc:
        st.error(f"벡터 DB 준비 중 오류가 발생했습니다: {exc}")
        st.stop()

    with st.sidebar:
        st.subheader("문서 현황")
        st.write(f"문서 {len(files)}개 · 청크 {collection.count()}개")
        for path in files:
            st.caption(f"• {path.name}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                st.caption("출처: " + ", ".join(message["sources"]))

    question = st.chat_input("화성시 민원에 대해 질문해 주세요")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("관련 문서를 찾고 있습니다..."):
                matches = retrieve(client, collection, question)
                answer = answer_question(client, question, matches)
            sources = (
                sorted({match["source"] for match in matches})
                if answer != NO_ANSWER
                else []
            )
            st.markdown(answer)
            if sources:
                st.caption("출처: " + ", ".join(sources))
        except Exception as exc:
            answer = f"답변 생성 중 오류가 발생했습니다: {exc}"
            sources = []
            st.error(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


if __name__ == "__main__":
    main()
