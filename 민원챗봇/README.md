# RAG 기반 화성시 민원챗봇

`data` 폴더의 화성시 민원 관련 TXT 문서를 OpenAI 임베딩 모델로 벡터화해 ChromaDB에 저장하고, 질문과 유사한 문서 구간을 검색하여 답변하는 Streamlit 앱입니다.

## 주요 기능

- `data/*.txt` 문서를 청크로 나누어 ChromaDB에 영구 저장
- OpenAI `text-embedding-3-small` 임베딩으로 질문과 문서의 유사도 검색
- 검색한 문서 내용만 사용해 답변 생성
- 관련 근거가 없으면 `자료에서 확인할 수 없습니다` 반환
- 답변 아래에 근거로 사용한 출처 파일명 표시
- 문서 내용이나 임베딩 모델이 바뀌면 다음 실행 시 벡터 DB 자동 재생성
- API 키를 `.env` 또는 시스템 환경변수로 관리

## 필요 환경

- Python 3.10 이상
- OpenAI API 키

## 설치 방법

PowerShell에서 프로젝트 폴더로 이동한 뒤 가상환경을 만들고 패키지를 설치합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 환경변수 설정

예제 파일을 복사합니다.

```powershell
Copy-Item .env.example .env
```

생성된 `.env` 파일에 실제 OpenAI API 키를 입력합니다.

```dotenv
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
```

`.env`는 `.gitignore`에 포함되어 있으므로 Git에 커밋되지 않습니다. 운영 환경에서는 `.env` 대신 시스템의 `OPENAI_API_KEY` 환경변수를 설정해도 됩니다.

## 실행 방법

```powershell
streamlit run app.py
```

브라우저에서 안내된 로컬 주소(기본 `http://localhost:8501`)를 열고 질문합니다. 첫 실행이나 TXT 문서 변경 후 첫 실행에는 임베딩 생성 때문에 시간이 조금 걸릴 수 있으며 OpenAI API 사용료가 발생합니다.

## 문서 추가 및 갱신

1. UTF-8 또는 CP949 형식의 `.txt` 파일을 `data` 폴더에 추가합니다.
2. 앱을 다시 실행하거나 Streamlit 화면을 새로고침합니다.
3. 앱이 변경을 감지하여 `chroma_db`를 자동으로 다시 만듭니다.

생성된 `chroma_db`는 로컬 캐시이며 Git에는 포함되지 않습니다.

## 프로젝트 구조

```text
.
├── app.py
├── data/
│   └── *.txt
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 검색 기준 조정

기본 검색 결과 수는 `TOP_K = 5`, 코사인 거리 기준은 `MAX_DISTANCE = 0.55`입니다. 관련 질문인데도 결과가 없으면 `app.py`의 `MAX_DISTANCE`를 조금 높이고, 무관한 결과가 자주 나오면 낮추세요.
