# 화성시 민원챗봇 (RAG)

`data`의 TXT 문서를 OpenAI 임베딩으로 벡터화하여 ChromaDB에 저장하고, 질문과 관련된 근거를 찾아 답변하는 Streamlit 앱입니다.

## 설치

Python 3.10 이상을 설치한 뒤 PowerShell에서 실행합니다.

```powershell
py -m pip install -r requirements.txt
```

OpenAI API 키를 환경 변수에 설정합니다. 키를 코드에 직접 작성하지 마세요.

```powershell
$env:OPENAI_API_KEY="여기에_API_키"
```

또는 프로젝트 폴더의 `.env.example`을 복사해 `.env` 파일을 만든 뒤 `OPENAI_API_KEY` 값을 입력해도 됩니다.

## 실행

```powershell
py -m streamlit run app.py
```

브라우저에서 표시된 주소(보통 `http://localhost:8501`)를 엽니다. 처음 실행하면 왼쪽의 **문서 인덱스 새로 만들기**를 누릅니다. 이후 질문을 입력하면 답변과 출처 파일명·근거 문단을 확인할 수 있습니다.

`data` 폴더의 문서를 수정하거나 추가했다면 인덱스를 다시 만드세요. 관련 근거가 부족하면 `자료에서 확인할 수 없습니다`라고 답합니다.
