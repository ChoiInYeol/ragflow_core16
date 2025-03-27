import os
from pathlib import Path

# 현재 디렉토리
CURRENT_DIR = Path(__file__).parent

# 데이터 디렉토리
DATA_DIR = CURRENT_DIR / "data"
DEFINITION_DIR = DATA_DIR / "definition"
PAPER_DIR = DATA_DIR / "paper"

# 출력 디렉토리
OUTPUT_DIR = CURRENT_DIR / "output"
REPORT_OUTPUT_DIR = OUTPUT_DIR / "reports"
EVENT_OUTPUT_DIR = OUTPUT_DIR / "events"

# API 설정
API_KEY = "ragflow-I0Y2ViN2M2MDU3OTExZjA5N2ZhMTIxOD"
BASE_URL = "http://localhost"

# 분석 설정
MAX_RETRIES = 3
TIMEOUT = 30

# 로깅 설정
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = OUTPUT_DIR / "ragflow.log"

# 데이터셋 이름
DATASET_NAMES = {
    "definition": "Definition",
    "paper": "Paper"
}

# 문서 처리 설정
DOCUMENT_PROCESSING = {
    "definition": {
        "chunk_size": 1000,
        "chunk_overlap": 200
    },
    "paper": {
        "chunk_size": 2000,
        "chunk_overlap": 400
    }
}

# LLM 설정
LLM_CONFIG = {
    "model_name": "gpt-4o-mini",
    "temperature": 0.25,
    "top_p": 0.45,
    "presence_penalty": 0.35,
    "frequency_penalty": 0.55,
    "max_tokens": 2048
}

# 프롬프트 설정
PROMPT_CONFIG = {
    "similarity_threshold": 0.35,
    "keywords_similarity_weight": 0.65,
    "top_n": 11,
    "show_quote": True,
    "variables": [
        {"key": "company", "optional": False},
        {"key": "institution", "optional": False},
        {"key": "date", "optional": False},
        {"key": "filename", "optional": False},
        {"key": "content", "optional": False}
    ]
} 