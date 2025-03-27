import os
import logging
from pathlib import Path
from ragflow_sdk import RAGFlow

from settings import (
    API_KEY,
    PAPER_DIR,
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_FILE,
    DATASET_NAMES,
    BASE_URL
)
from file_utils import (
    ensure_directory,
    process_paper_directory
)

def setup_logging():
    """로깅 설정을 초기화합니다."""
    ensure_directory(LOG_FILE.parent)
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )

def main():
    """메인 실행 함수"""
    # 로깅 설정
    setup_logging()
    
    # API 키 확인
    if not API_KEY:
        logging.error("RAGFLOW_API_KEY 환경 변수가 설정되지 않았습니다.")
        return
        
    try:
        # 테스트 모드
        TEST_MODE = False
        if TEST_MODE:
            company = "HD현대중공업"
            test_file = "기대할_것이_많다_HD현대중공업_미래에셋증권_표지_25.02.18.pdf"
            test_path = PAPER_DIR / company / test_file
            
            if not test_path.exists():
                logging.error(f"테스트 파일을 찾을 수 없습니다: {test_path}")
                return
                
            # 테스트 파일 업로드 및 파싱
            logging.info(f"테스트 파일 업로드 시작: {test_file}")
            process_paper_directory(test_path.parent, DATASET_NAMES["paper"])
            logging.info("테스트 파일 업로드 완료")
        else:
            # 모든 파일 업로드 및 파싱
            logging.info("PDF 파일 업로드 시작")
            process_paper_directory(PAPER_DIR, DATASET_NAMES["paper"])
            logging.info("PDF 파일 업로드 완료")
            
    except Exception as e:
        logging.error(f"프로그램 실행 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main() 