import os
import logging
from pathlib import Path
from ragflow_sdk import RAGFlow
from typing import Dict

from settings import (
    API_KEY,
    REPORT_OUTPUT_DIR,
    EVENT_OUTPUT_DIR,
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_FILE,
    DATASET_NAMES,
    BASE_URL
)
from ragflow_analyzer import RagflowAnalyzer
from file_utils import ensure_directory

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

def analyze_single_report(analyzer: RagflowAnalyzer, doc_name: str) -> Dict:
    """
    단일 보고서를 분석합니다.
    
    Args:
        analyzer (RagflowAnalyzer): 분석기 인스턴스
        doc_name (str): 분석할 문서명
        
    Returns:
        Dict: 분석 결과
    """
    try:
        # 보고서 분석
        report_result = analyzer.analyze_report(doc_name)
        logging.info(f"보고서 분석 완료: {doc_name}")
        return report_result
        
    except Exception as e:
        logging.error(f"보고서 분석 중 오류 발생: {doc_name} - {str(e)}")
        return None

def analyze_documents(dataset_name: str) -> None:
    """
    데이터셋의 문서들을 분석합니다.
    
    Args:
        dataset_name (str): 분석할 데이터셋 이름
    """
    # 분석기 초기화
    analyzer = RagflowAnalyzer(API_KEY)
    
    # 데이터셋에서 문서 목록 가져오기
    rag = RAGFlow(api_key=API_KEY, base_url=BASE_URL)
    dataset = rag.get_dataset(name=dataset_name)
    documents = dataset.list_documents()
    
    total_files = len(documents)
    logging.info(f"총 {total_files}개의 보고서 분석 시작")
    
    # 모든 분석 결과를 저장할 리스트
    all_results = []
    
    for i, doc in enumerate(documents, 1):
        logging.info(f"보고서 분석 중: {i}/{total_files} - {doc.name}")
        result = analyze_single_report(analyzer, doc.name)
        if result:
            all_results.append(result)
    
    # 모든 결과를 한 번에 저장
    if all_results:
        analyzer.save_to_text(all_results, REPORT_OUTPUT_DIR / "result.txt")
        analyzer.save_to_excel(all_results, REPORT_OUTPUT_DIR / "result.xlsx")
        logging.info(f"총 {len(all_results)}개의 분석 결과가 저장되었습니다.")
    
    logging.info("모든 보고서 분석 완료")

def main():
    """메인 실행 함수"""
    # 로깅 설정
    setup_logging()
    
    # API 키 확인
    if not API_KEY:
        logging.error("RAGFLOW_API_KEY 환경 변수가 설정되지 않았습니다.")
        return
        
    try:
        # 출력 디렉토리 생성
        ensure_directory(REPORT_OUTPUT_DIR)
        ensure_directory(EVENT_OUTPUT_DIR)
        
        # 테스트 모드
        TEST_MODE = False
        if TEST_MODE:
            test_file = "기대할_것이_많다_HD현대중공업_미래에셋증권_표지_25.02.18.pdf"
            # 테스트 파일 분석
            analyzer = RagflowAnalyzer(API_KEY)
            analyze_single_report(analyzer, test_file)
        else:
            # 모든 파일 분석
            analyze_documents(DATASET_NAMES["paper"])
            
    except Exception as e:
        logging.error(f"프로그램 실행 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main()