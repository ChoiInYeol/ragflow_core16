import os
import json
import logging
import pandas as pd
from typing import Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime
import time
from ragflow_sdk import RAGFlow
from ragflow_sdk.modules.chat import Chat
from report_models import ReportAnalysis
from event_models import EventAnalysis, ReportMetadata, CoreEvent

class RagflowAnalyzer:
    """애널리스트 보고서 분석을 위한 클래스"""
    
    def __init__(self, api_key: str, base_url: str = "http://localhost"):
        """
        RagflowAnalyzer 초기화
        
        Args:
            api_key (str): Ragflow API 키
            base_url (str): Ragflow 서버 URL
        """
        self.api_key = api_key
        self.base_url = base_url
        self.rag_object = RAGFlow(api_key=api_key, base_url=base_url)
        self.definition_dataset = self.rag_object.get_dataset(name="Definition")
        self.paper_dataset = self.rag_object.get_dataset(name="Paper")
        self.sector_data = self._load_sector_data()
        self.event_data = self._load_event_data()
        self.chat_assistant = None
        self.schema = ReportAnalysis.model_json_schema()
        self.event_schema = EventAnalysis.model_json_schema()
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _load_sector_data(self) -> dict:
        """섹터 데이터를 로드합니다."""
        for doc in self.definition_dataset.list_documents():
            if doc.name == "sector.json":
                for chunk in doc.list_chunks():
                    return json.loads(chunk.content)
        return {}
        
    def _load_event_data(self) -> dict:
        """이벤트 데이터를 로드합니다."""
        for doc in self.definition_dataset.list_documents():
            if doc.name == "event.json":
                for chunk in doc.list_chunks():
                    return json.loads(chunk.content)
        return {}
        
    def parse_filename(self, filename: str) -> Dict[str, str]:
        """
        파일명을 파싱하여 메타데이터를 추출합니다.
        
        Args:
            filename (str): 파싱할 파일명
            
        Returns:
            Dict[str, str]: 파싱된 메타데이터
        """
        name = os.path.splitext(filename)[0]
        parts = name.split('_')
        date_str = parts[-1]
        institution = parts[-3]
        company = parts[-4]
        date_obj = datetime.strptime(date_str, '%y.%m.%d')
        formatted_date = date_obj.strftime('%Y-%m-%d')
        return {
            'company': company,
            'institution': institution,
            'date': formatted_date,
            'original_filename': filename
        }
        
    def get_sorted_documents(self) -> List[Dict[str, str]]:
        """
        정렬된 문서 목록을 반환합니다.
        
        Returns:
            List[Dict[str, str]]: 정렬된 문서 목록
        """
        documents = []
        for doc in self.paper_dataset.list_documents():
            parsed_info = self.parse_filename(doc.name)
            documents.append(parsed_info)
        return sorted(documents, key=lambda x: x['date'], reverse=True)
        
    def _get_document_chunks(self, doc_name: str) -> List[str]:
        """
        문서의 청크 목록을 반환합니다.
        
        Args:
            doc_name (str): 문서명
            
        Returns:
            List[str]: 청크 목록
        """
        chunks = []
        for doc in self.paper_dataset.list_documents():
            if doc.name == doc_name:
                for chunk in doc.list_chunks():
                    chunks.append(chunk.content)
        return chunks
        
    def _generate_schema_prompt(self, schema: Dict) -> str:
        """
        스키마를 기반으로 프롬프트를 생성합니다.
        
        Args:
            schema (Dict): Pydantic 스키마
            
        Returns:
            str: 생성된 프롬프트
        """
        prompt_parts = []
        
        def process_field(field: Dict, parent_name: str = "") -> None:
            if "description" in field:
                current_path = f"{parent_name} - {field.get('title', '')}" if parent_name and field.get('title') else parent_name
                current_path = current_path.strip(" -")
                if current_path:
                    description = field['description']
                    prompt_parts.append(f"- {current_path}: {description}")
            
            if "properties" in field:
                for prop_name, prop_value in field["properties"].items():
                    new_parent = f"{parent_name} - {prop_value.get('title', prop_name)}" if parent_name else prop_value.get('title', prop_name)
                    process_field(prop_value, new_parent)
            
            if "items" in field and "properties" in field["items"]:
                items_parent = f"{parent_name} 항목" if parent_name else "항목"
                prompt_parts.append(f"- {items_parent} 정보:")
                for prop_name, prop_value in field["items"]["properties"].items():
                    new_parent = f"{items_parent} - {prop_value.get('title', prop_name)}"
                    process_field(prop_value, new_parent)
        
        if "properties" in schema:
            for prop_name, prop_value in schema["properties"].items():
                process_field(prop_value, prop_name)
        
        return "\n".join(prompt_parts)
        
    def _get_or_create_chat_assistant(self) -> "Chat":
        """
        채팅 어시스턴트를 가져오거나 생성합니다.
        
        Returns:
            Chat: 채팅 어시스턴트
        """
        CHAT_NAME = f"Noodle Machine"
        if self.chat_assistant is None:
            try:
                self.chat_assistant = self.rag_object.create_chat(
                    name=CHAT_NAME,
                    dataset_ids=[self.paper_dataset.id, self.definition_dataset.id]
                )
                
                schema_prompt = self._generate_schema_prompt(self.schema)
                
                prompt_template = (
                    "당신은 세계 최고의 금융 전문가이자 분석가입니다. 증권사의 애널리스트 보고서를 분석하여 "
                    "투자 포인트와 이벤트를 정확하게 파악하고 정리하세요.\n\n"
                    "분석 시 다음 사항을 엄격히 준수하세요:\n"
                    "1. 모든 내용은 보고서에 명시된 내용만 활용하며, 새로운 정보를 추가하지 마세요.\n"
                    "2. 핵심 내용은 반드시 따옴표로 인용하세요.\n"
                    "3. 수치 데이터는 정확하게 인용하세요.\n"
                    "4. 시점 정보를 명확히 표시하세요 (발생 시점, 예상 시점 등).\n\n"
                    "분석은 다음 두 부분으로 나누어 수행하세요:\n\n"
                    "1. 투자 포인트:\n"
                    "   - 주요 포인트: 핵심 투자 논리\n"
                    "   - 근거: 수치와 함께 투자 포인트의 근거\n\n"
                    "2. 정량적 분석:\n"
                    "   - 사실: 수치 기반의 객관적 사실\n"
                    "   - 의견: 수치 기반의 예측 및 전망\n\n"
                    "3. 정성적 분석:\n"
                    "   - 사실: 정성적 정보 기반의 객관적 사실\n"
                    "   - 의견: 정성적 정보 기반의 예측 및 전망\n\n"
                    "4. 중요 이벤트 분석:\n"
                    "   - 사실: 중요 이벤트 관련 객관적 사실\n"
                    "   - 의견: 중요 이벤트 관련 예측 및 전망\n\n"
                    "5. 공개 정보 분석:\n"
                    "   - 사실: 공개된 정보 기반의 객관적 사실\n"
                    "   - 의견: 공개 정보 기반의 예측 및 전망\n\n"
                    "6. 섹터 지표:\n"
                    "   - 산업/섹터별 주요 지표와 동향\n\n"
                    "각 이벤트별로 다음 구조로 분석하세요:\n\n"
                    "1. 이벤트 기본 정보:\n"
                    "   - 이벤트 카테고리 (자본/재무, 설비/R&D, 실적, M&A, 신사업, 계약, 공급망, 외부충격, 파트너리스크 등)\n"
                    "   - 이벤트 유형 (발표, 계획, 실행, 완료, 취소 등)\n"
                    "   - 이벤트 발생/예상 시점\n"
                    "   - 이벤트 설명 (구체적인 내용)\n"
                    "   - 원문 인용\n\n"
                    "2. 시장 영향 분석:\n"
                    "   - 단기 영향 (1-3개월)\n"
                    "   - 중기 영향 (3-12개월)\n"
                    "   - 장기 영향 (1년 이상)\n"
                    "   - 영향 정도 (높음/중간/낮음)\n\n"
                    "3. 파생 효과:\n"
                    "   - 직접적 효과 (매출, 영업이익, 자산 등)\n"
                    "   - 간접적 효과 (시장점유율, 경쟁력, 브랜드가치 등)\n"
                    "   - 위험 요소\n\n"
                    "4. 연관 이벤트:\n"
                    "   - 선행 이벤트\n"
                    "   - 후속 예상 이벤트 (발생 확률 포함)\n"
                    "   - 연관된 다른 기업/산업에 미치는 영향\n\n"
                    "문서 정보:\n"
                    "- 문서명: {filename}\n"
                    "- 회사: {company}\n"
                    "- 작성기관: {institution}\n"
                    "- 작성일: {date}\n"
                    "분석할 문서 내용:\n"
                    "{content}\n\n"
                    "각 섹션별로 위 구조에 맞춰 상세히 분석하되, 보고서에 명시된 내용만 활용하세요."
                )
                
                chat_config = {
                    "llm": {
                        "model_name": "gpt-4o-mini",
                        "temperature": 0.25,
                        "top_p": 0.45,
                        "presence_penalty": 0.35,
                        "frequency_penalty": 0.55,
                        "max_tokens": 2048
                    },
                    "prompt": {
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
                        ],
                        "prompt": prompt_template
                    }
                }
                
                self.chat_assistant.update(chat_config)
                self.logger.info(f"새로운 챗봇 생성됨: {CHAT_NAME}")
                
            except Exception as e:
                if "Duplicated chat name" in str(e):
                    chats = self.rag_object.list_chats(name=CHAT_NAME)
                    if chats:
                        self.chat_assistant = chats[0]
                        self.logger.info(f"기존 챗봇 사용: {CHAT_NAME}")
                    else:
                        raise Exception(f"챗봇을 찾을 수 없음: {CHAT_NAME}")
                else:
                    raise e
        return self.chat_assistant
        
    def analyze_report(self, doc_name: str) -> Dict:
        """
        단일 보고서를 분석합니다.
        
        Args:
            doc_name (str): 분석할 문서명
            
        Returns:
            Dict: 분석 결과
        """
        doc_info = self.parse_filename(doc_name)
        self.logger.info(f"=== 문서 정보 ===")
        self.logger.info(f"파일명: {doc_name}")
        self.logger.info(f"회사명: {doc_info['company']}")
        self.logger.info(f"작성기관: {doc_info['institution']}")
        self.logger.info(f"작성일: {doc_info['date']}")
        
        chunks = self._get_document_chunks(doc_name)
        if not chunks:
            self.logger.error(f"문서 청크를 찾을 수 없음: {doc_name}")
            return {**doc_info, "analysis": {}, "error": "문서 청크를 찾을 수 없음"}
        
        content = "".join(chunks)
        self.logger.info(f"=== 문서 청크 수: {len(chunks)} ===")
        
        assistant = self._get_or_create_chat_assistant()
        session = assistant.create_session(f"{doc_info['date']}-{doc_info['company']}")
        
        try:
            self.logger.info("=== 챗봇 응답 처리 시작 ===")
            
            question = (
                "아래 구조에 맞게 정보를 추출하고 분석하세요. 번호와 내용만 출력하며, 추가 설명은 넣지 마세요.\n\n"
                "1. investment_point - key_points: 투자 포인트의 주요 내용\n"
                "2. investment_point - reasonings: 투자 포인트에 대한 근거 (수치 포함)\n"
                "3. analysis - figure - fact: 정량적 데이터 기반 사실 (수치 포함)\n"
                "4. analysis - figure - opinion: 정량적 데이터 기반 예측\n"
                "5. analysis - nonfigure - fact: 정성적 정보 기반 사실\n"
                "6. analysis - nonfigure - opinion: 정성적 정보 기반 예측\n"
                "7. analysis - material - fact: 중요 이벤트 기반 사실\n"
                "8. analysis - material - opinion: 중요 이벤트 기반 예측\n"
                "9. analysis - public - fact: 공개 정보 기반 사실\n"
                "10. analysis - public - opinion: 공개 정보 기반 예측\n"
                "11. sector_indicators: 섹터별 주요 지표 (수치 포함)\n\n"
                "12. events - category: 이벤트 카테고리 (자본/재무, 설비/R&D, 실적, M&A, 신사업, 계약, 공급망, 외부충격, 파트너리스크 등)\n"
                "13. events - type: 이벤트 유형 (발표, 계획, 실행, 완료, 취소 등)\n"
                "14. events - timestamp: 이벤트 발생/예상 시점\n"
                "15. events - description: 이벤트 설명 (구체적인 내용)\n"
                "16. events - source_quote: 원문 인용\n"
                "17. events - market_impact - short_term: 단기 영향 (1-3개월)\n"
                "18. events - market_impact - mid_term: 중기 영향 (3-12개월)\n"
                "19. events - market_impact - long_term: 장기 영향 (1년 이상)\n"
                "20. events - market_impact - severity: 영향 정도 (높음/중간/낮음)\n"
                "21. events - derived_effects - direct: 직접적 효과 (매출, 영업이익, 자산 등)\n"
                "22. events - derived_effects - indirect: 간접적 효과 (시장점유율, 경쟁력, 브랜드가치 등)\n"
                "23. events - derived_effects - risks: 위험 요소\n"
                "24. events - related_events - preceding: 선행 이벤트\n"
                "25. events - related_events - subsequent: 후속 예상 이벤트 (발생 확률 포함)\n"
                "26. events - related_events - related_impact: 연관된 다른 기업/산업에 미치는 영향\n"
                "\n"
            )
            
            response_stream = session.ask(
                question=question,
                stream=True,
                company=doc_info['company'],
                institution=doc_info['institution'],
                date=doc_info['date'],
                filename=doc_name,
                content=content
            )
            
            result_content = ""
            last_content = ""
            for message in response_stream:
                if message.content:
                    current_content = message.content
                    if current_content != last_content:
                        result_content += current_content[len(last_content):]
                        last_content = current_content
                        print(current_content[len(last_content):], end='', flush=True)
            
            self.logger.info("\n=== 응답 처리 완료 ===")
            self.logger.info("=== 원본 응답 ===")
            self.logger.info(result_content)
            
            if not result_content.strip():
                raise ValueError("챗봇 응답이 비어 있습니다.")
            
            try:
                analysis_result = self._parse_text_to_schema(result_content)
                result = {**doc_info, "analysis": analysis_result}
                
                # 결과 저장
                from settings import REPORT_OUTPUT_DIR
                self.save_to_text([result], REPORT_OUTPUT_DIR / "result.txt")
                self.save_to_excel([result], REPORT_OUTPUT_DIR / "result.xlsx")
                self.logger.info("=== 결과 저장 완료 ===")
                
                return result
                
            except Exception as e:
                self.logger.error(f"=== 파싱 오류 ===")
                self.logger.error(f"오류 메시지: {str(e)}")
                return {
                    **doc_info,
                    "error": f"파싱 실패: {str(e)}",
                    "raw_content": result_content
                }
            
        except Exception as e:
            self.logger.error(f"=== 분석 중 오류 발생 ===")
            self.logger.error(str(e))
            return {**doc_info, "error": str(e), "raw_content": result_content if 'result_content' in locals() else ""}
            
    def analyze_all_reports(self) -> List[Dict]:
        """
        모든 보고서를 분석합니다.
        
        Returns:
            List[Dict]: 분석 결과 목록
        """
        results = []
        for doc in self.get_sorted_documents():
            result = self.analyze_report(doc['original_filename'])
            results.append(result)
        return results
        
    def save_to_text(self, results: List[Dict], output_file: Union[str, Path]) -> None:
        """
        분석 결과를 텍스트 파일로 저장합니다.
        
        Args:
            results (List[Dict]): 분석 결과 목록
            output_file (Union[str, Path]): 저장할 파일 경로
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # 기존 파일이 있으면 삭제
            if output_file.exists():
                output_file.unlink()
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(f"=== 분석 결과 ===\n")
                    f.write(f"회사명: {result['company']}\n")
                    f.write(f"작성기관: {result['institution']}\n")
                    f.write(f"작성일: {result['date']}\n")
                    f.write(f"파일명: {result['original_filename']}\n")
                    f.write("\n")
                    
                    if 'error' in result:
                        f.write(f"오류: {result['error']}\n")
                        if 'raw_content' in result:
                            f.write("\n원본 응답:\n")
                            f.write(result['raw_content'])
                    else:
                        f.write("=== 투자 포인트 ===\n")
                        f.write(f"주요 내용: {result['analysis']['investment_point']['key_points']['0']}\n")
                        f.write(f"근거: {result['analysis']['investment_point']['reasonings']['0']}\n\n")
                        
                        f.write("=== 정량 분석 ===\n")
                        f.write(f"사실: {result['analysis']['analysis']['figure']['fact']['0']}\n")
                        f.write(f"의견: {result['analysis']['analysis']['figure']['opinion']['0']}\n\n")
                        
                        f.write("=== 정성 분석 ===\n")
                        f.write(f"사실: {result['analysis']['analysis']['nonfigure']['fact']['0']}\n")
                        f.write(f"의견: {result['analysis']['analysis']['nonfigure']['opinion']['0']}\n\n")
                        
                        f.write("=== 중요 이벤트 ===\n")
                        f.write(f"사실: {result['analysis']['analysis']['material']['fact']['0']}\n")
                        f.write(f"의견: {result['analysis']['analysis']['material']['opinion']['0']}\n\n")
                        
                        f.write("=== 공개 정보 ===\n")
                        f.write(f"사실: {result['analysis']['analysis']['public']['fact']['0']}\n")
                        f.write(f"의견: {result['analysis']['analysis']['public']['opinion']['0']}\n\n")
                        
                        f.write("=== 섹터 지표 ===\n")
                        f.write(f"{result['analysis']['sector_indicators']['0']}\n\n")
                        
                        f.write("=== 이벤트 분석 ===\n")
                        event = result['analysis']['events']['0']
                        f.write(f"카테고리: {event['category']}\n")
                        f.write(f"유형: {event['type']}\n")
                        f.write(f"시점: {event['timestamp']}\n")
                        f.write(f"설명: {event['description']}\n")
                        f.write(f"원문: {event['source_quote']}\n\n")
                        
                        f.write("시장 영향:\n")
                        f.write(f"단기 영향: {event['market_impact']['short_term']}\n")
                        f.write(f"중기 영향: {event['market_impact']['mid_term']}\n")
                        f.write(f"장기 영향: {event['market_impact']['long_term']}\n")
                        f.write(f"영향 정도: {event['market_impact']['severity']}\n\n")
                        
                        f.write("파생 효과:\n")
                        f.write(f"직접적 효과: {event['derived_effects']['direct']}\n")
                        f.write(f"간접적 효과: {event['derived_effects']['indirect']}\n")
                        f.write(f"위험 요소: {event['derived_effects']['risks']}\n\n")
                        
                        f.write("연관 이벤트:\n")
                        f.write(f"선행 이벤트: {event['related_events']['preceding']}\n")
                        f.write(f"후속 예상 이벤트: {event['related_events']['subsequent']}\n")
                        f.write(f"연관 영향: {event['related_events']['related_impact']}\n")
                    
                    f.write("\n" + "="*50 + "\n\n")
                    
            logging.info(f"텍스트 파일 '{output_file}'이 저장되었습니다.")
            
        except Exception as e:
            logging.error(f"텍스트 파일 저장 중 오류 발생: {str(e)}")
            raise
        
    def save_to_excel(self, results: List[Dict], output_file: Union[str, Path]) -> None:
        """
        분석 결과를 Excel 파일로 저장합니다.
        
        Args:
            results (List[Dict]): 저장할 분석 결과 리스트
            output_file (Union[str, Path]): 저장할 파일 경로
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # 데이터프레임을 위한 데이터 구조 생성
            data = {
                "회사명": [],
                "작성기관": [],
                "작성일": [],
                "파일명": [],
                "투자포인트_주요내용": [],
                "투자포인트_근거": [],
                "정량분석_사실": [],
                "정량분석_의견": [],
                "정성분석_사실": [],
                "정성분석_의견": [],
                "중요이벤트_사실": [],
                "중요이벤트_의견": [],
                "공개정보_사실": [],
                "공개정보_의견": [],
                "섹터지표": [],
                "이벤트_카테고리": [],
                "이벤트_유형": [],
                "이벤트_시점": [],
                "이벤트_설명": [],
                "이벤트_원문": [],
                "시장영향_단기": [],
                "시장영향_중기": [],
                "시장영향_장기": [],
                "시장영향_정도": [],
                "파생효과_직접": [],
                "파생효과_간접": [],
                "파생효과_위험": [],
                "연관이벤트_선행": [],
                "연관이벤트_후속": [],
                "연관이벤트_영향": []
            }
            
            for result in results:
                data["회사명"].append(result["company"])
                data["작성기관"].append(result["institution"])
                data["작성일"].append(result["date"])
                data["파일명"].append(result["original_filename"])
                data["투자포인트_주요내용"].append(result["analysis"]["investment_point"]["key_points"]["0"])
                data["투자포인트_근거"].append(result["analysis"]["investment_point"]["reasonings"]["0"])
                data["정량분석_사실"].append(result["analysis"]["analysis"]["figure"]["fact"]["0"])
                data["정량분석_의견"].append(result["analysis"]["analysis"]["figure"]["opinion"]["0"])
                data["정성분석_사실"].append(result["analysis"]["analysis"]["nonfigure"]["fact"]["0"])
                data["정성분석_의견"].append(result["analysis"]["analysis"]["nonfigure"]["opinion"]["0"])
                data["중요이벤트_사실"].append(result["analysis"]["analysis"]["material"]["fact"]["0"])
                data["중요이벤트_의견"].append(result["analysis"]["analysis"]["material"]["opinion"]["0"])
                data["공개정보_사실"].append(result["analysis"]["analysis"]["public"]["fact"]["0"])
                data["공개정보_의견"].append(result["analysis"]["analysis"]["public"]["opinion"]["0"])
                data["섹터지표"].append(result["analysis"]["sector_indicators"]["0"])
                
                event = result["analysis"]["events"]["0"]
                data["이벤트_카테고리"].append(event["category"])
                data["이벤트_유형"].append(event["type"])
                data["이벤트_시점"].append(event["timestamp"])
                data["이벤트_설명"].append(event["description"])
                data["이벤트_원문"].append(event["source_quote"])
                data["시장영향_단기"].append(event["market_impact"]["short_term"])
                data["시장영향_중기"].append(event["market_impact"]["mid_term"])
                data["시장영향_장기"].append(event["market_impact"]["long_term"])
                data["시장영향_정도"].append(event["market_impact"]["severity"])
                data["파생효과_직접"].append(event["derived_effects"]["direct"])
                data["파생효과_간접"].append(event["derived_effects"]["indirect"])
                data["파생효과_위험"].append(event["derived_effects"]["risks"])
                data["연관이벤트_선행"].append(event["related_events"]["preceding"])
                data["연관이벤트_후속"].append(event["related_events"]["subsequent"])
                data["연관이벤트_영향"].append(event["related_events"]["related_impact"])
            
            # 데이터프레임 생성 및 저장
            df = pd.DataFrame(data)
            df.to_excel(output_file, index=False, engine='openpyxl')
            logging.info(f"Excel 파일 '{output_file}'이 저장되었습니다. DataFrame 크기: {df.shape}")
            
        except Exception as e:
            logging.error(f"Excel 파일 저장 중 오류 발생: {str(e)}")
            raise
        
    def _parse_text_to_schema(self, text: str) -> Dict:
        """
        텍스트를 스키마 구조로 파싱합니다.
        
        Args:
            text (str): 파싱할 텍스트
            
        Returns:
            Dict: 파싱된 결과
        """
        analysis_dict = {
            "investment_point": {
                "key_points": {"0": ""},
                "reasonings": {"0": ""}
            },
            "analysis": {
                "figure": {"fact": {"0": ""}, "opinion": {"0": ""}},
                "nonfigure": {"fact": {"0": ""}, "opinion": {"0": ""}},
                "material": {"fact": {"0": ""}, "opinion": {"0": ""}},
                "public": {"fact": {"0": ""}, "opinion": {"0": ""}}
            },
            "sector_indicators": {"0": ""},
            "events": {
                "0": {
                    "category": "",
                    "type": "",
                    "timestamp": "",
                    "description": "",
                    "source_quote": "",
                    "market_impact": {
                        "short_term": "",
                        "mid_term": "",
                        "long_term": "",
                        "severity": ""
                    },
                    "derived_effects": {
                        "direct": "",
                        "indirect": "",
                        "risks": ""
                    },
                    "related_events": {
                        "preceding": "",
                        "subsequent": "",
                        "related_impact": ""
                    }
                }
            }
        }
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            try:
                # 숫자와 내용을 분리할 때 더 유연하게 처리
                parts = line.split(". ", 1)
                if len(parts) != 2:
                    continue
                    
                num = parts[0].strip()
                content = parts[1].strip()
                
                # 숫자만 추출
                num = ''.join(filter(str.isdigit, num))
                if not num:
                    continue
                
                # 필드명 제거하고 실제 값만 저장
                content = content.split(": ", 1)[-1] if ": " in content else content
                
                if num == "1":
                    analysis_dict["investment_point"]["key_points"]["0"] = content
                elif num == "2":
                    analysis_dict["investment_point"]["reasonings"]["0"] = content
                elif num == "3":
                    analysis_dict["analysis"]["figure"]["fact"]["0"] = content
                elif num == "4":
                    analysis_dict["analysis"]["figure"]["opinion"]["0"] = content
                elif num == "5":
                    analysis_dict["analysis"]["nonfigure"]["fact"]["0"] = content
                elif num == "6":
                    analysis_dict["analysis"]["nonfigure"]["opinion"]["0"] = content
                elif num == "7":
                    analysis_dict["analysis"]["material"]["fact"]["0"] = content
                elif num == "8":
                    analysis_dict["analysis"]["material"]["opinion"]["0"] = content
                elif num == "9":
                    analysis_dict["analysis"]["public"]["fact"]["0"] = content
                elif num == "10":
                    analysis_dict["analysis"]["public"]["opinion"]["0"] = content
                elif num == "11":
                    analysis_dict["sector_indicators"]["0"] = content
                elif num == "12":
                    analysis_dict["events"]["0"]["category"] = content
                elif num == "13":
                    analysis_dict["events"]["0"]["type"] = content
                elif num == "14":
                    analysis_dict["events"]["0"]["timestamp"] = content
                elif num == "15":
                    analysis_dict["events"]["0"]["description"] = content
                elif num == "16":
                    analysis_dict["events"]["0"]["source_quote"] = content
                elif num == "17":
                    analysis_dict["events"]["0"]["market_impact"]["short_term"] = content
                elif num == "18":
                    analysis_dict["events"]["0"]["market_impact"]["mid_term"] = content
                elif num == "19":
                    analysis_dict["events"]["0"]["market_impact"]["long_term"] = content
                elif num == "20":
                    analysis_dict["events"]["0"]["market_impact"]["severity"] = content
                elif num == "21":
                    analysis_dict["events"]["0"]["derived_effects"]["direct"] = content
                elif num == "22":
                    analysis_dict["events"]["0"]["derived_effects"]["indirect"] = content
                elif num == "23":
                    analysis_dict["events"]["0"]["derived_effects"]["risks"] = content
                elif num == "24":
                    analysis_dict["events"]["0"]["related_events"]["preceding"] = content
                elif num == "25":
                    analysis_dict["events"]["0"]["related_events"]["subsequent"] = content
                elif num == "26":
                    analysis_dict["events"]["0"]["related_events"]["related_impact"] = content
                    
            except Exception as e:
                self.logger.error(f"파싱 중 오류 발생: {str(e)}")
                continue
                
        return analysis_dict 