import os
from ragflow_sdk import RAGFlow
from ragflow_sdk.modules.chat import Chat
from datetime import datetime
import json
import pandas as pd
from typing import Dict, List, Tuple
import re
import yaml
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class KeyPoints(BaseModel):
    """주요 포인트 모델"""
    points: Dict[str, str] = Field(
        description="주요 포인트 목록",
        example={"0": "주요 포인트 1", "1": "주요 포인트 2"}
    )

class Reasonings(BaseModel):
    """근거 모델"""
    reasons: Dict[str, str] = Field(
        description="근거 목록",
        example={"0": "근거 1", "1": "근거 2"}
    )

class InvestmentPoint(BaseModel):
    """투자 포인트 모델"""
    key_points: KeyPoints = Field(description="주요 포인트")
    reasonings: Reasonings = Field(description="근거")

class AnalysisCategory(BaseModel):
    """분석 카테고리 모델"""
    fact: Dict[str, str] = Field(
        description="사실 정보",
        example={"0": "사실 1", "1": "사실 2"}
    )
    opinion: Dict[str, str] = Field(
        description="의견 정보",
        example={"0": "의견 1", "1": "의견 2"}
    )

class Analysis(BaseModel):
    """분석 모델"""
    figure: AnalysisCategory = Field(
        description="정량적 데이터 분석 (수치 기반 근거)\n"
                   "- 구체적인 수치, 차트, 그래프, 통계 데이터 등\n"
                   "- 매출, 영업이익, 성장률 등 정량적 지표\n"
                   "- 재무제표 기반의 수치적 분석"
    )
    nonfigure: AnalysisCategory = Field(
        description="정성적 정보 분석 (수치 없는 설명)\n"
                   "- 회사의 시장 지배력, 브랜드 가치 등\n"
                   "- 경영진의 리더십, 기업 문화 등\n"
                   "- 산업 동향, 시장 환경 등 정성적 요인"
    )
    material: AnalysisCategory = Field(
        description="중요 이벤트 또는 주요 영향 요인 분석\n"
                   "- 대규모 계약 체결, 사업 구조 개편\n"
                   "- 경영진 변화, 전략적 제휴\n"
                   "- 기업 가치에 중대한 영향을 미치는 사항"
    )
    public: AnalysisCategory = Field(
        description="공개 정보 분석\n"
                   "- 공시된 정보, 미디어 보도 내용\n"
                   "- 투자자들에게 이미 알려진 정보\n"
                   "- 시장에서 공유되는 정보의 신뢰도"
    )

class Event(BaseModel):
    """이벤트 모델"""
    category: str = Field(description="이벤트 카테고리", example="시장")
    type: str = Field(description="이벤트 유형", example="긍정적")
    description: str = Field(description="이벤트 설명", example="시장 상황 개선")
    probability: float = Field(description="발생 확률", ge=0.0, le=1.0, example=0.7)

class ReportAnalysis(BaseModel):
    """보고서 분석 모델"""
    investment_point: InvestmentPoint = Field(description="투자 포인트")
    analysis: Analysis = Field(description="분석 결과")
    sector_indicators: Dict[str, str] = Field(
        description="섹터 지표",
        example={"0": "지표 1", "1": "지표 2"}
    )
    events: Dict[str, Event] = Field(
        description="이벤트 목록",
        example={"0": {"category": "시장", "type": "긍정적", "description": "시장 상황 개선", "probability": 0.7}}
    )

    class Config:
        json_schema_extra = {
            "example": {
                "investment_point": {
                    "key_points": {"points": {"0": "주요 포인트 1", "1": "주요 포인트 2"}},
                    "reasonings": {"reasons": {"0": "근거 1", "1": "근거 2"}}
                },
                "analysis": {
                    "figure": {"fact": {"0": "사실 1"}, "opinion": {"0": "의견 1"}},
                    "nonfigure": {"fact": {"0": "사실 1"}, "opinion": {"0": "의견 1"}},
                    "material": {"fact": {"0": "사실 1"}, "opinion": {"0": "의견 1"}},
                    "public": {"fact": {"0": "사실 1"}, "opinion": {"0": "의견 1"}}
                },
                "sector_indicators": {"0": "지표 1", "1": "지표 2"},
                "events": {
                    "0": {"category": "시장", "type": "긍정적", "description": "시장 상황 개선", "probability": 0.7}
                }
            }
        }

class RagflowAnalyzer:
    """
    RAGFlow를 이용한 보고서 분석기
    
    Attributes:
        api_key (str): RAGFlow API 키
        base_url (str): RAGFlow 서버 URL
        rag_object (RAGFlow): RAGFlow 객체
        definition_dataset: Definition 데이터셋
        paper_dataset: Paper 데이터셋
        sector_data (dict): 섹터 데이터
        event_data (dict): 이벤트 데이터
        chat_assistant: 챗봇 인스턴스 저장
        parser: JsonOutputParser instance for parsing responses
    """
    
    def __init__(self, api_key: str, base_url: str = "http://localhost"):
        """
        RagflowAnalyzer 초기화
        
        Args:
            api_key (str): RAGFlow API 키
            base_url (str): RAGFlow 서버 URL
        """
        self.api_key = api_key
        self.base_url = base_url
        self.rag_object = RAGFlow(api_key=api_key, base_url=base_url)
        self.definition_dataset = self.rag_object.get_dataset(name="Definition")
        self.paper_dataset = self.rag_object.get_dataset(name="Paper")
        self.sector_data = self._load_sector_data()
        self.event_data = self._load_event_data()
        self.chat_assistant = None  # 챗봇 인스턴스 저장
        
        # JsonOutputParser 설정
        self.parser = JsonOutputParser(
            pydantic_object=ReportAnalysis,
            schema_extra={
                "example": ReportAnalysis.Config.json_schema_extra["example"]
            }
        )
        
        # 포맷 지시사항 커스터마이징 - 문자열 상수 사용
        self.format_instructions = (
            "다음 형식에 맞춰 JSON으로 응답해주세요. 응답은 다음 구조를 가져야 합니다:\n\n"
            "investment_point: 객체\n"
            "   key_points: 객체\n"
            "       points: 객체 (키: 문자열 숫자, 값: 문자열)\n"
            "reasonings: 객체\n"
            "   reasons: 객체 (키: 문자열 숫자, 값: 문자열)\n"
            "analysis: 객체\n"
            "   figure: 객체\n"
            "       fact: 객체 (키: 문자열 숫자, 값: 문자열)\n"
            "       opinion: 객체 (키: 문자열 숫자, 값: 문자열)\n"
            "   nonfigure: 객체 (figure와 동일한 구조)\n"
            "   material: 객체 (figure와 동일한 구조)\n"
            "   public: 객체 (figure와 동일한 구조)\n"
            "sector_indicators: 객체 (키: 문자열 숫자, 값: 문자열)\n"
            "events: 객체\n"
            "  키: 문자열 숫자\n"
            "  값: 객체\n"
            "      category: 문자열\n"
            "      type: 문자열\n"
            "      description: 문자열\n"
            "      probability: 숫자 (0-1 사이)"
        )
        
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
        파일명을 파싱하여 종목명, 작성기관, 작성일을 추출합니다.
        
        Args:
            filename (str): 파싱할 파일명
            
        Returns:
            Dict[str, str]: 파싱된 정보를 담은 딕셔너리
        """
        # 파일 확장자 제거
        name = os.path.splitext(filename)[0]
        print(f"=== 파일명 파싱 디버깅 ===")
        print(f"원본 파일명: {filename}")
        print(f"확장자 제거: {name}")
        
        # 언더스코어로 분리
        parts = name.split('_')
        print(f"분리된 부분: {parts}")
        
        # 마지막 4개 요소 추출
        date_str = parts[-1]  # 25.02.18
        institution = parts[-3]  # 미래에셋증권
        company = parts[-4]  # HD현대중공업
        
        print(f"추출된 정보:")
        print(f"- 날짜: {date_str}")
        print(f"- 기관: {institution}")
        print(f"- 회사: {company}")
        
        # 날짜 형식 변환 (25.02.18 -> 2025-02-18)
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
        데이터셋의 문서들을 파싱하고 날짜순으로 정렬합니다.
        
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
        문서의 청크들을 검색합니다.
        
        Args:
            doc_name (str): 문서 이름
            
        Returns:
            List[str]: 문서 청크 내용 리스트
        """
        chunks = []
        for doc in self.paper_dataset.list_documents():
            if doc.name == doc_name:
                for chunk in doc.list_chunks():
                    chunks.append(chunk.content)
        return chunks

    def _get_or_create_chat_assistant(self) -> 'Chat':
        """
        챗봇을 가져오거나 생성합니다.
        이미 존재하는 경우 기존 챗봇을 반환하고, 없는 경우 새로 생성합니다.
        
        Returns:
            Chat: 챗봇 인스턴스
        """
        CHAT_NAME = "Report_Analyzer"
        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'chat_config.json')
        
        if self.chat_assistant is None:
            try:
                # 챗봇 생성 시도
                self.chat_assistant = self.rag_object.create_chat(
                    name=CHAT_NAME,
                    dataset_ids=[self.paper_dataset.id]
                )
                
                # 설정 파일 로드
                try:
                    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                        chat_config = json.load(f)
                    
                    # 프롬프트 템플릿 로드
                    template_file = chat_config['prompt'].get('template_file', 'prompt_template.txt')
                    template_path = os.path.join(os.path.dirname(__file__), template_file)
                    
                    try:
                        with open(template_path, 'r', encoding='utf-8') as f:
                            prompt_template = f.read()
                        
                        # 커스텀 포맷 지시사항 추가
                        prompt_template = prompt_template.replace("{format_instructions}", self.format_instructions)
                        
                        # template_file 제거하고 prompt 설정
                        chat_config['prompt'].pop('template_file', None)
                        chat_config['prompt']['prompt'] = prompt_template
                        
                        print(f"=== 프롬프트 템플릿 로드 완료 ===")
                        print(f"템플릿 파일: {template_file}")
                        print(f"포맷 지시사항 추가됨")
                        
                    except FileNotFoundError as e:
                        print("=== 파일을 찾을 수 없음 ===")
                        print(f"파일 경로: {str(e)}")
                        print("기본 프롬프트를 사용합니다.")
                    
                except FileNotFoundError:
                    print("=== 설정 파일을 찾을 수 없음 ===")
                    print(f"파일 경로: {CONFIG_PATH}")
                    print("기본 설정을 사용합니다.")
                    chat_config = {
                        "llm": {
                            "model_name": "gpt-4o-mini",
                            "temperature": 0.1,
                            "top_p": 0.3,
                            "presence_penalty": 0.2,
                            "frequency_penalty": 0.7,
                            "max_tokens": 2048
                        },
                        "prompt": {
                            "similarity_threshold": 0.2,
                            "keywords_similarity_weight": 0.7,
                            "top_n": 8,
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
                
                # 챗봇 설정 업데이트
                self.chat_assistant.update(chat_config)
                print(f"새로운 챗봇 생성됨: {CHAT_NAME}")
                
            except Exception as e:
                if "Duplicated chat name" in str(e):
                    # 이미 존재하는 경우 기존 챗봇 찾기
                    chats = self.rag_object.list_chats(name=CHAT_NAME)
                    if chats:
                        self.chat_assistant = chats[0]
                        print(f"기존 챗봇 사용: {CHAT_NAME}")
                    else:
                        raise Exception(f"챗봇을 찾을 수 없음: {CHAT_NAME}")
                else:
                    raise e
                    
        return self.chat_assistant
    
    def _extract_json_from_response(self, response: str) -> dict:
        """
        응답에서 JSON 형식의 데이터를 추출합니다.
        
        Args:
            response (str): 챗봇 응답
            
        Returns:
            dict: 추출된 데이터
        """
        try:
            # 응답에서 JSON 부분만 추출
            json_str = response.strip()
            if json_str.startswith('**ERROR**'):
                print(f"=== 오류 응답 ===")
                print(json_str)
                return {}
            
            # JSON 시작과 끝 찾기
            json_start = json_str.find("{")
            json_end = json_str.rfind("}") + 1
            if json_start == -1 or json_end <= json_start:
                print("=== 유효한 JSON 형식을 찾을 수 없음 ===")
                print("전체 응답:")
                print(json_str)
                return {}
            
            # JSON 부분만 추출
            json_content = json_str[json_start:json_end]
            
            # JSON 전처리
            json_content = self._preprocess_json(json_content)
            
            # JSON 파싱 시도
            try:
                result = json.loads(json_content)
                if not result:
                    print("=== JSON 파싱 결과가 비어있음 ===")
                    print("JSON 내용:")
                    print(json_content)
                    return {}
                
                # 필수 필드 확인
                required_fields = ['investment_point', 'analysis', 'sector_indicators', 'events']
                missing_fields = [field for field in required_fields if field not in result]
                if missing_fields:
                    print(f"=== 필수 필드 누락 ===")
                    print(f"누락된 필드: {missing_fields}")
                    print("JSON 내용:")
                    print(json_content)
                    return {}
                
                # 필드 구조 검증
                if not self._validate_json_structure(result):
                    print("=== JSON 구조 검증 실패 ===")
                    print("JSON 내용:")
                    print(json_content)
                    return {}
                    
                print("=== 추출된 데이터 ===")
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return result
                
            except json.JSONDecodeError as e:
                print(f"=== JSON 파싱 오류 ===")
                print(f"오류 메시지: {str(e)}")
                print("JSON 내용:")
                print(json_content)
                
                # 오류가 발생한 위치 주변의 컨텍스트 출력
                error_pos = e.pos
                start = max(0, error_pos - 20)
                end = min(len(json_content), error_pos + 20)
                print(f"오류 위치 주변: {json_content[start:end]}")
                return {}
                
        except Exception as e:
            print(f"=== 처리 중 오류 발생 ===")
            print(f"오류 메시지: {str(e)}")
            print("전체 응답:")
            print(response)
            return {}
    
    def _preprocess_json(self, json_content: str) -> str:
        """
        JSON 문자열을 전처리합니다.
        
        Args:
            json_content (str): 전처리할 JSON 문자열
            
        Returns:
            str: 전처리된 JSON 문자열
        """
        try:
            # 불필요한 제어 문자 제거
            json_content = re.sub(r'[\n\t\r]', '', json_content)
            
            # 작은따옴표를 큰따옴표로 변환
            json_content = re.sub(r"([^\\])'", r'\1"', json_content)
            
            # 숫자 키를 큰따옴표로 감싸기
            json_content = re.sub(r'(\d+):', r'"\1":', json_content)
            
            # 키-값 쌍 보정
            json_content = re.sub(r"'\s*:\s*'", '": "', json_content)
            
            # 리스트와 객체 끝의 불필요한 쉼표 제거
            json_content = re.sub(r',\s*([}\]])', r'\1', json_content)
            
            # 숫자 값의 따옴표 제거 (키가 아닌 경우에만)
            json_content = re.sub(r':\s*"(\d+\.?\d*)"', r': \1', json_content)
            
            # 불필요한 공백 제거
            json_content = re.sub(r'\s+', ' ', json_content)
            
            # 중복된 따옴표 제거
            json_content = re.sub(r'""', '"', json_content)
            
            # 이스케이프된 따옴표 처리
            json_content = re.sub(r'\\"', '"', json_content)
            
            # 빈 문자열 처리
            json_content = re.sub(r'""', 'null', json_content)
            
            # 유효성 검사
            try:
                json.loads(json_content)
            except json.JSONDecodeError as e:
                print(f"=== JSON 전처리 후 유효성 검사 실패 ===")
                print(f"오류 메시지: {str(e)}")
                print("전처리된 JSON:")
                print(json_content)
                raise
            
            return json_content
            
        except Exception as e:
            print(f"=== JSON 전처리 중 오류 발생 ===")
            print(f"오류 메시지: {str(e)}")
            print("원본 JSON:")
            print(json_content)
            raise
    
    def _validate_json_structure(self, data: dict) -> bool:
        """
        JSON 구조를 검증합니다.
        
        Args:
            data (dict): 검증할 JSON 데이터
            
        Returns:
            bool: 검증 결과
        """
        try:
            # investment_point 검증
            if not isinstance(data.get('investment_point'), dict):
                return False
            if not isinstance(data['investment_point'].get('key_points'), dict):
                return False
            if not isinstance(data['investment_point'].get('reasonings'), dict):
                return False
            
            # analysis 검증
            if not isinstance(data.get('analysis'), dict):
                return False
            for category in ['figure', 'nonfigure', 'material', 'public']:
                if not isinstance(data['analysis'].get(category), dict):
                    return False
                for subcategory in ['fact', 'opinion']:
                    if not isinstance(data['analysis'][category].get(subcategory), dict):
                        return False
            
            # sector_indicators 검증
            if not isinstance(data.get('sector_indicators'), dict):
                return False
            
            # events 검증
            if not isinstance(data.get('events'), dict):
                return False
            for event_key, event in data['events'].items():
                if not isinstance(event, dict):
                    return False
                required_event_fields = ['category', 'type', 'description', 'probability']
                if not all(field in event for field in required_event_fields):
                    return False
                if not isinstance(event['probability'], (int, float)):
                    return False
            
            return True
            
        except Exception as e:
            print(f"=== 구조 검증 중 오류 발생 ===")
            print(str(e))
            return False
    
    def analyze_report(self, doc_name: str) -> Dict:
        """
        개별 보고서를 분석합니다.
        
        Args:
            doc_name (str): 분석할 문서 이름
            
        Returns:
            Dict: 분석 결과
        """
        # 문서 정보 파싱
        doc_info = self.parse_filename(doc_name)
        print(f"=== 문서 정보 ===")
        print(f"파일명: {doc_name}")
        print(f"회사명: {doc_info['company']}")
        print(f"작성기관: {doc_info['institution']}")
        print(f"작성일: {doc_info['date']}")
        
        # 문서 청크 검색
        chunks = self._get_document_chunks(doc_name)
        if not chunks:
            print(f"문서 청크를 찾을 수 없음: {doc_name}")
            return {
                **doc_info,
                "analysis": {},
                "error": "문서 청크를 찾을 수 없음"
            }
            
        # 문서 내용 결합
        content = "".join(chunks)
        print(f"=== 문서 청크 수: {len(chunks)} ===")
        
        # 챗봇 가져오기 또는 생성
        assistant = self._get_or_create_chat_assistant()
        
        # 세션 생성
        session = assistant.create_session(f"Analysis for {doc_name}")
        
        try:
            print("=== 챗봇 응답 처리 시작 ===")
            
            # 분석 실행 (스트리밍 활성화)
            response_stream = session.ask(
                question="",  # 빈 질문으로 프롬프트의 변수만 사용
                stream=True,  # 스트리밍 활성화
                company=doc_info['company'],
                institution=doc_info['institution'],
                date=doc_info['date'],
                filename=doc_name,
                content=content  # 문서 내용 전달
            )
            
            # 스트리밍 응답 처리
            result_content = ""
            last_content = ""
            
            for message in response_stream:
                if message.content:
                    current_content = message.content
                    if current_content != last_content:
                        result_content += current_content[len(last_content):]
                        last_content = current_content
                        print(current_content[len(last_content):], end='', flush=True)
            
            print("\n=== 응답 처리 완료 ===")
            
            # JsonOutputParser를 사용하여 결과 파싱
            try:
                analysis_result = self.parser.parse(result_content)
                result = {
                    **doc_info,
                    "analysis": analysis_result.dict()
                }
            except Exception as e:
                print(f"=== JSON 파싱 오류 ===")
                print(f"오류 메시지: {str(e)}")
                return {
                    **doc_info,
                    "analysis": {},
                    "error": f"JSON 파싱 실패: {str(e)}"
                }
            
            # 결과 저장 시도
            try:
                # JSON 저장
                with open("test_result.json", 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                # Excel 저장
                self.save_to_excel([result], "test_result.xlsx")
                print("=== 결과 저장 완료 ===")
                
            except Exception as e:
                print(f"=== 결과 저장 중 오류 발생 ===")
                print(str(e))
                # 저장 실패 시에도 결과는 반환
                pass
            
            return result
            
        except Exception as e:
            print(f"=== 분석 중 오류 발생 ===")
            print(str(e))
            # 에러 메시지가 'message'인 경우 더 자세한 정보 출력
            if str(e) == "'message'":
                print("서버에서 오류 메시지를 받았습니다. 응답을 확인해주세요.")
            return {
                **doc_info,
                "analysis": {},
                "error": str(e)
            }
    
    def analyze_all_reports(self) -> List[Dict]:
        """
        모든 보고서를 분석합니다.
        
        Returns:
            List[Dict]: 분석된 모든 보고서 결과
        """
        results = []
        for doc in self.get_sorted_documents():
            result = self.analyze_report(doc['original_filename'])
            results.append(result)
        return results
    
    def save_to_excel(self, results: List[Dict], output_file: str = "analysis_results.xlsx"):
        """
        분석 결과를 Excel 파일로 저장합니다.
        
        Args:
            results (List[Dict]): 저장할 분석 결과
            output_file (str): 출력 파일 경로
        """
        # 데이터 정리
        excel_data = []
        for result in results:
            # 기본 정보
            row = {
                '회사명': result['company'],
                '작성기관': result['institution'],
                '작성일': result['date'],
                '파일명': result['original_filename']
            }
            
            # 분석 결과
            analysis = result.get('analysis', {})
            
            # 투자 포인트
            investment_point = analysis.get('investment_point', {})
            key_points = investment_point.get('key_points', {})
            reasonings = investment_point.get('reasonings', {})
            row['투자포인트_주요포인트'] = '\n'.join([key_points.get(str(i), '') for i in range(len(key_points))])
            row['투자포인트_근거'] = '\n'.join([reasonings.get(str(i), '') for i in range(len(reasonings))])
            
            # 분석 결과
            analysis_data = analysis.get('analysis', {})
            for category in ['figure', 'nonfigure', 'material', 'public']:
                cat_data = analysis_data.get(category, {})
                facts = cat_data.get('fact', {})
                opinions = cat_data.get('opinion', {})
                row[f'분석_{category}_사실'] = '\n'.join([facts.get(str(i), '') for i in range(len(facts))])
                row[f'분석_{category}_의견'] = '\n'.join([opinions.get(str(i), '') for i in range(len(opinions))])
            
            # 섹터 지표
            sector_indicators = analysis.get('sector_indicators', {})
            row['섹터지표'] = '\n'.join([sector_indicators.get(str(i), '') for i in range(len(sector_indicators))])
            
            # 이벤트
            events = analysis.get('events', {})
            for i in range(len(events)):
                event = events.get(str(i), {})
                row[f'이벤트{i+1}_카테고리'] = event.get('category', '')
                row[f'이벤트{i+1}_타입'] = event.get('type', '')
                row[f'이벤트{i+1}_설명'] = event.get('description', '')
                row[f'이벤트{i+1}_확률'] = event.get('probability', '')
            
            excel_data.append(row)
        
        # 데이터프레임 생성 및 저장
        df = pd.DataFrame(excel_data)
        df.to_excel(output_file, index=False)

def main():
    # API 키 설정
    API_KEY = "ragflow-I0Y2ViN2M2MDU3OTExZjA5N2ZhMTIxOD"
    
    # 분석기 초기화
    analyzer = RagflowAnalyzer(API_KEY)
    
    # 테스트 모드 설정
    TEST_MODE = True
    TEST_FILE = "기대할_것이_많다_HD현대중공업_미래에셋증권_표지_25.02.18.pdf"  # 테스트할 파일명
    
    if TEST_MODE:
        print(f"테스트 모드: {TEST_FILE} 파일 분석 중...")
        result = analyzer.analyze_report(TEST_FILE)
        
        # 결과 저장
        analyzer.save_to_excel([result], "test_result.xlsx")
        
        # 결과 출력
        print("=== 분석 결과 ===")
        print(f"회사명: {result['company']}")
        print(f"작성기관: {result['institution']}")
        print(f"작성일: {result['date']}")
        
        if 'error' in result:
            print("=== 오류 발생 ===")
            print(result['error'])
        else:
            print("=== 상세 분석 ===")
            print(json.dumps(result['analysis'], ensure_ascii=False, indent=2))
    else:
        print("전체 보고서 분석 중...")
        results = analyzer.analyze_all_reports()
        
        # 결과 저장
        analyzer.save_to_excel(results)
        print(f"분석 완료: {len(results)}개 보고서 처리됨")

if __name__ == "__main__":
    main() 