import os
from ragflow_sdk import RAGFlow
from ragflow_sdk.modules.chat import Chat
from datetime import datetime
from typing import Dict, List, Union
import pandas as pd
from pydantic import BaseModel, Field
import json

# Pydantic 모델 정의
class ReportAnalysis(BaseModel):
    """애널리스트 보고서 분석 결과를 나타내는 모델"""
    investment_point: Dict[str, Dict[str, str]] = Field(
        description="투자 포인트와 그 근거를 포함하는 섹션입니다. 보고서의 핵심 주장을 인용하고, 이를 뒷받침하는 구체적인 데이터와 분석을 포함합니다.",
        example={
            "key_points": {
                "0": "\"신규 LNG선 수주 물량 확대와 해양플랜트 시장 회복으로 2025년 실적 개선 기대\"",
                "1": "\"중동 지역 신규 프로젝트 진출로 수주 다각화 달성\"",
                "2": "\"친환경 선박 기술력 강화로 경쟁력 확보\""
            },
            "reasonings": {
                "0": "\"2024년 4분기 LNG선 수주 2.5조원 달성, 2025년 3.2조원 예상\"",
                "1": "\"사우디아라비아 신규 플랜트 프로젝트 1.8조원 수주 확정\"",
                "2": "\"메탄가 연료 추진 선박 기술 개발 완료, 2025년 상용화 예정\""
            }
        }
    )
    
    analysis: Dict[str, Dict[str, Dict[str, str]]] = Field(
        description="정량적/정성적 분석 결과를 포함하는 섹션입니다. 보고서의 구체적인 내용을 인용하고, 각 카테고리별로 사실과 의견을 구분하여 정리합니다.",
        example={
            "figure": {
                "fact": {
                    "0": "\"2024년 영업이익 2,822억원 기록, 전년 대비 15% 증가\"",
                    "1": "\"영업이익률 15.2% 달성으로 업계 평균 대비 3.5%p 상회\"",
                    "2": "\"연구개발비 1,200억원 투자, 전년 대비 25% 증가\""
                },
                "opinion": {
                    "0": "\"2025년 LNG선 수주 물량 확대와 해양플랜트 시장 회복으로 영업이익 3,500억원 달성 전망\"",
                    "1": "\"신규 사업 진출과 원가 효율화로 영업이익률 18% 달성 가능\"",
                    "2": "\"R&D 투자 확대로 친환경 선박 시장에서의 경쟁력 강화 기대\""
                }
            },
            "nonfigure": {
                "fact": {
                    "0": "\"글로벌 LNG선 시장 점유율 35%로 1위 달성\"",
                    "1": "\"중동 주요 에너지 기업과 5개사 파트너십 확보\"",
                    "2": "\"친환경 선박 관련 특허 출원 100건 이상으로 기술력 입증\""
                },
                "opinion": {
                    "0": "\"메탄가 연료 추진 선박 기술력으로 경쟁사 대비 2년 이상 앞서는 기술 우위 확보\"",
                    "1": "\"중동 지역 신규 프로젝트 진출로 해양플랜트 시장에서의 입지 강화\"",
                    "2": "\"친환경 선박 시장에서의 차별화된 경쟁력으로 시장 지배력 확대 기대\""
                }
            },
            "material": {
                "fact": {
                    "0": "\"울산 신규 조선소 착공, 2025년 하반기 가동 예정\"",
                    "1": "\"친환경 선박 제작을 위한 신규 장비 2,500억원 규모 구매 계약\"",
                    "2": "\"신규 프로젝트 대응을 위한 엔지니어 500명 채용 계획\""
                },
                "opinion": {
                    "0": "\"신규 조선소 가동으로 생산성 30% 향상 기대\"",
                    "1": "\"최신 장비 도입으로 친환경 선박 생산 경쟁력 강화\"",
                    "2": "\"인력 확충으로 신규 수주 물량 대응력 확보\""
                }
            },
            "public": {
                "fact": {
                    "0": "\"주요 주주 지분율 변동 없이 경영권 안정성 유지\"",
                    "1": "\"배당성향 30% 유지로 주주가치 제고 지속\"",
                    "2": "\"기업지배구조 개선으로 외부평가 등급 상향 조정\""
                },
                "opinion": {
                    "0": "\"주주 안정성 확보로 장기적 경영 계획 추진 가능\"",
                    "1": "\"지속적인 배당 정책으로 기관투자자 관심도 증가\"",
                    "2": "\"기업지배구조 개선으로 ESG 평가 상향 기대\""
                }
            }
        }
    )
    
    sector_indicators: Dict[str, str] = Field(
        description="섹터별 주요 지표를 포함하는 섹션입니다. 보고서에서 언급된 구체적인 수치와 지표를 인용하여 정리합니다.",
        example={
            "market_cap": "\"시가총액 12.5조원으로 업계 1위\"",
            "pe_ratio": "\"PER 15.2배로 업계 평균 대비 20% 할인\"",
            "dividend_yield": "\"배당수익률 2.1%로 안정적 수익 제공\"",
            "roe": "\"ROE 18.5%로 업계 최고 수준\"",
            "debt_ratio": "\"부채비율 45.2%로 업계 평균 대비 10%p 낮은 수준\"",
            "operating_margin": "\"영업이익률 15.2%로 업계 평균 대비 3.5%p 상회\"",
            "revenue_growth": "\"매출 성장률 12.5%로 업계 평균 대비 5%p 높은 수준\"",
            "market_share": "\"글로벌 LNG선 시장 점유율 35%로 1위\""
        }
    )
    
    events: Dict[str, Dict[str, Union[str, float]]] = Field(
        description="예상 이벤트 목록을 포함하는 섹션입니다. 보고서에서 언급된 구체적인 이벤트와 그에 대한 설명을 인용하여 정리합니다.",
        example={
            "0": {
                "category": "시장",
                "type": "긍정적",
                "description": "\"글로벌 LNG 수요 증가로 LNG선 발주 물량 확대\"",
                "probability": 0.7
            },
            "1": {
                "category": "기술",
                "type": "긍정적",
                "description": "\"메탄가 연료 추진 선박 상용화 성공\"",
                "probability": 0.6
            },
            "2": {
                "category": "정책",
                "type": "부정적",
                "description": "\"해운업계 탄소배출 규제 강화 가능성\"",
                "probability": 0.3
            }
        }
    )

class KeyPoints(BaseModel):
    points: Dict[str, str] = Field(
        description="투자 포인트의 주요 내용. 구체적이고 간결하게 작성되며, 보고서에서 강조된 핵심 사항을 포함해야 함.",
        example={"0": "2025년 매출 성장 기대", "1": "수주 증가로 인한 실적 개선"}
    )

class Reasonings(BaseModel):
    reasons: Dict[str, str] = Field(
        description="투자 포인트의 근거. 과거 데이터 기반 사실과 애널리스트의 예측을 포함하며, 구체적인 수치와 함께 작성되어야 함.",
        example={"0": "2024년 매출 4조원 기록", "1": "2025년 수주 126억불 예상"}
    )

class InvestmentPoint(BaseModel):
    key_points: KeyPoints = Field(description="투자 포인트의 주요 내용")
    reasonings: Reasonings = Field(description="투자 포인트에 대한 근거")

class AnalysisCategory(BaseModel):
    fact: Dict[str, str] = Field(
        description="과거 데이터 또는 이미 발생한 이벤트 기반의 사실. 수치 데이터와 함께 구체적으로 작성.",
        example={"0": "2024년 영업이익 2,822억원 기록"}
    )
    opinion: Dict[str, str] = Field(
        description="애널리스트의 예측 또는 주장. 미래 전망이나 주관적 판단 포함.",
        example={"0": "2025년 실적 개선 예상"}
    )

class Analysis(BaseModel):
    figure: AnalysisCategory = Field(description="정량적 데이터 분석")
    nonfigure: AnalysisCategory = Field(description="정성적 정보 분석")
    material: AnalysisCategory = Field(description="중요 이벤트 분석")
    public: AnalysisCategory = Field(description="공개 정보 분석")

class Event(BaseModel):
    category: str = Field(description="이벤트 카테고리", example="시장")
    type: str = Field(description="이벤트 유형", example="긍정적")
    description: str = Field(description="이벤트 설명", example="시장 상황 개선")
    probability: float = Field(description="발생 확률 (0~1)", ge=0.0, le=1.0, example=0.7)

class RagflowAnalyzer:
    def __init__(self, api_key: str, base_url: str = "http://localhost"):
        self.api_key = api_key
        self.base_url = base_url
        self.rag_object = RAGFlow(api_key=api_key, base_url=base_url)
        self.definition_dataset = self.rag_object.get_dataset(name="Definition")
        self.paper_dataset = self.rag_object.get_dataset(name="Paper")
        self.sector_data = self._load_sector_data()
        self.event_data = self._load_event_data()
        self.chat_assistant = None
        self.schema = ReportAnalysis.model_json_schema()

    def _load_sector_data(self) -> dict:
        for doc in self.definition_dataset.list_documents():
            if doc.name == "sector.json":
                for chunk in doc.list_chunks():
                    return json.loads(chunk.content)
        return {}

    def _load_event_data(self) -> dict:
        for doc in self.definition_dataset.list_documents():
            if doc.name == "event.json":
                for chunk in doc.list_chunks():
                    return json.loads(chunk.content)
        return {}

    def parse_filename(self, filename: str) -> Dict[str, str]:
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
        documents = []
        for doc in self.paper_dataset.list_documents():
            parsed_info = self.parse_filename(doc.name)
            documents.append(parsed_info)
        return sorted(documents, key=lambda x: x['date'], reverse=True)

    def _get_document_chunks(self, doc_name: str) -> List[str]:
        chunks = []
        for doc in self.paper_dataset.list_documents():
            if doc.name == doc_name:
                for chunk in doc.list_chunks():
                    chunks.append(chunk.content)
        return chunks

    def _generate_schema_prompt(self, schema: Dict) -> str:
        """Pydantic 스키마를 기반으로 간단한 프롬프트를 생성합니다."""
        prompt_parts = []
        
        def process_field(field: Dict, parent_name: str = "") -> None:
            """필드를 처리하여 간단한 설명만 포함합니다."""
            if "description" in field:
                current_path = f"{parent_name} - {field.get('title', '')}" if parent_name and field.get('title') else parent_name
                current_path = current_path.strip(" -")
                if current_path:
                    description = field['description']
                    prompt_parts.append(f"- {current_path}: {description}")
            
            # 중첩된 속성 처리
            if "properties" in field:
                for prop_name, prop_value in field["properties"].items():
                    new_parent = f"{parent_name} - {prop_value.get('title', prop_name)}" if parent_name else prop_value.get('title', prop_name)
                    process_field(prop_value, new_parent)
            
            # 리스트 항목 처리
            if "items" in field and "properties" in field["items"]:
                items_parent = f"{parent_name} 항목" if parent_name else "항목"
                prompt_parts.append(f"- {items_parent} 정보:")
                for prop_name, prop_value in field["items"]["properties"].items():
                    new_parent = f"{items_parent} - {prop_value.get('title', prop_name)}"
                    process_field(prop_value, new_parent)
        
        # 스키마의 최상위 레벨부터 시작
        if "properties" in schema:
            for prop_name, prop_value in schema["properties"].items():
                process_field(prop_value, prop_name)
        
        return "\n".join(prompt_parts)

    def _get_or_create_chat_assistant(self) -> "Chat":
        CHAT_NAME = f"Noodle Machine"
        if self.chat_assistant is None:
            try:
                self.chat_assistant = self.rag_object.create_chat(
                    name=CHAT_NAME,
                    dataset_ids=[self.paper_dataset.id, self.definition_dataset.id]
                )
                
                # Pydantic 스키마를 기반으로 프롬프트 생성
                schema_prompt = self._generate_schema_prompt(self.schema)
                print("=== 스키마 프롬프트 ===")
                print(schema_prompt)
                print("=====================")
                
                prompt_template = (
                    "당신은 세계 최고의 금융 전문가이자 분석가입니다. 증권사의 애널리스트 보고서를 분석하여 투자자들이 이해하기 쉽게 정보를 추출하고 정리하세요.\n\n"
                    "분석 시 다음 사항을 참고하세요:\n"
                    "1. 보고서의 핵심 내용은 따옴표로 인용하세요.\n"
                    "2. 수치 데이터는 정확하게 인용하세요.\n"
                    "3. 애널리스트의 의견과 예측도 인용하세요.\n\n"
                    "분석할 때 다음 스키마 정보를 참고하세요:\n\n"
                    f"{schema_prompt}\n\n"
                    "문서 정보:\n"
                    "- 문서명: {filename}\n"
                    "- 회사: {company}\n"
                    "- 작성기관: {institution}\n"
                    "- 작성일: {date}\n"
                    "분석할 문서 내용:\n"
                    "{content}\n\n"
                    "반드시 보고서에 명시된 내용만 활용하며, 새로운 정보를 추가하지 마세요. 모든 구체적인 내용은 따옴표로 인용하세요."
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
                print(f"새로운 챗봇 생성됨: {CHAT_NAME}")
            except Exception as e:
                if "Duplicated chat name" in str(e):
                    chats = self.rag_object.list_chats(name=CHAT_NAME)
                    if chats:
                        self.chat_assistant = chats[0]
                        print(f"기존 챗봇 사용: {CHAT_NAME}")
                    else:
                        raise Exception(f"챗봇을 찾을 수 없음: {CHAT_NAME}")
                else:
                    raise e
        return self.chat_assistant

    def _parse_text_to_schema(self, text: str) -> Dict:
        """텍스트 응답을 스키마 구조로 파싱 (번호 기반 처리)"""
        analysis_dict = {
            "investment_point": {"key_points": {"points": {}}, "reasonings": {"reasons": {}}},
            "analysis": {
                "figure": {"fact": {}, "opinion": {}},
                "nonfigure": {"fact": {}, "opinion": {}},
                "material": {"fact": {}, "opinion": {}},
                "public": {"fact": {}, "opinion": {}}
            },
            "sector_indicators": {},
            "events": {}
        }
        
        lines = text.split('\n')
        event_data = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(". ", 1)
            if len(parts) < 2:
                continue
            
            num, content = parts[0], parts[1]
            num = num.strip()
            
            if num == "1":
                analysis_dict["investment_point"]["key_points"]["points"]["0"] = content
            elif num == "2":
                analysis_dict["investment_point"]["reasonings"]["reasons"]["0"] = content
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
                event_data["category"] = content.split(": ", 1)[1] if ": " in content else content
            elif num == "13":
                event_data["type"] = content.split(": ", 1)[1] if ": " in content else content
            elif num == "14":
                event_data["description"] = content.split(": ", 1)[1] if ": " in content else content
            elif num == "15":
                probability_str = content.split(": ", 1)[1] if ": " in content else content
                try:
                    probability = float(probability_str)
                except ValueError:
                    probability = 0.5  # 기본값
                event_data["probability"] = probability
                analysis_dict["events"]["0"] = event_data
        
        return analysis_dict

    def analyze_report(self, doc_name: str) -> Dict:
        doc_info = self.parse_filename(doc_name)
        print(f"=== 문서 정보 ===")
        print(f"파일명: {doc_name}")
        print(f"회사명: {doc_info['company']}")
        print(f"작성기관: {doc_info['institution']}")
        print(f"작성일: {doc_info['date']}")
        
        chunks = self._get_document_chunks(doc_name)
        if not chunks:
            print(f"문서 청크를 찾을 수 없음: {doc_name}")
            return {**doc_info, "analysis": {}, "error": "문서 청크를 찾을 수 없음"}
        
        content = "".join(chunks)
        print(f"=== 문서 청크 수: {len(chunks)} ===")
        
        assistant = self._get_or_create_chat_assistant()
        session = assistant.create_session(f"{doc_info['date']}-{doc_info['company']}")
        
        try:
            print("=== 챗봇 응답 처리 시작 ===")
            
            # 번호 기반 출력 형식 지시
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
                "11. sector_indicators: 섹터별 주요 지표 (수치 포함)\n"
                "12. events - category: 이벤트 카테고리\n"
                "13. events - type: 이벤트 유형\n"
                "14. events - description: 이벤트 설명\n"
                "15. events - probability: 발생 확률 (0~1)\n"
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
            
            print("\n=== 응답 처리 완료 ===")
            print("=== 원본 응답 ===")
            print(result_content)  # 디버깅용 원본 응답 출력
            
            if not result_content.strip():
                raise ValueError("챗봇 응답이 비어 있습니다.")
            
            try:
                # 텍스트를 스키마로 파싱
                analysis_result = self._parse_text_to_schema(result_content)
                result = {**doc_info, "analysis": analysis_result}
                
                # 결과 저장
                with open("test_result.json", 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                self.save_to_text([result], "test_result.txt")
                self.save_to_excel([result], "test_result.xlsx")
                print("=== 결과 저장 완료 ===")
                
                return result
                
            except Exception as e:
                print(f"=== 파싱 오류 ===")
                print(f"오류 메시지: {str(e)}")
                return {
                    **doc_info,
                    "error": f"파싱 실패: {str(e)}",
                    "raw_content": result_content
                }
            
        except Exception as e:
            print(f"=== 분석 중 오류 발생 ===")
            print(str(e))
            return {**doc_info, "error": str(e), "raw_content": result_content if 'result_content' in locals() else ""}

    def analyze_all_reports(self) -> List[Dict]:
        results = []
        for doc in self.get_sorted_documents():
            result = self.analyze_report(doc['original_filename'])
            results.append(result)
        return results

    def save_to_text(self, results: List[Dict], output_file: str = "analysis_results.txt") -> None:
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
                    f.write("분석 내용:\n")
                    analysis = result['analysis']
                    for key, value in analysis["investment_point"]["key_points"]["points"].items():
                        f.write(f"investment_point - key_points: {value}\n")
                    for key, value in analysis["investment_point"]["reasonings"]["reasons"].items():
                        f.write(f"investment_point - reasonings: {value}\n")
                    for key, value in analysis["analysis"]["figure"]["fact"].items():
                        f.write(f"analysis - figure - fact: {value}\n")
                    for key, value in analysis["analysis"]["figure"]["opinion"].items():
                        f.write(f"analysis - figure - opinion: {value}\n")
                    for key, value in analysis["analysis"]["nonfigure"]["fact"].items():
                        f.write(f"analysis - nonfigure - fact: {value}\n")
                    for key, value in analysis["analysis"]["nonfigure"]["opinion"].items():
                        f.write(f"analysis - nonfigure - opinion: {value}\n")
                    for key, value in analysis["analysis"]["material"]["fact"].items():
                        f.write(f"analysis - material - fact: {value}\n")
                    for key, value in analysis["analysis"]["material"]["opinion"].items():
                        f.write(f"analysis - material - opinion: {value}\n")
                    for key, value in analysis["analysis"]["public"]["fact"].items():
                        f.write(f"analysis - public - fact: {value}\n")
                    for key, value in analysis["analysis"]["public"]["opinion"].items():
                        f.write(f"analysis - public - opinion: {value}\n")
                    for key, value in analysis["sector_indicators"].items():
                        f.write(f"sector_indicators: {value}\n")
                    for key, event in analysis["events"].items():
                        f.write(f"events - category: {event['category']}\n")
                        f.write(f"events - type: {event['type']}\n")
                        f.write(f"events - description: {event['description']}\n")
                        f.write(f"events - probability: {event['probability']}\n")
                
                f.write("\n" + "="*50 + "\n\n")

    def save_to_excel(self, results: List[Dict], output_file: str = "analysis_results.xlsx") -> None:
        data = []
        for result in results:
            row = {
                '종목명': result['company'],
                '작성기관': result['institution'],
                '작성일': result['date'],
                '파일명': result['original_filename']
            }
            if 'error' not in result:
                analysis = result['analysis']
                def extract_content(text, prefix):
                    if text.startswith(prefix):
                        return text[len(prefix):].strip()
                    return text
                
                row.update({
                    '투자 포인트 - 주요 포인트': '\n'.join(
                        [extract_content(v, "investment_point - key_points: ") for v in analysis["investment_point"]["key_points"]["points"].values()]
                    ),
                    '투자 포인트 - 근거': '\n'.join(
                        [extract_content(v, "investment_point - reasonings: ") for v in analysis["investment_point"]["reasonings"]["reasons"].values()]
                    ),
                    '정량적 분석 - 사실': '\n'.join(
                        [extract_content(v, "analysis - figure - fact: ") for v in analysis["analysis"]["figure"]["fact"].values()]
                    ),
                    '정량적 분석 - 의견': '\n'.join(
                        [extract_content(v, "analysis - figure - opinion: ") for v in analysis["analysis"]["figure"]["opinion"].values()]
                    ),
                    '정성적 분석 - 사실': '\n'.join(
                        [extract_content(v, "analysis - nonfigure - fact: ") for v in analysis["analysis"]["nonfigure"]["fact"].values()]
                    ),
                    '정성적 분석 - 의견': '\n'.join(
                        [extract_content(v, "analysis - nonfigure - opinion: ") for v in analysis["analysis"]["nonfigure"]["opinion"].values()]
                    ),
                    '중요 이벤트 - 사실': '\n'.join(
                        [extract_content(v, "analysis - material - fact: ") for v in analysis["analysis"]["material"]["fact"].values()]
                    ),
                    '중요 이벤트 - 의견': '\n'.join(
                        [extract_content(v, "analysis - material - opinion: ") for v in analysis["analysis"]["material"]["opinion"].values()]
                    ),
                    '공개 정보 - 사실': '\n'.join(
                        [extract_content(v, "analysis - public - fact: ") for v in analysis["analysis"]["public"]["fact"].values()]
                    ),
                    '공개 정보 - 의견': '\n'.join(
                        [extract_content(v, "analysis - public - opinion: ") for v in analysis["analysis"]["public"]["opinion"].values()]
                    ),
                    '섹터 지표': '\n'.join(
                        [extract_content(v, "sector_indicators: ") for v in analysis["sector_indicators"].values()]
                    ),
                    '예상 이벤트 - 카테고리': '\n'.join([e['category'] for e in analysis["events"].values()]),
                    '예상 이벤트 - 유형': '\n'.join([e['type'] for e in analysis["events"].values()]),
                    '예상 이벤트 - 설명': '\n'.join([e['description'] for e in analysis["events"].values()]),
                    '예상 이벤트 - 확률': '\n'.join([str(e['probability']) for e in analysis["events"].values()])
                })
            else:
                row['오류'] = result['error']
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_excel(output_file, index=False)
        print(f"Excel 파일 '{output_file}'이 저장되었습니다. DataFrame 크기: {df.shape}")

def main():
    API_KEY = "ragflow-I0Y2ViN2M2MDU3OTExZjA5N2ZhMTIxOD"
    analyzer = RagflowAnalyzer(API_KEY)
    
    TEST_MODE = True
    TEST_FILE = "기대할_것이_많다_HD현대중공업_미래에셋증권_표지_25.02.18.pdf"
    
    if TEST_MODE:
        print(f"테스트 모드: {TEST_FILE} 파일 분석 중...")
        result = analyzer.analyze_report(TEST_FILE)
        analyzer.save_to_text([result], "test_result.txt")
        analyzer.save_to_excel([result], "test_result.xlsx")
        
        print("=== 분석 결과 ===")
        print(f"회사명: {result['company']}")
        print(f"작성기관: {result['institution']}")
        print(f"작성일: {result['date']}")
        
        if 'error' in result:
            print("=== 오류 발생 ===")
            print(result['error'])
            if 'raw_content' in result:
                print("원본 응답:")
                print(result['raw_content'])
        else:
            print("=== 상세 분석 ===")
            analysis = result['analysis']
            for key, value in analysis["investment_point"]["key_points"]["points"].items():
                print(f"investment_point - key_points: {value}")
            for key, value in analysis["investment_point"]["reasonings"]["reasons"].items():
                print(f"investment_point - reasonings: {value}")
            for key, value in analysis["analysis"]["figure"]["fact"].items():
                print(f"analysis - figure - fact: {value}")
            for key, value in analysis["analysis"]["figure"]["opinion"].items():
                print(f"analysis - figure - opinion: {value}")
            for key, value in analysis["analysis"]["nonfigure"]["fact"].items():
                print(f"analysis - nonfigure - fact: {value}")
            for key, value in analysis["analysis"]["nonfigure"]["opinion"].items():
                print(f"analysis - nonfigure - opinion: {value}")
            for key, value in analysis["analysis"]["material"]["fact"].items():
                print(f"analysis - material - fact: {value}")
            for key, value in analysis["analysis"]["material"]["opinion"].items():
                print(f"analysis - material - opinion: {value}")
            for key, value in analysis["analysis"]["public"]["fact"].items():
                print(f"analysis - public - fact: {value}")
            for key, value in analysis["analysis"]["public"]["opinion"].items():
                print(f"analysis - public - opinion: {value}")
            for key, value in analysis["sector_indicators"].items():
                print(f"sector_indicators: {value}")
            for key, event in analysis["events"].items():
                print(f"events: {event['category']}|{event['type']}|{event['description']}|{event['probability']}")
    else:
        print("전체 보고서 분석 중...")
        results = analyzer.analyze_all_reports()
        analyzer.save_to_text(results, "all_results.txt")
        analyzer.save_to_excel(results, "all_results.xlsx")
        print(f"분석 완료: {len(results)}개 보고서 처리됨")

if __name__ == "__main__":
    main()