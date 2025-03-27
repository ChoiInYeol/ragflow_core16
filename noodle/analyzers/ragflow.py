import json
from typing import Dict, List
from ragflow_sdk import RAGFlow
from ragflow_sdk.modules.chat import Chat

from models.report import ReportAnalysis
from utils.prompt import generate_schema_prompt, get_analysis_prompt, get_structured_output_prompt
from utils.file import parse_filename, save_to_text, save_to_excel
import os

class RagflowAnalyzer:
    def __init__(self, api_key: str, base_url: str = "http://localhost", use_event_data: bool = True, use_sector_data: bool = True):
        self.api_key = api_key
        self.base_url = base_url
        self.use_event_data = use_event_data
        self.use_sector_data = use_sector_data
        self.rag_object = RAGFlow(api_key=api_key, base_url=base_url)
        self.definition_dataset = self.rag_object.get_dataset(name="Definition")
        self.paper_dataset = self.rag_object.get_dataset(name="Paper")
        self.sector_data = self._load_sector_data() if use_sector_data else {}
        self.event_data = self._load_event_data() if use_event_data else {}
        self.chat_assistant = None
        self.schema = ReportAnalysis.model_json_schema()

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

    def get_sorted_documents(self) -> List[Dict[str, str]]:
        """정렬된 문서 목록을 반환합니다."""
        documents = []
        for doc in self.paper_dataset.list_documents():
            parsed_info = parse_filename(doc.name)
            documents.append(parsed_info)
        return sorted(documents, key=lambda x: x['date'], reverse=True)

    def _get_document_chunks(self, doc_name: str) -> List[str]:
        """문서의 청크들을 반환합니다."""
        chunks = []
        for doc in self.paper_dataset.list_documents():
            if doc.name == doc_name:
                for chunk in doc.list_chunks():
                    chunks.append(chunk.content)
        return chunks

    def _get_sector_prompt(self) -> str:
        """섹터 데이터를 기반으로 프롬프트를 생성합니다."""
        if not self.use_sector_data:
            return "섹터 지표 정보: 섹터 데이터를 사용하지 않습니다."
            
        prompt_parts = ["섹터 지표 정보:"]
        for indicator, description in self.sector_data.items():
            prompt_parts.append(f"- {indicator}: {description}")
        return "\n".join(prompt_parts)

    def _get_event_prompt(self) -> str:
        """이벤트 데이터를 기반으로 프롬프트를 생성합니다."""
        if not self.use_event_data:
            return "이벤트 카테고리 정보: 이벤트 데이터를 사용하지 않습니다."
            
        prompt_parts = ["이벤트 카테고리 정보:"]
        for category, events in self.event_data.items():
            prompt_parts.append(f"\n{category} 카테고리:")
            for event_type, description in events.items():
                prompt_parts.append(f"- {event_type}: {description}")
        return "\n".join(prompt_parts)

    def _get_or_create_chat_assistant(self) -> "Chat":
        """챗봇 어시스턴트를 생성하거나 기존 것을 반환합니다."""
        CHAT_NAME = f"Noodle Machine"
        if self.chat_assistant is None:
            try:
                self.chat_assistant = self.rag_object.create_chat(
                    name=CHAT_NAME,
                    dataset_ids=[self.paper_dataset.id, self.definition_dataset.id]
                )
                
                # Pydantic 스키마를 기반으로 프롬프트 생성
                schema_prompt = generate_schema_prompt(
                    self.schema,
                    use_event_data=self.use_event_data,
                    use_sector_data=self.use_sector_data
                )
                sector_prompt = self._get_sector_prompt()
                event_prompt = self._get_event_prompt()
                
                print("=== 스키마 프롬프트 ===")
                print(schema_prompt)
                print("\n=== 섹터 프롬프트 ===")
                print(sector_prompt)
                print("\n=== 이벤트 프롬프트 ===")
                print(event_prompt)
                print("=====================")
                
                prompt_template = get_analysis_prompt(
                    schema_prompt,
                    use_event_data=self.use_event_data,
                    use_sector_data=self.use_sector_data
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
                            {"key": "content", "optional": False},
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
        """텍스트 응답을 스키마 구조로 파싱합니다."""
        analysis_dict = {
            "investment_point": {"key_points": {"points": {}}, "reasonings": {"reasons": {}}},
            "analysis": {
                "figure": {"fact": {}, "opinion": {}},
                "nonfigure": {"fact": {}, "opinion": {}},
                "material": {"fact": {}, "opinion": {}},
                "public": {"fact": {}, "opinion": {}}
            },
            "sector": {},
            "sector_indicators": {},
            "events": {}
        }
        
        lines = text.split('\n')
        current_event = {}
        event_counter = 0
        
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
                # 섹터 정보 파싱
                try:
                    sector_data = eval(content)  # 문자열을 딕셔너리로 변환
                    analysis_dict["sector"] = sector_data
                except:
                    analysis_dict["sector"]["sector"] = content
            elif num == "12":
                # 섹터 지표 파싱
                try:
                    sector_indicators_data = eval(content)  # 문자열을 딕셔너리로 변환
                    analysis_dict["sector_indicators"] = sector_indicators_data
                except:
                    analysis_dict["sector_indicators"]["0"] = content
            elif num == "13":
                # 새로운 이벤트 시작
                if current_event:
                    analysis_dict["events"][str(event_counter)] = current_event
                    event_counter += 1
                current_event = {"category": content}
            elif num == "14":
                if current_event:
                    current_event["type"] = content
            elif num == "15":
                if current_event:
                    current_event["description"] = content
            elif num == "16":
                if current_event:
                    try:
                        probability = float(content)
                    except ValueError:
                        probability = 0.5
                    current_event["probability"] = probability
        
        # 마지막 이벤트 처리
        if current_event:
            analysis_dict["events"][str(event_counter)] = current_event
        
        return analysis_dict

    def analyze_report(self, doc_name: str) -> Dict:
        """보고서를 분석합니다."""
        doc_info = parse_filename(doc_name)
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
            
            question = get_structured_output_prompt(
                use_event_data=self.use_event_data,
                use_sector_data=self.use_sector_data
            )
            
            response_stream = session.ask(
                question=question,
                stream=True,
                company=doc_info['company'],
                institution=doc_info['institution'],
                date=doc_info['date'],
                filename=doc_name,
                content=content,
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
                
                # 결과 저장 (임시 파일로 저장)
                if not os.path.exists("output"):
                    os.makedirs("output")
                    
                # 임시 파일로 저장
                temp_file = f"output/temp_{doc_info['company']}_{doc_info['date']}.json"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
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
        """모든 보고서를 분석합니다."""
        results = []
        for doc in self.get_sorted_documents():
            result = self.analyze_report(doc['original_filename'])
            results.append(result)
            
            # 임시 파일 삭제
            temp_file = f"output/temp_{result['company']}_{result['date']}.json"
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        # 모든 결과를 하나의 파일로 저장
        if results:
            if not os.path.exists("output"):
                os.makedirs("output")
            
            # JSON 파일로 저장
            with open("output/all_reports_analysis.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # 텍스트 파일로 저장
            save_to_text(results, "output/all_reports_analysis.txt")
            
            # Excel 파일로 저장
            save_to_excel(results, "output/all_reports_analysis.xlsx")
            
            print("=== 전체 분석 결과 저장 완료 ===")
            print(f"- JSON 파일: output/all_reports_analysis.json")
            print(f"- 텍스트 파일: output/all_reports_analysis.txt")
            print(f"- Excel 파일: output/all_reports_analysis.xlsx")
        
        return results 