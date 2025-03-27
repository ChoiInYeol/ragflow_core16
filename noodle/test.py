import os
from ragflow_sdk import RAGFlow
from ragflow_sdk.modules.chat import Chat
from datetime import datetime
import json
import pandas as pd
from typing import Dict, List, Tuple
import re
import yaml

class RagflowAnalyzer:
    def __init__(self, api_key: str, base_url: str = "http://localhost"):
        self.api_key = api_key
        self.base_url = base_url
        self.rag_object = RAGFlow(api_key=api_key, base_url=base_url)
        self.definition_dataset = self.rag_object.get_dataset(name="Definition")
        self.paper_dataset = self.rag_object.get_dataset(name="Paper")
        self.sector_data = self._load_sector_data()
        self.event_data = self._load_event_data()
        self.template_data = self._load_template()
        self.chat_assistant = None
    
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
    
    def _load_template(self) -> dict:
        for doc in self.definition_dataset.list_documents():
            if doc.name == "template.json":
                for chunk in doc.list_chunks():
                    return json.loads(chunk.content)
        return {}
    
    def parse_filename(self, filename: str) -> Dict[str, str]:
        name = os.path.splitext(filename)[0]
        print(f"=== 파일명 파싱 디버깅 ===")
        print(f"원본 파일명: {filename}")
        print(f"확장자 제거: {name}")
        
        parts = name.split('_')
        print(f"분리된 부분: {parts}")
        
        date_str = parts[-1]
        institution = parts[-3]
        company = parts[-4]
        
        print(f"추출된 정보:")
        print(f"- 날짜: {date_str}")
        print(f"- 기관: {institution}")
        print(f"- 회사: {company}")
        
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

    def _get_or_create_chat_assistant(self) -> 'Chat':
        CHAT_NAME = "Report_Analyzer"
        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'chat_config.json')
        
        if self.chat_assistant is None:
            try:
                self.chat_assistant = self.rag_object.create_chat(
                    name=CHAT_NAME,
                    dataset_ids=[self.paper_dataset.id]
                )
                
                try:
                    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                        chat_config = json.load(f)
                    
                    template_path = os.path.join(os.path.dirname(__file__), chat_config['prompt']['template_file'])
                    try:
                        with open(template_path, 'r', encoding='utf-8') as f:
                            prompt_template = f.read()
                        chat_config['prompt']['prompt'] = prompt_template
                        chat_config['prompt']['variables'].append({"key": "template", "optional": False})
                    except FileNotFoundError as e:
                        print(f"=== 파일을 찾을 수 없음: {str(e)} ===")
                        print("기본 프롬프트를 사용합니다.")
                    
                except FileNotFoundError:
                    print(f"=== 설정 파일을 찾을 수 없음: {CONFIG_PATH} ===")
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
                                {"key": "content", "optional": False},
                                {"key": "template", "optional": False}
                            ],
                            "prompt": ("""
                                당신은 금융 보고서 분석 전문가입니다. 다음 보고서를 분석하여 주요 내용을 정리해주세요.
                                문서 정보: 문서명: {filename}, 회사: {company}, 작성기관: {institution}, 작성일: {date} 
                                분석할 문서 내용: {content} 
                                다음 JSON 스키마를 정확히 따라 응답해주세요: {template}
                                
                                **반드시 지켜야 할 규칙**:
                                1. 모든 문자열은 큰따옴표(")로 감싸주세요. 작은따옴표(')는 절대 사용하지 마세요。
                                2. 숫자나 불리언 값은 따옴표 없이 작성하세요 (예: 80, true).
                                3. 모든 필드는 반드시 포함되며, 값이 없더라도 빈 배열([]) 또는 빈 객체({})로 유지하세요.
                                4. JSON 형식을 엄격히 준수하며, 유효한 JSON만 반환하세요.
                                5. 불필요한 설명, 주석, 텍스트는 절대 포함시키지 마세요.
                                6. 리스트(예: key_points, reasonings, fact, opinion, sector_indicators, events)는 반드시 배열([])로 작성하세요.
                                7. 배열 내 요소는 인덱스 키("0", "1" 등)를 사용하지 말고, 순서대로 나열하세요.
                                8. 배열의 마지막 요소 뒤에 쉼표(,)를 넣지 마세요.
                                9. events는 배열([])로 작성하며, 각 이벤트는 객체({})로 구성하세요.
                                
                                **예시**:
                                {
                                    "investment_point": {
                                        "key_points": ["포인트1", "포인트2", "포인트3"],
                                        "reasonings": ["근거1", "근거2", "근거3"]
                                    },
                                    "analysis": {
                                        "figure": {
                                            "fact": ["사실1", "사실2"],
                                            "opinion": ["의견1", "의견2"]
                                        }
                                    },
                                    "sector_indicators": ["지표1", "지표2"],
                                    "events": [
                                        {
                                            "category": "카테고리",
                                            "type": "타입",
                                            "description": "설명",
                                            "probability": 80
                                        }
                                    ]
                                }
                            """)
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
    
    def _fix_json_structure(self, data: dict) -> dict:
        def convert_to_list(obj):
            if isinstance(obj, dict):
                if all(k.isdigit() for k in obj.keys()):
                    return [obj[k] for k in sorted(obj.keys(), key=int)]
                return {k: convert_to_list(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_list(item) for item in obj]
            return obj
        return convert_to_list(data)
    
    def _extract_json_from_response(self, response: str) -> dict:
        try:
            json_str = response.strip()
            if json_str.startswith('**ERROR**'):
                print(f"=== 오류 응답 ===")
                print(json_str)
                return {}
            
            json_start = json_str.find("{")
            json_end = json_str.rfind("}") + 1
            if json_start == -1 or json_end <= json_start:
                print("=== 유효한 JSON 형식을 찾을 수 없음 ===")
                print("전체 응답:")
                print(json_str)
                return {}
            
            json_content = json_str[json_start:json_end]
            json_content = self._preprocess_json(json_content)
            
            try:
                result = json.loads(json_content)
                if not result:
                    print("=== JSON 파싱 결과가 비어있음 ===")
                    print("JSON 내용:")
                    print(json_content)
                    return {}
                
                result = self._fix_json_structure(result)
                
                required_fields = ['investment_point', 'analysis', 'sector_indicators', 'events']
                missing_fields = [field for field in required_fields if field not in result]
                if missing_fields:
                    print(f"=== 필수 필드 누락 ===")
                    print(f"누락된 필드: {missing_fields}")
                    print("JSON 내용:")
                    print(json_content)
                    return {}
                
                if not self._validate_json_structure(result):
                    print("=== JSON 구조 검증 실패 ===")
                    print("JSON 내용:")
                    print(json_content)
                    return {}
                    
                print("=== 추출 및 보정된 데이터 ===")
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return result
                
            except json.JSONDecodeError as e:
                print(f"=== JSON 파싱 오류 ===")
                print(f"오류 메시지: {str(e)}")
                print("JSON 내용:")
                print(json_content)
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
        json_content = re.sub(r'[\n\t\r]', '', json_content)
        json_content = re.sub(r"([^\\])'", r'\1"', json_content)
        json_content = re.sub(r'(\d+):', r'"\1":', json_content)
        json_content = re.sub(r"'\s*:\s*'", '": "', json_content)
        json_content = re.sub(r',\s*([}\]])', r'\1', json_content)
        json_content = re.sub(r':\s*"(\d+\.?\d*)"', r': \1', json_content)
        json_content = re.sub(r'\s+', ' ', json_content)
        json_content = re.sub(r'""', '"', json_content)
        json_content = re.sub(r'\\"', '"', json_content)
        json_content = re.sub(r'""', 'null', json_content)
        
        try:
            json.loads(json_content)
        except json.JSONDecodeError as e:
            print(f"=== JSON 전처리 후 유효성 검사 실패 ===")
            print(f"오류 메시지: {str(e)}")
            print("전처리된 JSON:")
            print(json_content)
            raise
        
        return json_content
    
    def _validate_json_structure(self, data: dict) -> bool:
        try:
            if not isinstance(data.get('investment_point'), dict):
                return False
            if not isinstance(data['investment_point'].get('key_points'), list):
                return False
            if not isinstance(data['investment_point'].get('reasonings'), list):
                return False
            
            if not isinstance(data.get('analysis'), dict):
                return False
            for category in ['figure', 'nonfigure', 'material', 'public']:
                if not isinstance(data['analysis'].get(category), dict):
                    return False
                for subcategory in ['fact', 'opinion']:
                    if not isinstance(data['analysis'][category].get(subcategory), list):
                        return False
            
            if not isinstance(data.get('sector_indicators'), list):
                return False
            
            if not isinstance(data.get('events'), list):
                return False
            for event in data['events']:
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
        session = assistant.create_session(f"Analysis for {doc_name}")
        
        try:
            print("=== 챗봇 응답 처리 시작 ===")
            
            # 프롬프트 직접 생성
            prompt_template = assistant.config['prompt']['prompt']
            prompt = prompt_template.format(
                filename=doc_name,
                company=doc_info['company'],
                institution=doc_info['institution'],
                date=doc_info['date'],
                content=content,
                template=json.dumps(self.template_data, ensure_ascii=False)
            )
            
            # 분석 실행
            response_stream = session.ask(
                question=prompt,
                stream=True
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
            
            analysis_result = self._extract_json_from_response(result_content)
            if not analysis_result:
                print("=== 분석 결과가 비어있음 ===")
                return {**doc_info, "analysis": {}, "error": "분석 결과 추출 실패"}
                
            result = {**doc_info, "analysis": analysis_result}
            
            try:
                with open("test_result.json", 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                self.save_to_excel([result], "test_result.xlsx")
                print("=== 결과 저장 완료 ===")
            except Exception as e:
                print(f"=== 결과 저장 중 오류 발생 ===")
                print(str(e))
            
            return result
            
        except Exception as e:
            print(f"=== 분석 중 오류 발생 ===")
            print(str(e))
            if str(e) == "'message'":
                print("서버에서 오류 메시지를 받았습니다. 응답을 확인해주세요.")
            return {**doc_info, "analysis": {}, "error": str(e)}
    
    def analyze_all_reports(self) -> List[Dict]:
        results = []
        for doc in self.get_sorted_documents():
            result = self.analyze_report(doc['original_filename'])
            results.append(result)
        return results
    
    def save_to_excel(self, results: List[Dict], output_file: str = "analysis_results.xlsx"):
        excel_data = []
        for result in results:
            row = {
                '회사명': result['company'],
                '작성기관': result['institution'],
                '작성일': result['date'],
                '파일명': result['original_filename']
            }
            
            analysis = result.get('analysis', {})
            investment_point = analysis.get('investment_point', {})
            row['투자포인트_주요포인트'] = '\n'.join(investment_point.get('key_points', []))
            row['투자포인트_근거'] = '\n'.join(investment_point.get('reasonings', []))
            
            analysis_data = analysis.get('analysis', {})
            for category in ['figure', 'nonfigure', 'material', 'public']:
                cat_data = analysis_data.get(category, {})
                row[f'분석_{category}_사실'] = '\n'.join(cat_data.get('fact', []))
                row[f'분석_{category}_의견'] = '\n'.join(cat_data.get('opinion', []))
            
            row['섹터지표'] = '\n'.join(analysis.get('sector_indicators', []))
            
            events = analysis.get('events', [])
            for i, event in enumerate(events, 1):
                row[f'이벤트{i}_카테고리'] = event.get('category', '')
                row[f'이벤트{i}_타입'] = event.get('type', '')
                row[f'이벤트{i}_설명'] = event.get('description', '')
                row[f'이벤트{i}_확률'] = str(event.get('probability', ''))
            
            excel_data.append(row)
        
        df = pd.DataFrame(excel_data)
        df.to_excel(output_file, index=False)

def main():
    API_KEY = "ragflow-I0Y2ViN2M2MDU3OTExZjA5N2ZhMTIxOD"
    analyzer = RagflowAnalyzer(API_KEY)
    
    TEST_MODE = True
    TEST_FILE = "기대할_것이_많다_HD현대중공업_미래에셋증권_표지_25.02.18.pdf"
    
    if TEST_MODE:
        print(f"테스트 모드: {TEST_FILE} 파일 분석 중...")
        result = analyzer.analyze_report(TEST_FILE)
        analyzer.save_to_excel([result], "test_result.xlsx")
        
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
        analyzer.save_to_excel(results)
        print(f"분석 완료: {len(results)}개 보고서 처리됨")

if __name__ == "__main__":
    main()