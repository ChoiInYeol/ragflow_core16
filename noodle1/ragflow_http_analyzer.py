import os
import requests
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

class RagflowHttpAnalyzer:
    def __init__(self, api_key: str, base_url: str = "http://localhost:9380"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        self.schema = ReportAnalysis.model_json_schema()

    def _get_dataset_id(self, dataset_name: str) -> str:
        """데이터셋 ID를 조회합니다."""
        response = requests.get(
            f"{self.base_url}/api/v1/datasets",
            headers=self.headers
        )
        response.raise_for_status()
        datasets = response.json()
        
        for dataset in datasets:
            if dataset['name'] == dataset_name:
                return dataset['id']
        raise ValueError(f"데이터셋을 찾을 수 없음: {dataset_name}")

    def _get_document_chunks(self, dataset_id: str, doc_name: str) -> List[str]:
        """문서의 청크를 조회합니다."""
        response = requests.get(
            f"{self.base_url}/api/v1/datasets/{dataset_id}/documents/{doc_name}/chunks",
            headers=self.headers
        )
        response.raise_for_status()
        chunks = response.json()
        return [chunk['content'] for chunk in chunks]

    def parse_filename(self, filename: str) -> Dict[str, str]:
        """파일명을 파싱하여 문서 정보를 추출합니다."""
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
        """정렬된 문서 목록을 조회합니다."""
        try:
            dataset_id = self._get_dataset_id("Paper")
            response = requests.get(
                f"{self.base_url}/api/v1/datasets/{dataset_id}/documents",
                headers=self.headers
            )
            response.raise_for_status()
            documents = response.json()
            
            parsed_docs = []
            for doc in documents:
                parsed_info = self.parse_filename(doc['name'])
                parsed_docs.append(parsed_info)
            
            return sorted(parsed_docs, key=lambda x: x['date'], reverse=True)
        except Exception as e:
            print(f"문서 목록 조회 실패: {str(e)}")
            return []

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
        """보고서를 분석합니다."""
        doc_info = self.parse_filename(doc_name)
        print(f"=== 문서 정보 ===")
        print(f"파일명: {doc_name}")
        print(f"회사명: {doc_info['company']}")
        print(f"작성기관: {doc_info['institution']}")
        print(f"작성일: {doc_info['date']}")
        
        try:
            dataset_id = self._get_dataset_id("Paper")
            chunks = self._get_document_chunks(dataset_id, doc_name)
            
            if not chunks:
                print(f"문서 청크를 찾을 수 없음: {doc_name}")
                return {**doc_info, "analysis": {}, "error": "문서 청크를 찾을 수 없음"}
            
            content = "".join(chunks)
            print(f"=== 문서 청크 수: {len(chunks)} ===")
            
            # RAGFlow API를 통한 분석 요청
            response = requests.post(
                f"{self.base_url}/api/v1/chat/completions",
                headers=self.headers,
                json={
                    "messages": [
                        {
                            "role": "system",
                            "content": "당신은 세계 최고의 금융 전문가이자 분석가입니다. 증권사의 애널리스트 보고서를 분석하여 투자자들이 이해하기 쉽게 정보를 추출하고 정리하세요."
                        },
                        {
                            "role": "user",
                            "content": f"문서 정보:\n- 문서명: {doc_name}\n- 회사: {doc_info['company']}\n- 작성기관: {doc_info['institution']}\n- 작성일: {doc_info['date']}\n\n분석할 문서 내용:\n{content}"
                        }
                    ],
                    "model": "gpt-4",
                    "temperature": 0.25,
                    "max_tokens": 2048
                }
            )
            response.raise_for_status()
            result = response.json()
            
            # 응답 파싱 및 결과 저장
            analysis_result = self._parse_text_to_schema(result['choices'][0]['message']['content'])
            final_result = {**doc_info, "analysis": analysis_result}
            
            # 결과 저장
            with open("test_result.json", 'w', encoding='utf-8') as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)
            self.save_to_text([final_result], "test_result.txt")
            self.save_to_excel([final_result], "test_result.xlsx")
            print("=== 결과 저장 완료 ===")
            
            return final_result
            
        except Exception as e:
            print(f"=== 분석 중 오류 발생 ===")
            print(str(e))
            return {**doc_info, "error": str(e)}

    def analyze_all_reports(self) -> List[Dict]:
        """모든 보고서를 분석합니다."""
        results = []
        for doc in self.get_sorted_documents():
            result = self.analyze_report(doc['original_filename'])
            results.append(result)
        return results

    def save_to_text(self, results: List[Dict], output_file: str = "analysis_results.txt") -> None:
        """분석 결과를 텍스트 파일로 저장합니다."""
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
        """분석 결과를 Excel 파일로 저장합니다."""
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
    analyzer = RagflowHttpAnalyzer(API_KEY)
    
    TEST_MODE = False
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
        analyzer.save_to_text(results, "output/all_results.txt")
        analyzer.save_to_excel(results, "output/all_results.xlsx")
        print(f"분석 완료: {len(results)}개 보고서 처리됨")

if __name__ == "__main__":
    main() 