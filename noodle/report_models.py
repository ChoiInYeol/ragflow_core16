from typing import Dict, List, Union
from pydantic import BaseModel, Field

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