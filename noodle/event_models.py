from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class ReportMetadata(BaseModel):
    """보고서 메타데이터를 나타내는 모델"""
    company_name: str = Field(description="회사명")
    institution: str = Field(description="증권사/기관명")
    date: str = Field(description="보고서 작성일")
    title: str = Field(description="보고서 제목")

class EventImpact(BaseModel):
    """이벤트의 영향력을 나타내는 모델"""
    market_reaction: str = Field(description="예상되는 시장 반응")
    business_impact: str = Field(description="사업에 미치는 영향")
    risk_factors: List[str] = Field(description="위험 요인 목록")

class SubsequentEvent(BaseModel):
    """후속 이벤트를 나타내는 모델"""
    description: str = Field(description="후속 이벤트 설명")
    probability: float = Field(description="발생 확률 (0~1)", ge=0.0, le=1.0)
    impact: Optional[EventImpact] = Field(description="예상 영향", default=None)

class CoreEvent(BaseModel):
    """핵심 이벤트를 나타내는 모델"""
    event_id: str = Field(description="이벤트 고유 ID")
    category: str = Field(description="이벤트 카테고리")
    type: str = Field(description="이벤트 유형")
    description: str = Field(description="이벤트 설명")
    source_quote: str = Field(description="원문 인용")
    impact: EventImpact = Field(description="이벤트 영향")
    subsequent_events: List[SubsequentEvent] = Field(description="후속 이벤트 목록")

class EventAnalysis(BaseModel):
    """이벤트 분석 결과를 나타내는 모델"""
    metadata: ReportMetadata = Field(description="보고서 메타데이터")
    core_events: List[CoreEvent] = Field(description="핵심 이벤트 목록") 