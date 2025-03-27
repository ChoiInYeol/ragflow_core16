def generate_schema_prompt(schema: dict, use_event_data: bool = True, use_sector_data: bool = True) -> str:
    """Pydantic 스키마를 기반으로 간단한 프롬프트를 생성합니다."""
    prompt_parts = []
    
    def process_field(field: dict, parent_name: str = "") -> None:
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
            # 이벤트와 섹터 데이터 사용 여부에 따라 필드 필터링
            if prop_name == "events" and not use_event_data:
                continue
            if prop_name == "sector_indicators" and not use_sector_data:
                continue
            process_field(prop_value, prop_name)
    
    return "\n".join(prompt_parts)

def get_analysis_prompt(schema_prompt: str, use_event_data: bool = True, use_sector_data: bool = True) -> str:
    """분석을 위한 프롬프트 템플릿을 반환합니다."""
    base_prompt = (
        "당신은 세계 최고의 금융 전문가이자 분석가입니다. 증권사의 애널리스트 보고서를 분석하여 투자자들이 이해하기 쉽게 정보를 추출하고 정리하세요.\n\n"
        "분석 시 다음 사항을 참고하세요:\n"
        "1. 보고서의 핵심 내용은 따옴표로 인용하세요.\n"
        "2. 수치 데이터는 정확하게 인용하세요.\n"
        "3. 애널리스트의 의견과 예측도 인용하세요.\n"
    )
    
    base_prompt += (
        "분석할 때 다음 스키마 정보를 참고하세요:\n\n"
        f"{schema_prompt}\n\n"
        "문서 정보:\n"
        "- 문서명: {filename}\n"
        "- 회사: {company}\n"
        "- 작성기관: {institution}\n"
        "- 작성일: {date}\n\n"
    )
    
    base_prompt += (
        "분석할 문서 내용:\n"
        "{content}\n\n"
        "반드시 보고서에 명시된 내용만 활용하며, 새로운 정보를 추가하지 마세요. 모든 구체적인 내용은 따옴표로 인용하세요."
    )
    
    return base_prompt

def get_structured_output_prompt(use_event_data: bool = True, use_sector_data: bool = True) -> str:
    """구조화된 출력을 위한 프롬프트를 반환합니다."""
    prompt_parts = [
        "아래 구조에 맞게 정보를 추출하고 분석하세요. 번호와 내용만 출력하며, 추가 설명은 넣지 마세요.\n\n",
        "1. investment_point - key_points: 투자 포인트의 주요 내용\n",
        "2. investment_point - reasonings: 투자 포인트에 대한 근거 (수치 포함)\n",
        "3. analysis - figure - fact: 정량적 데이터 기반 사실 (수치 포함)\n",
        "4. analysis - figure - opinion: 정량적 데이터 기반 예측\n",
        "5. analysis - nonfigure - fact: 정성적 정보 기반 사실\n",
        "6. analysis - nonfigure - opinion: 정성적 정보 기반 예측\n",
        "7. analysis - material - fact: 중요 이벤트 기반 사실\n",
        "8. analysis - material - opinion: 중요 이벤트 기반 예측\n",
        "9. analysis - public - fact: 공개 정보 기반 사실\n",
        "10. analysis - public - opinion: 공개 정보 기반 예측\n"
        "11. sector: 섹터 정보\n"
        "12. sector_indicators: 섹터별 주요 지표 (수치 포함)\n"
        "13. events - category: 이벤트 카테고리\n"
        "14. events - type: 이벤트 요약\n"
        "15. events - description: 이벤트 설명 (구체적인 내용과 근거 포함)\n",
        "16. events - probability: 발생 확률 (0~1)\n"
    ]

    
    return "".join(prompt_parts) 