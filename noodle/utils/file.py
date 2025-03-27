import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List

def parse_filename(filename: str) -> Dict[str, str]:
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

def save_to_text(results: List[Dict], output_file: str = "analysis_results.txt") -> None:
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

def save_to_excel(results: List[Dict], output_file: str = "analysis_results.xlsx") -> None:
    """분석 결과를 엑셀 파일로 저장합니다."""
    data = []
    for result in results:
        if 'error' in result:
            data.append({
                '회사명': result['company'],
                '작성기관': result['institution'],
                '작성일': result['date'],
                '원본 파일명': result['original_filename'],
                '오류': result['error']
            })
            continue
            
        analysis = result['analysis']
        
        # 기본 정보
        row = {
            '회사명': result['company'],
            '작성기관': result['institution'],
            '작성일': result['date'],
            '원본 파일명': result['original_filename']
        }
        
        # 투자 포인트 처리
        key_points = analysis["investment_point"]["key_points"]["points"]
        reasonings = analysis["investment_point"]["reasonings"]["reasons"]
        row['투자 포인트 - 주요 내용'] = key_points.get('0', '').replace('investment_point - key_points: ', '')
        row['투자 포인트 - 근거'] = reasonings.get('0', '').replace('investment_point - reasonings: ', '')
        
        # 분석 내용 처리
        figure_facts = analysis["analysis"]["figure"]["fact"]
        figure_opinions = analysis["analysis"]["figure"]["opinion"]
        nonfigure_facts = analysis["analysis"]["nonfigure"]["fact"]
        nonfigure_opinions = analysis["analysis"]["nonfigure"]["opinion"]
        material_facts = analysis["analysis"]["material"]["fact"]
        material_opinions = analysis["analysis"]["material"]["opinion"]
        public_facts = analysis["analysis"]["public"]["fact"]
        public_opinions = analysis["analysis"]["public"]["opinion"]
        
        row['정량 분석 - 사실'] = figure_facts.get('0', '').replace('analysis - figure - fact: ', '')
        row['정량 분석 - 예측'] = figure_opinions.get('0', '').replace('analysis - figure - opinion: ', '')
        row['정성 분석 - 사실'] = nonfigure_facts.get('0', '').replace('analysis - nonfigure - fact: ', '')
        row['정성 분석 - 예측'] = nonfigure_opinions.get('0', '').replace('analysis - nonfigure - opinion: ', '')
        row['중요 이벤트 - 사실'] = material_facts.get('0', '').replace('analysis - material - fact: ', '')
        row['중요 이벤트 - 예측'] = material_opinions.get('0', '').replace('analysis - material - opinion: ', '')
        row['공개 정보 - 사실'] = public_facts.get('0', '').replace('analysis - public - fact: ', '')
        row['공개 정보 - 예측'] = public_opinions.get('0', '').replace('analysis - public - opinion: ', '')
        
        # 섹터 정보 처리
        sector = analysis.get("sector", {})
        row['섹터'] = sector.get('sector', '')
        row['하위 섹터'] = sector.get('sub_sector', '')
        
        # 섹터 지표 처리
        sector_indicators = analysis.get("sector_indicators", {})
        row['섹터 지표'] = sector_indicators.get('0', '')
        row['추가 섹터 지표'] = sector_indicators.get('1', '')
        
        # 이벤트 처리
        events = analysis.get("events", {})
        if events:
            first_event = next(iter(events.values()))
            row['이벤트 카테고리'] = first_event.get('category', '')
            row['이벤트 요약'] = first_event.get('type', '')
            row['이벤트 설명'] = first_event.get('description', '')
            row['이벤트 확률'] = first_event.get('probability', '')
        else:
            row['이벤트 카테고리'] = ''
            row['이벤트 요약'] = ''
            row['이벤트 설명'] = ''
            row['이벤트 확률'] = ''
        
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"엑셀 파일 저장 완료: {output_file} (크기: {df.shape})") 