from analyzers.ragflow import RagflowAnalyzer
from utils.file import save_to_excel

def main():
    API_KEY = "ragflow-I0Y2ViN2M2MDU3OTExZjA5N2ZhMTIxOD"
    
    # 이벤트와 섹터 데이터 사용 여부 설정
    USE_EVENT_DATA = False  # 이벤트 데이터 사용 여부
    USE_SECTOR_DATA = False  # 섹터 데이터 사용 여부
    
    analyzer = RagflowAnalyzer(
        api_key=API_KEY,
        use_event_data=USE_EVENT_DATA,
        use_sector_data=USE_SECTOR_DATA
    )
    
    TEST_MODE = False
    TEST_FILE = "기대할_것이_많다_HD현대중공업_미래에셋증권_표지_25.02.18.pdf"
    
    if TEST_MODE:
        print(f"테스트 모드: {TEST_FILE} 파일 분석 중...")
        print(f"이벤트 데이터 사용: {'예' if USE_EVENT_DATA else '아니오'}")
        print(f"섹터 데이터 사용: {'예' if USE_SECTOR_DATA else '아니오'}")
        
        result = analyzer.analyze_report(TEST_FILE)
        
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
            
            # 섹터 정보 출력
            print("\n=== 섹터 정보 ===")
            for key, value in analysis["sector"].items():
                print(f"sector: {value}")
            
            print("\n=== 섹터 지표 ===")
            for key, value in analysis["sector_indicators"].items():
                print(f"sector_indicators: {value}")
            
            print("\n=== 이벤트 정보 ===")
            for key, event in analysis["events"].items():
                print(f"events - category: {event['category']}")
                print(f"events - type: {event['type']}")
                print(f"events - description: {event['description']}")
                print(f"events - probability: {event['probability']}")
                print("---")
    else:
        print("전체 보고서 분석 중...")
        results = analyzer.analyze_all_reports()
        print(f"분석 완료: {len(results)}개 보고서 처리됨")
        
        # 전체 분석 결과를 Excel 파일로 저장
        save_to_excel(results, "output/all_reports_analysis.xlsx")
        print("전체 분석 결과가 output/all_reports_analysis.xlsx 파일로 저장되었습니다.")

if __name__ == "__main__":
    main() 