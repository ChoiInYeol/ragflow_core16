import os
import json
from pathlib import Path
from typing import Dict, Any, List, Union
import shutil
from datetime import datetime
from ragflow_sdk import RAGFlow
from settings import API_KEY, BASE_URL

def ensure_directory(path: Union[str, Path]) -> None:
    """
    디렉토리가 존재하지 않으면 생성합니다.
    
    Args:
        path (Union[str, Path]): 생성할 디렉토리 경로
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """
    데이터를 JSON 파일로 저장합니다.
    
    Args:
        data (Dict[str, Any]): 저장할 데이터
        filepath (Union[str, Path]): 저장할 파일 경로
    """
    filepath = Path(filepath)
    ensure_directory(filepath.parent)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    JSON 파일을 로드합니다.
    
    Args:
        filepath (Union[str, Path]): 로드할 파일 경로
        
    Returns:
        Dict[str, Any]: 로드된 데이터
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
        
def get_pdf_files(directory: Union[str, Path]) -> List[Path]:
    """
    디렉토리에서 PDF 파일 목록을 가져옵니다.
    
    Args:
        directory (Union[str, Path]): 검색할 디렉토리
        
    Returns:
        List[Path]: PDF 파일 경로 목록
    """
    directory = Path(directory)
    return list(directory.rglob("*.pdf"))
    
def get_json_files(directory: Union[str, Path]) -> List[Path]:
    """
    디렉토리에서 JSON 파일 목록을 가져옵니다.
    
    Args:
        directory (Union[str, Path]): 검색할 디렉토리
        
    Returns:
        List[Path]: JSON 파일 경로 목록
    """
    directory = Path(directory)
    return list(directory.rglob("*.json"))
    
def get_excel_files(directory: Union[str, Path]) -> List[Path]:
    """
    디렉토리에서 Excel 파일 목록을 가져옵니다.
    
    Args:
        directory (Union[str, Path]): 검색할 디렉토리
        
    Returns:
        List[Path]: Excel 파일 경로 목록
    """
    directory = Path(directory)
    return list(directory.rglob("*.xlsx"))
    
def get_text_files(directory: Union[str, Path]) -> List[Path]:
    """
    디렉토리에서 텍스트 파일 목록을 가져옵니다.
    
    Args:
        directory (Union[str, Path]): 검색할 디렉토리
        
    Returns:
        List[Path]: 텍스트 파일 경로 목록
    """
    directory = Path(directory)
    return list(directory.rglob("*.txt"))
    
def read_text_file(filepath: Union[str, Path]) -> str:
    """
    텍스트 파일을 읽습니다.
    
    Args:
        filepath (Union[str, Path]): 읽을 파일 경로
        
    Returns:
        str: 파일 내용
    """
    filepath = Path(filepath)
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()
        
def write_text_file(content: str, filepath: Union[str, Path]) -> None:
    """
    텍스트 파일을 씁니다.
    
    Args:
        content (str): 쓸 내용
        filepath (Union[str, Path]): 쓸 파일 경로
    """
    filepath = Path(filepath)
    ensure_directory(filepath.parent)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
        
def append_text_file(content: str, filepath: Union[str, Path]) -> None:
    """
    텍스트 파일에 내용을 추가합니다.
    
    Args:
        content (str): 추가할 내용
        filepath (Union[str, Path]): 파일 경로
    """
    filepath = Path(filepath)
    ensure_directory(filepath.parent)
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(content)
        
def delete_file(filepath: Union[str, Path]) -> None:
    """
    파일을 삭제합니다.
    
    Args:
        filepath (Union[str, Path]): 삭제할 파일 경로
    """
    filepath = Path(filepath)
    if filepath.exists():
        filepath.unlink()
        
def upload_pdf_to_dataset(pdf_path: Union[str, Path], dataset_name: str) -> None:
    """
    PDF 파일을 데이터셋에 업로드합니다.
    
    Args:
        pdf_path (Union[str, Path]): 업로드할 PDF 파일 경로
        dataset_name (str): 대상 데이터셋 이름
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        
    # 파일명에서 메타데이터 추출
    filename = pdf_path.name
    name_parts = filename.split('_')
    if len(name_parts) < 4:
        raise ValueError(f"잘못된 파일명 형식입니다: {filename}")
        
    company = name_parts[-4]
    institution = name_parts[-3]
    date_str = name_parts[-1].replace('.pdf', '')
    
    # 날짜 형식 변환
    try:
        date_obj = datetime.strptime(date_str, '%y.%m.%d')
        formatted_date = date_obj.strftime('%Y-%m-%d')
    except ValueError:
        raise ValueError(f"잘못된 날짜 형식입니다: {date_str}")
    
    # 데이터셋에 파일 업로드
    rag = RAGFlow(api_key=API_KEY, base_url=BASE_URL)
    dataset = rag.get_dataset(name=dataset_name)
    
    # 파일 업로드
    with open(pdf_path, 'rb') as f:
        document = {
            "display_name": filename,
            "blob": f.read(),
            "meta_fields": {
                'company': company,
                'institution': institution,
                'date': formatted_date
            },
            "chunk_method": "paper",
            "parser_config": {
                "raptor": {"user_raptor": False}
            }
        }
        dataset.upload_documents([document])
        
    # 업로드된 문서 찾기
    documents = dataset.list_documents(keywords=filename)
    if not documents:
        raise ValueError(f"업로드된 문서를 찾을 수 없습니다: {filename}")
        
    # 문서 파싱 설정 및 실행
    document = documents[0]
    document.update({
        "chunk_method": "paper",
        "parser_config": {
            "raptor": {"user_raptor": False}
        }
    })
    
    # 비동기 파싱 실행
    dataset.async_parse_documents([document.id])
    print(f"문서 파싱 시작: {filename}")

def process_paper_directory(paper_dir: Union[str, Path], dataset_name: str) -> None:
    """
    paper 디렉토리의 모든 PDF 파일을 처리합니다.
    
    Args:
        paper_dir (Union[str, Path]): paper 디렉토리 경로
        dataset_name (str): 대상 데이터셋 이름
    """
    paper_dir = Path(paper_dir)
    if not paper_dir.exists():
        raise FileNotFoundError(f"paper 디렉토리를 찾을 수 없습니다: {paper_dir}")
        
    pdf_files = get_pdf_files(paper_dir)
    total_files = len(pdf_files)
    
    print(f"총 {total_files}개의 PDF 파일을 처리합니다.")
    
    for i, pdf_path in enumerate(pdf_files, 1):
        try:
            print(f"처리 중: {i}/{total_files} - {pdf_path.name}")
            upload_pdf_to_dataset(pdf_path, dataset_name)
            print(f"업로드 완료: {pdf_path.name}")
        except Exception as e:
            print(f"오류 발생: {pdf_path.name} - {str(e)}")
            
    print("모든 파일 처리 완료") 