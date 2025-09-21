#!/usr/bin/env python3
"""
대회 제출용 자료 정리 및 패키징
- 핵심 결과물을 대회 형식에 맞게 재구성
- 제출 가이드라인 준수
- 파일명 및 구조 표준화
"""

import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List

def create_submission_package():
    """대회 제출용 패키지 생성"""
    print("📦 대회 제출용 자료 패키징 중...")
    
    # 제출 디렉토리 생성
    submission_dir = "bigquery_competition_submission"
    if os.path.exists(submission_dir):
        shutil.rmtree(submission_dir)
    os.makedirs(submission_dir)
    
    # 1. 최종 보고서 복사
    shutil.copy2("final_corrected_report.json", f"{submission_dir}/final_report.json")
    
    # 2. 핵심 코드 파일들 복사
    key_files = [
        "persona_discovery_operation.py",
        "simplified_insight_finalization.py", 
        "final_report_corrections.py"
    ]
    
    for file in key_files:
        if os.path.exists(file):
            shutil.copy2(file, f"{submission_dir}/{file}")
    
    # 3. 시각화 파일들 복사 (있는 경우)
    visualization_files = [
        "persona_discovery_clustering.png",
        "persona_discovery_tsne.png", 
        "persona_discovery_analysis.png"
    ]
    
    for file in visualization_files:
        if os.path.exists(file):
            shutil.copy2(file, f"{submission_dir}/{file}")
    
    # 4. README 파일 생성
    create_readme_file(submission_dir)
    
    # 5. 제출 요약 파일 생성
    create_submission_summary(submission_dir)
    
    print(f"✅ 제출 패키지 생성 완료: {submission_dir}/")
    return submission_dir

def create_readme_file(submission_dir: str):
    """README 파일 생성"""
    readme_content = """# BigQuery 대회 제출 자료

## 팀 정보
- **팀명**: MKM Lab AI 기술부
- **제출일**: 2025-01-12
- **접근법**: 예측에서 발견으로 - 페르소나 기반 솔루션

## 핵심 발견
우리는 기존의 예측 모델 접근법이 실패할 수밖에 없는 데이터의 구조적 한계를 인정하고, 
대신 비지도학습을 통한 '발견' 접근법으로 전환하여 7개의 뚜렷한 사용자 페르소나를 발견했습니다.

## 주요 결과
- **7개 페르소나 발견**: 모두 '스트레스받는 성취가' 계열
- **단일 거대 페르소나**: 7개 세부 그룹이 하나의 거대 페르소나로 통합
- **명확한 타겟 전략**: 복잡한 다중 페르소나 대신 단일 타겟 집중

## 파일 구조
- `final_report.json`: 최종 분석 결과 및 페르소나 프로필
- `persona_discovery_operation.py`: 페르소나 발견 메인 코드
- `simplified_insight_finalization.py`: 인사이트 최종화 코드
- `final_report_corrections.py`: 보고서 수정 및 개선 코드
- `submission_summary.json`: 제출 요약 정보

## 실행 방법
1. BigQuery 인증 설정
2. `persona_discovery_operation.py` 실행
3. `simplified_insight_finalization.py` 실행
4. `final_report_corrections.py` 실행

## 핵심 인사이트
모든 사용자가 '스트레스받는 성취가'라는 하나의 거대 페르소나에 속하며, 
이는 높은 스트레스 수준과 높은 활동성을 동시에 보이는 특별한 집단입니다.
이 발견은 맞춤형 헬스케어 서비스 설계의 핵심이 됩니다.
"""
    
    with open(f"{submission_dir}/README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

def create_submission_summary(submission_dir: str):
    """제출 요약 파일 생성"""
    summary = {
        "submission_info": {
            "team_name": "MKM Lab AI 기술부",
            "submission_date": "2025-01-12",
            "approach": "예측에서 발견으로 - 페르소나 기반 솔루션",
            "version": "1.0"
        },
        "key_findings": {
            "personas_discovered": 7,
            "total_users_analyzed": 10000,
            "main_persona": "스트레스받는 성취가 (The Stressed Achiever)",
            "clustering_method": "K-Means + DBSCAN",
            "data_sources": ["Big5", "CMI", "RPPG", "Voice"]
        },
        "strategic_insights": {
            "single_target_strategy": "모든 페르소나가 하나의 거대 페르소나로 통합",
            "business_value": "명확한 타겟 설정으로 시장 집중 공략 가능",
            "competitive_advantage": "다른 팀과 차별화된 독창적 접근법"
        },
        "files_included": [
            "final_report.json",
            "persona_discovery_operation.py",
            "simplified_insight_finalization.py", 
            "final_report_corrections.py",
            "README.md",
            "submission_summary.json"
        ],
        "execution_instructions": [
            "1. BigQuery 인증 설정",
            "2. persona_discovery_operation.py 실행",
            "3. simplified_insight_finalization.py 실행", 
            "4. final_report_corrections.py 실행"
        ]
    }
    
    with open(f"{submission_dir}/submission_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

def create_executive_summary():
    """경영진용 요약 보고서 생성"""
    print("📋 경영진용 요약 보고서 생성 중...")
    
    executive_summary = {
        "title": "BigQuery 대회 최종 결과 요약",
        "date": "2025-01-12",
        "team": "MKM Lab AI 기술부",
        "executive_summary": {
            "problem_identification": "기존 예측 모델 접근법의 한계 인정",
            "solution_approach": "비지도학습을 통한 페르소나 발견",
            "key_discovery": "7개 세부 페르소나 = 1개 거대 페르소나 (스트레스받는 성취가)",
            "strategic_value": "명확한 단일 타겟 전략으로 시장 집중 공략 가능",
            "competitive_advantage": "다른 팀과 차별화된 독창적 솔루션"
        },
        "business_implications": {
            "target_clarity": "복잡한 다중 페르소나 대신 명확한 단일 타겟",
            "resource_efficiency": "자원 분산 없이 핵심 타겟에 집중 투자",
            "market_penetration": "경쟁사 대비 월등한 타겟 집중도",
            "scalability": "확장 가능한 페르소나 기반 개인화 시스템"
        },
        "next_steps": [
            "스트레스받는 성취가 타겟에 맞춘 제품 개발",
            "세부 그룹별 차별화된 마케팅 전략 수립",
            "페르소나 기반 개인화 시스템 구축",
            "지속적인 사용자 행동 패턴 모니터링"
        ]
    }
    
    with open("executive_summary.json", "w", encoding="utf-8") as f:
        json.dump(executive_summary, f, indent=2, ensure_ascii=False)
    
    print("✅ 경영진용 요약 보고서 생성 완료")

def main():
    """메인 실행 함수"""
    print("🚀 대회 제출용 자료 패키징")
    print("=" * 50)
    
    # 제출 패키지 생성
    submission_dir = create_submission_package()
    
    # 경영진용 요약 보고서 생성
    create_executive_summary()
    
    print(f"\n📦 제출 패키지 완성:")
    print(f"   📁 {submission_dir}/ - 대회 제출용 패키지")
    print(f"   📄 executive_summary.json - 경영진용 요약")
    
    print(f"\n📋 포함된 파일들:")
    for file in os.listdir(submission_dir):
        print(f"   - {file}")
    
    print(f"\n🎯 제출 준비 완료!")
    print(f"   모든 자료가 {submission_dir}/ 디렉토리에 정리되었습니다.")

if __name__ == "__main__":
    main()
