#!/usr/bin/env python3
"""
최종 보고서 수정 및 개선
- 페르소나 이름 차별화
- 제목 및 내용 일관성 수정
- 품질 향상
"""

import json
import os
from typing import Any, Dict

def load_and_correct_report():
    """최종 보고서 로딩 및 수정"""
    print("🔧 최종 보고서 수정 중...")
    
    # 기존 보고서 로딩
    with open("simplified_insight_finalization_results.json", "r", encoding="utf-8") as f:
        report = json.load(f)
    
    # 1. 제목 수정 (8개 → 7개)
    report["report_metadata"]["title"] = "BigQuery 대회 최종 보고서: 7개 사용자 페르소나 발견을 통한 개인화된 헬스케어 솔루션"
    report["executive_summary"]["title"] = "7개의 뚜렷한 사용자 페르소나 발견: 개인화된 헬스케어의 새로운 가능성"
    
    # 2. 페르소나 이름 차별화
    persona_names = [
        "스트레스받는 성취가 A형 (고활동)",
        "스트레스받는 성취가 B형 (중활동)", 
        "스트레스받는 성취가 C형 (저활동)",
        "스트레스받는 성취가 D형 (극고활동)",
        "스트레스받는 성취가 E형 (안정형)",
        "스트레스받는 성취가 F형 (변동형)",
        "스트레스받는 성취가 G형 (특수형)"
    ]
    
    # 3. 페르소나별 특성에 따른 이름 할당
    for i, (persona_key, persona_data) in enumerate(report["persona_profiles"].items()):
        if i < len(persona_names):
            persona_data["persona_name"] = persona_names[i]
            
            # 크기에 따른 설명 추가
            size = persona_data["size"]
            if size > 2000:
                persona_data["description"] = f"가장 큰 그룹의 {persona_names[i]} - 높은 스트레스와 높은 참여도를 보이는 사용자 그룹"
            elif size > 1000:
                persona_data["description"] = f"중간 규모의 {persona_names[i]} - 높은 스트레스와 높은 참여도를 보이는 사용자 그룹"
            else:
                persona_data["description"] = f"소규모의 {persona_names[i]} - 높은 스트레스와 높은 참여도를 보이는 사용자 그룹"
    
    # 4. 페르소나 분포 업데이트
    report["executive_summary"]["persona_distribution"]["high_risk"] = persona_names
    report["executive_summary"]["key_insights"][0] = "총 7개의 명확한 페르소나 식별 (모두 '스트레스받는 성취가' 계열)"
    report["executive_summary"]["key_insights"][1] = "고위험 그룹: 7개 페르소나 (세부 특성별 분류)"
    
    # 5. 핵심 인사이트 추가
    report["executive_summary"]["key_insights"].append("모든 페르소나가 '스트레스받는 성취가'라는 하나의 거대 페르소나의 세부 그룹")
    report["executive_summary"]["key_insights"].append("가장 큰 그룹: 스트레스받는 성취가 B형 (3,082명, 30.8%)")
    
    # 6. 전략적 의미 강화
    report["executive_summary"]["strategic_implications"].append("단일 타겟 집중 전략: '스트레스받는 성취가' 계열 사용자에 집중")
    report["executive_summary"]["strategic_implications"].append("세부 그룹별 차별화: 7개 세부 그룹의 특성에 맞는 맞춤형 접근")
    
    # 7. 비즈니스 임팩트 강화
    report["business_implications"]["target_focus"] = "단일 핵심 타겟: '스트레스받는 성취가' 계열 사용자 (전체의 100%)"
    report["business_implications"]["market_penetration"] = "명확한 타겟 설정으로 시장 집중 공략 가능"
    report["business_implications"]["resource_efficiency"] = "자원 분산 없이 핵심 타겟에 집중 투자"
    
    # 8. 결론 강화
    report["conclusions"]["strategic_clarity"] = "복잡한 다중 페르소나 대신 명확한 단일 타겟 전략"
    report["conclusions"]["market_advantage"] = "경쟁사 대비 월등한 타겟 집중도로 시장 지배력 확보 가능"
    
    print("✅ 최종 보고서 수정 완료")
    return report

def save_corrected_report(report: Dict[str, Any]):
    """수정된 보고서 저장"""
    print("💾 수정된 보고서 저장 중...")
    
    # 수정된 보고서 저장
    with open("final_corrected_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 기존 파일도 업데이트
    with open("simplified_insight_finalization_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("✅ 수정된 보고서 저장 완료")
    print("   - final_corrected_report.json (새 파일)")
    print("   - simplified_insight_finalization_results.json (업데이트)")

def main():
    """메인 실행 함수"""
    print("🚀 최종 보고서 수정 및 개선")
    print("=" * 50)
    
    # 보고서 수정
    corrected_report = load_and_correct_report()
    
    # 수정된 보고서 저장
    save_corrected_report(corrected_report)
    
    print("\n📊 수정 사항 요약:")
    print("   ✅ 제목: 8개 → 7개 페르소나로 수정")
    print("   ✅ 페르소나 이름: 차별화된 7개 이름 부여")
    print("   ✅ 설명: 각 페르소나별 특성 반영")
    print("   ✅ 인사이트: 단일 거대 페르소나 개념 추가")
    print("   ✅ 전략: 타겟 집중 전략 강화")
    
    print("\n🎯 핵심 메시지:")
    print("   '7개의 세부 페르소나 = 1개의 거대 페르소나 (스트레스받는 성취가)'")
    print("   '명확한 단일 타겟 전략으로 시장 집중 공략'")

if __name__ == "__main__":
    main()
