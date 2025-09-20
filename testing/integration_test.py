#!/usr/bin/env python3
"""
종합 시스템 통합 테스트
"""

from ai_generate_system import AIGenerateSystem
from multimodal_integrated_system import MultimodalIntegratedSystem
from vector_search_system import Big5VectorSearch


def test_integration():
    print("=== 종합 시스템 통합 테스트 ===")

    print("1. Vector Search + AI Generate 통합 테스트")
    vs = Big5VectorSearch()
    ai = AIGenerateSystem()

    # 샘플 사용자 데이터
    target_user = {
        "EXT1": 4,
        "EXT2": 3,
        "EXT3": 5,
        "EXT4": 4,
        "EXT5": 3,
        "EXT6": 4,
        "EXT7": 5,
        "EXT8": 3,
        "EXT9": 4,
        "EXT10": 5,
        "EST1": 2,
        "EST2": 3,
        "EST3": 2,
        "EST4": 1,
        "EST5": 3,
        "EST6": 2,
        "EST7": 1,
        "EST8": 3,
        "EST9": 2,
        "EST10": 1,
        "AGR1": 4,
        "AGR2": 5,
        "AGR3": 4,
        "AGR4": 5,
        "AGR5": 4,
        "AGR6": 5,
        "AGR7": 4,
        "AGR8": 5,
        "AGR9": 4,
        "AGR10": 5,
        "CSN1": 3,
        "CSN2": 4,
        "CSN3": 3,
        "CSN4": 4,
        "CSN5": 3,
        "CSN6": 4,
        "CSN7": 3,
        "CSN8": 4,
        "CSN9": 3,
        "CSN10": 4,
        "OPN1": 5,
        "OPN2": 4,
        "OPN3": 5,
        "OPN4": 4,
        "OPN5": 5,
        "OPN6": 4,
        "OPN7": 5,
        "OPN8": 4,
        "OPN9": 5,
        "OPN10": 4,
    }

    try:
        # 데이터 로드
        all_data = vs.load_data(limit=1000)
        report = ai.generate_comprehensive_report(target_user, all_data)
        print(f"✅ 통합 보고서 생성 성공: {len(report)}개 섹션")
        print(f'   유사 사용자: {report["similar_users"]["count"]}명')
        print(f'   개인화 조언: {len(report["personalized_recommendations"])}개')
        print(f'   SHAP 인사이트: {len(report["shap_insights"])}개')
    except Exception as e:
        print(f"❌ 통합 테스트 실패: {e}")

    print("\n2. 멀티모달 시스템 테스트")
    mm = MultimodalIntegratedSystem()
    try:
        data = mm.load_multimodal_data(limit=50)
        print(f"✅ 멀티모달 데이터 로드: {len(data)}개 데이터셋")
        for modality, df in data.items():
            print(f"   {modality.upper()}: {len(df)}건")
    except Exception as e:
        print(f"❌ 멀티모달 테스트 실패: {e}")

    print("\n✅ 종합 시스템 통합 테스트 완료!")


if __name__ == "__main__":
    test_integration()
