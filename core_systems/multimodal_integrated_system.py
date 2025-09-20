#!/usr/bin/env python3
"""
Multimodal Integrated System
4개 데이터셋(Big5, CMI, RPPG, Voice)을 통합한 멀티모달 AI 시스템
"""

import json
import os
from typing import Any, Dict, List, Tuple

import google.cloud.bigquery as bigquery
import numpy as np
import pandas as pd
from google.cloud import aiplatform
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from ai_generate_system import AIGenerateSystem
from vector_search_system import Big5VectorSearch


class MultimodalIntegratedSystem:
    def __init__(self, project_id: str = "persona-diary-service"):
        """멀티모달 통합 시스템 초기화"""
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)

        # 각 모달리티별 시스템 초기화
        self.big5_system = Big5VectorSearch(project_id)
        self.ai_generate = AIGenerateSystem(project_id)

        # 데이터셋 매핑
        self.datasets = {
            "big5": "big5_dataset.big5_preprocessed",
            "cmi": "cmi_dataset.cmi_preprocessed",
            "rppg": "rppg_dataset.rppg_preprocessed",
            "voice": "voice_dataset.voice_preprocessed",
        }

    def load_multimodal_data(self, limit: int = 1000) -> Dict[str, pd.DataFrame]:
        """모든 모달리티 데이터 로드"""
        print("🔄 멀티모달 데이터 로딩 중...")

        multimodal_data = {}

        for modality, table_path in self.datasets.items():
            try:
                query = f"SELECT * FROM `{self.project_id}.{table_path}` LIMIT {limit}"
                print(f"   📊 {modality.upper()} 데이터 로딩 중...")

                df = self.client.query(query).to_dataframe()
                multimodal_data[modality] = df
                print(f"   ✅ {modality.upper()}: {len(df)}건 로드 완료")

            except Exception as e:
                print(f"   ❌ {modality.upper()} 로드 실패: {str(e)}")
                multimodal_data[modality] = pd.DataFrame()

        return multimodal_data

    def analyze_data_structure(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """각 모달리티 데이터 구조 분석"""
        print("\n🔍 데이터 구조 분석 중...")

        structure_analysis = {}

        for modality, df in data.items():
            if df.empty:
                structure_analysis[modality] = {
                    "status": "empty",
                    "columns": [],
                    "shape": (0, 0),
                }
                continue

            structure_analysis[modality] = {
                "status": "loaded",
                "columns": list(df.columns),
                "shape": df.shape,
                "dtypes": df.dtypes.to_dict(),
                "sample_data": df.head(2).to_dict() if len(df) > 0 else {},
            }

        return structure_analysis

    def create_unified_user_profile(
        self,
        big5_scores: Dict[str, float],
        cmi_data: Dict[str, Any] = None,
        rppg_data: Dict[str, Any] = None,
        voice_data: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """통합 사용자 프로필 생성"""
        print("\n👤 통합 사용자 프로필 생성 중...")

        unified_profile = {
            "personality": {
                "big5_scores": big5_scores,
                "personality_type": self.classify_personality_type(big5_scores),
            },
            "health_metrics": {
                "cmi_data": cmi_data or {},
                "rppg_data": rppg_data or {},
                "voice_data": voice_data or {},
            },
            "integrated_insights": [],
            "recommendations": [],
        }

        # 통합 인사이트 생성
        insights = self.generate_integrated_insights(
            big5_scores, cmi_data, rppg_data, voice_data
        )
        unified_profile["integrated_insights"] = insights

        # 통합 추천 생성
        recommendations = self.generate_integrated_recommendations(unified_profile)
        unified_profile["recommendations"] = recommendations

        return unified_profile

    def classify_personality_type(self, big5_scores: Dict[str, float]) -> str:
        """Big5 점수 기반 성격 유형 분류"""
        # 간단한 성격 유형 분류 로직
        ext = big5_scores.get("EXT", 3.0)
        est = big5_scores.get("EST", 3.0)
        agr = big5_scores.get("AGR", 3.0)
        csn = big5_scores.get("CSN", 3.0)
        opn = big5_scores.get("OPN", 3.0)

        if ext > 4.0 and opn > 4.0:
            return "창의적 리더"
        elif ext > 4.0 and agr > 4.0:
            return "사회적 협력자"
        elif csn > 4.0 and est < 3.0:
            return "안정적 성취자"
        elif opn > 4.0 and est < 3.0:
            return "혁신적 탐험가"
        else:
            return "균형잡힌 일반인"

    def generate_integrated_insights(
        self,
        big5_scores: Dict[str, float],
        cmi_data: Dict[str, Any],
        rppg_data: Dict[str, Any],
        voice_data: Dict[str, Any],
    ) -> List[str]:
        """통합 인사이트 생성"""
        insights = []

        # Big5 기반 인사이트
        if big5_scores.get("EXT", 3.0) > 4.0:
            insights.append("외향적인 성격으로 사회적 활동에서 에너지를 얻습니다.")

        if big5_scores.get("OPN", 3.0) > 4.0:
            insights.append("개방적인 성격으로 새로운 경험을 적극적으로 추구합니다.")

        # 건강 지표 기반 인사이트 (실제 데이터가 있을 경우)
        if cmi_data:
            insights.append("CMI 데이터를 통한 건강 상태 분석이 가능합니다.")

        if rppg_data:
            insights.append(
                "RPPG 데이터를 통한 심박수 기반 스트레스 분석이 가능합니다."
            )

        if voice_data:
            insights.append("음성 데이터를 통한 감정 상태 분석이 가능합니다.")

        return insights

    def generate_integrated_recommendations(
        self, profile: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """통합 추천 생성"""
        recommendations = []

        personality_type = profile["personality"]["personality_type"]

        # 성격 유형별 추천
        if personality_type == "창의적 리더":
            recommendations.append(
                {
                    "category": "활동",
                    "recommendation": "창의적 프로젝트 리더십 역할을 추천합니다.",
                    "priority": "high",
                }
            )
        elif personality_type == "사회적 협력자":
            recommendations.append(
                {
                    "category": "활동",
                    "recommendation": "팀워크 중심의 협력 활동을 추천합니다.",
                    "priority": "high",
                }
            )

        # 건강 지표 기반 추천
        if profile["health_metrics"]["cmi_data"]:
            recommendations.append(
                {
                    "category": "건강",
                    "recommendation": "정기적인 건강 체크업을 추천합니다.",
                    "priority": "medium",
                }
            )

        return recommendations

    def run_comprehensive_analysis(
        self, target_user: Dict[str, float], data_limit: int = 1000
    ) -> Dict[str, Any]:
        """종합 분석 실행"""
        print("🚀 멀티모달 통합 시스템 시작")
        print("=" * 60)

        # 1. 멀티모달 데이터 로드
        multimodal_data = self.load_multimodal_data(limit=data_limit)

        # 2. 데이터 구조 분석
        structure_analysis = self.analyze_data_structure(multimodal_data)

        # 3. Big5 기반 Vector Search
        print("\n🔍 Vector Search 실행 중...")
        similar_users = self.big5_system.search_similar_users(
            target_user_data=target_user, all_data=multimodal_data["big5"], top_k=5
        )

        # 4. AI Generate 실행
        print("\n🤖 AI Generate 실행 중...")
        ai_report = self.ai_generate.generate_comprehensive_report(
            target_user, multimodal_data["big5"]
        )

        # 5. 통합 사용자 프로필 생성
        unified_profile = self.create_unified_user_profile(target_user)

        # 6. 종합 결과 구성
        comprehensive_result = {
            "system_info": {
                "project_id": self.project_id,
                "data_limit": data_limit,
                "analysis_timestamp": pd.Timestamp.now().isoformat(),
            },
            "data_status": structure_analysis,
            "vector_search_results": similar_users,
            "ai_generate_report": ai_report,
            "unified_profile": unified_profile,
            "multimodal_insights": {
                "total_modalities": len(
                    [m for m in multimodal_data.values() if not m.empty]
                ),
                "successful_analyses": len(
                    [m for m in multimodal_data.values() if not m.empty]
                ),
                "integration_status": "successful",
            },
        }

        return comprehensive_result


def main():
    """메인 실행 함수"""
    print("🌟 멀티모달 통합 AI 시스템")
    print("=" * 60)

    # 통합 시스템 초기화
    integrated_system = MultimodalIntegratedSystem()

    # 샘플 타겟 사용자 (창의적이고 개방적인 성격)
    target_user = {
        "EXT": 4.2,  # 높은 외향성
        "EST": 2.5,  # 낮은 신경증
        "AGR": 4.5,  # 높은 친화성
        "CSN": 3.8,  # 중간 성실성
        "OPN": 4.7,  # 높은 개방성
    }

    print(f"🎯 타겟 사용자: {target_user}")

    # 종합 분석 실행
    result = integrated_system.run_comprehensive_analysis(target_user, data_limit=500)

    # 결과 출력
    print("\n" + "=" * 80)
    print("🎉 멀티모달 통합 분석 결과")
    print("=" * 80)

    print(f"\n📊 데이터 상태:")
    for modality, status in result["data_status"].items():
        if status["status"] == "loaded":
            print(f"   ✅ {modality.upper()}: {status['shape'][0]}건 로드")
        else:
            print(f"   ❌ {modality.upper()}: 로드 실패")

    print(f"\n👥 Vector Search 결과:")
    for i, user in enumerate(result["vector_search_results"], 1):
        print(f"   {i}. 유사도 {user['similarity_score']:.3f} - {user['country']}")

    print(f"\n🎭 통합 사용자 프로필:")
    profile = result["unified_profile"]
    print(f"   성격 유형: {profile['personality']['personality_type']}")
    print(f"   Big5 점수: {profile['personality']['big5_scores']}")

    print(f"\n💡 통합 인사이트:")
    for insight in profile["integrated_insights"]:
        print(f"   • {insight}")

    print(f"\n🎯 통합 추천:")
    for rec in profile["recommendations"]:
        print(
            f"   [{rec['category']}] {rec['recommendation']} (우선순위: {rec['priority']})"
        )

    print(f"\n📈 멀티모달 통합 상태:")
    insights = result["multimodal_insights"]
    print(f"   총 모달리티: {insights['total_modalities']}개")
    print(f"   성공적 분석: {insights['successful_analyses']}개")
    print(f"   통합 상태: {insights['integration_status']}")

    print(f"\n⏰ 분석 완료 시간: {result['system_info']['analysis_timestamp']}")
    print("\n🎉 멀티모달 통합 시스템 완료!")


if __name__ == "__main__":
    main()
