#!/usr/bin/env python3
"""
AI Generate System for Personalized Recommendations
Vector Search 결과를 활용한 개인화 조언 생성 시스템
"""

import json
import os
import warnings
from typing import Dict, List, Tuple

import google.cloud.bigquery as bigquery
import numpy as np
import pandas as pd
import shap
from google.cloud import aiplatform
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# SHAP 경고 메시지 무시
warnings.filterwarnings("ignore", category=UserWarning)

from vector_search_system import Big5VectorSearch


class AIGenerateSystem:
    def __init__(self, project_id: str = "persona-diary-service"):
        """AI Generate 시스템 초기화"""
        self.project_id = project_id
        self.vector_search = Big5VectorSearch(project_id)

        # 성격 특성별 조언 템플릿
        self.advice_templates = {
            "EXT": {
                "high": "외향적인 성격을 가진 당신은 사회적 활동을 통해 에너지를 얻습니다. 새로운 사람들과의 만남을 적극적으로 추천합니다.",
                "low": "내향적인 성격을 가진 당신은 혼자만의 시간이 중요합니다. 조용한 환경에서의 활동을 추천합니다.",
            },
            "EST": {
                "high": "감정적으로 민감한 성격을 가진 당신은 스트레스 관리가 중요합니다. 명상이나 요가 같은 활동을 추천합니다.",
                "low": "감정적으로 안정적인 성격을 가진 당신은 리더십 역할에 적합합니다. 팀 프로젝트를 주도해보세요.",
            },
            "AGR": {
                "high": "친화적인 성격을 가진 당신은 협력적인 환경에서 최고의 성과를 낼 수 있습니다. 팀워크 중심의 활동을 추천합니다.",
                "low": "경쟁적인 성격을 가진 당신은 개인적인 성취를 중시합니다. 독립적인 프로젝트에 도전해보세요.",
            },
            "CSN": {
                "high": "성실한 성격을 가진 당신은 계획적이고 체계적인 접근이 필요합니다. 장기적인 목표 설정을 추천합니다.",
                "low": "유연한 성격을 가진 당신은 변화에 잘 적응합니다. 새로운 기회를 적극적으로 탐색해보세요.",
            },
            "OPN": {
                "high": "개방적인 성격을 가진 당신은 새로운 경험을 추구합니다. 창의적인 활동이나 예술을 추천합니다.",
                "low": "전통적인 성격을 가진 당신은 안정적인 환경을 선호합니다. 검증된 방법론을 활용해보세요.",
            },
        }

    def analyze_personality_traits(self, profile: Dict[str, float]) -> Dict[str, str]:
        """성격 특성 분석 및 분류"""
        trait_analysis = {}

        for trait, score in profile.items():
            if score > 0.5:  # 정규화된 점수가 높음
                trait_analysis[trait] = "high"
            else:
                trait_analysis[trait] = "low"

        return trait_analysis

    def generate_personalized_advice(
        self, target_profile: Dict[str, float], similar_users: List[Dict]
    ) -> Dict[str, any]:
        """개인화된 조언 생성"""
        print("🤖 AI Generate 시작...")

        # 타겟 사용자 성격 분석
        target_analysis = self.analyze_personality_traits(target_profile)

        # 유사 사용자들의 공통 패턴 분석
        common_traits = self.analyze_common_traits(similar_users)

        # 개인화된 조언 생성
        personalized_advice = {
            "target_analysis": target_analysis,
            "common_traits": common_traits,
            "recommendations": [],
            "insights": [],
        }

        # 각 성격 특성별 조언 생성
        for trait, level in target_analysis.items():
            advice = self.advice_templates[trait][level]
            personalized_advice["recommendations"].append(
                {"trait": trait, "level": level, "advice": advice}
            )

        # 유사 사용자 기반 인사이트 생성
        insights = self.generate_insights_from_similar_users(similar_users)
        personalized_advice["insights"] = insights

        return personalized_advice

    def analyze_common_traits(self, similar_users: List[Dict]) -> Dict[str, any]:
        """유사 사용자들의 공통 특성 분석"""
        if not similar_users:
            return {}

        # 성격 프로필 수집
        profiles = [user["personality_profile"] for user in similar_users]

        # 각 특성별 평균 계산
        trait_averages = {}
        for trait in ["EXT", "EST", "AGR", "CSN", "OPN"]:
            scores = [profile[trait] for profile in profiles]
            trait_averages[trait] = {
                "average": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
            }

        return trait_averages

    def generate_insights_from_similar_users(
        self, similar_users: List[Dict]
    ) -> List[str]:
        """유사 사용자 기반 인사이트 생성"""
        insights = []

        if not similar_users:
            return insights

        # 국가별 분포 분석
        countries = [user["country"] for user in similar_users]
        country_counts = {}
        for country in countries:
            country_counts[country] = country_counts.get(country, 0) + 1

        # 가장 많은 국가
        most_common_country = max(country_counts, key=country_counts.get)
        insights.append(
            f"유사한 성격을 가진 사용자들은 주로 {most_common_country} 출신입니다."
        )

        # 유사도 분포 분석
        similarities = [user["similarity_score"] for user in similar_users]
        avg_similarity = np.mean(similarities)
        insights.append(f"평균 유사도는 {avg_similarity:.3f}로 매우 높은 수준입니다.")

        # 성격 특성 패턴 분석
        profiles = [user["personality_profile"] for user in similar_users]
        ext_scores = [profile["EXT"] for profile in profiles]
        if np.mean(ext_scores) > 0.5:
            insights.append("유사 사용자들은 대부분 외향적인 성향을 보입니다.")
        else:
            insights.append("유사 사용자들은 대부분 내향적인 성향을 보입니다.")

        return insights

    def generate_shap_insights(
        self, target_profile: Dict[str, float], similar_users: List[Dict]
    ) -> List[Dict]:
        """
        SHAP을 활용하여 추천의 이유를 심층적으로 분석합니다.
        """
        print("🔬 SHAP 기반 인사이트 분석 중...")

        # 가상의 타겟 변수 (예: 사용자 만족도)를 예측하는 모델 시뮬레이션
        df_similar = pd.DataFrame(
            [user["personality_profile"] for user in similar_users]
        )
        if df_similar.empty or len(df_similar) < 2:
            print("SHAP 분석을 위한 유사 사용자 데이터가 부족합니다.")
            return []

        # 가상의 타겟 변수: '추천 결과 만족도'를 Big5 점수의 선형 결합으로 시뮬레이션
        df_similar["satisfaction"] = (
            df_similar["EXT"] * 0.2 + df_similar["OPN"] * 0.3 - df_similar["EST"] * 0.1
        )

        X = df_similar.drop(columns=["satisfaction"])
        y = df_similar["satisfaction"]

        model = LinearRegression().fit(X, y)

        explainer = shap.Explainer(model, X)

        target_df = pd.DataFrame([target_profile])
        target_df = target_df.reindex(columns=X.columns, fill_value=3.0)

        shap_values = explainer(target_df)

        insights = []
        for trait, value in zip(X.columns, shap_values.values[0]):
            insights.append(
                {
                    "trait": trait,
                    "impact": float(value),
                    "description": f"'{trait}' 성향은 추천 결과에 {value:.3f} 만큼의 영향력을 가집니다.",
                }
            )

        # 영향력 순으로 정렬
        insights.sort(key=lambda x: abs(x["impact"]), reverse=True)

        return insights

    def generate_comprehensive_report(
        self, target_user_data: Dict[str, float], all_data: pd.DataFrame = None
    ) -> Dict[str, any]:
        """종합적인 개인화 보고서 생성"""
        print("📊 종합 보고서 생성 중...")

        # all_data가 제공되지 않으면 자동으로 로드
        if all_data is None:
            print("🔄 데이터 자동 로드 중...")
            all_data = self.vector_search.load_data(limit=1000)

        # Vector Search로 유사 사용자 찾기
        similar_users = self.vector_search.search_similar_users(
            target_user_data=target_user_data, all_data=all_data, top_k=10
        )

        # 타겟 사용자 성격 프로필 계산
        target_vector = []
        for trait, columns in self.vector_search.big5_columns.items():
            trait_score = target_user_data.get(trait, 3.0)
            target_vector.extend([trait_score] * 10)

        target_vector = np.array(target_vector)
        all_vectors = self.vector_search.preprocess_data(all_data)
        target_profile = self.vector_search.get_personality_profile(target_vector)

        # 개인화된 조언 생성
        personalized_advice = self.generate_personalized_advice(
            target_profile, similar_users
        )

        # SHAP 기반 심층 분석
        shap_insights = self.generate_shap_insights(target_profile, similar_users)

        # 종합 보고서 구성
        comprehensive_report = {
            "user_profile": {
                "target_scores": target_user_data,
                "analyzed_profile": target_profile,
                "trait_analysis": personalized_advice["target_analysis"],
            },
            "similar_users": {
                "count": len(similar_users),
                "top_5": similar_users[:5],
                "common_traits": personalized_advice["common_traits"],
            },
            "personalized_recommendations": personalized_advice["recommendations"],
            "insights": personalized_advice["insights"],
            "shap_insights": shap_insights,
            "generated_at": pd.Timestamp.now().isoformat(),
        }

        return comprehensive_report


def main():
    """메인 실행 함수"""
    print("🚀 AI Generate System 시작")

    # AI Generate 시스템 초기화
    ai_system = AIGenerateSystem()

    # 데이터 로드
    print("\n📊 데이터 로딩 중...")
    data = ai_system.vector_search.load_data(limit=1000)

    # 샘플 타겟 사용자 (외향적이고 개방적인 성격)
    target_user = {
        "EXT": 4.5,  # 높은 외향성
        "EST": 2.0,  # 낮은 신경증
        "AGR": 4.0,  # 높은 친화성
        "CSN": 3.5,  # 중간 성실성
        "OPN": 4.8,  # 높은 개방성
    }

    print(f"\n🎯 타겟 사용자: {target_user}")

    # 종합 보고서 생성
    print("\n🤖 AI Generate 실행 중...")
    report = ai_system.generate_comprehensive_report(target_user, data)

    # 결과 출력
    print("\n" + "=" * 80)
    print("🎉 AI Generate 종합 보고서")
    print("=" * 80)

    print(f"\n📋 사용자 프로필:")
    print(f"   원본 점수: {report['user_profile']['target_scores']}")
    print(f"   분석된 프로필: {report['user_profile']['analyzed_profile']}")
    print(f"   특성 분석: {report['user_profile']['trait_analysis']}")

    print(f"\n👥 유사 사용자 ({report['similar_users']['count']}명):")
    for i, user in enumerate(report["similar_users"]["top_5"], 1):
        print(f"   {i}. 유사도 {user['similarity_score']:.3f} - {user['country']}")

    print(f"\n💡 개인화된 조언:")
    for rec in report["personalized_recommendations"]:
        print(f"   [{rec['trait']} - {rec['level']}] {rec['advice']}")

    print(f"\n🔍 인사이트:")
    for insight in report["insights"]:
        print(f"   • {insight}")

    print(f"\n🔬 SHAP 기반 심층 분석:")
    for insight in report["shap_insights"]:
        print(f"   • {insight['trait']}의 영향력: {insight['impact']:.3f}")
        print(f"     {insight['description']}")

    print(f"\n⏰ 생성 시간: {report['generated_at']}")
    print("\n🎉 AI Generate 완료!")


if __name__ == "__main__":
    main()
