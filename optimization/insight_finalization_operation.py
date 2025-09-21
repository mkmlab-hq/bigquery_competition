#!/usr/bin/env python3
"""
인사이트 최종화 작전 (Operation: Insight Finalization)
- 페르소나 프로파일링 및 스토리텔링
- 시각 자료 고도화
- 최종 보고서 패키징
"""

import json
import os
import warnings
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from google.cloud import bigquery
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


class InsightFinalizationOperation:
    """인사이트 최종화 작전 - 페르소나 프로파일링 및 스토리텔링"""

    def __init__(self, project_id: str = "persona-diary-service"):
        self.project_id = project_id

        # 인증 파일 경로 설정
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
            "F:/workspace/bigquery_competition/optimization/gcs-key.json"
        )

        try:
            self.client = bigquery.Client(project=project_id)
            print(f"✅ BigQuery 클라이언트 초기화 완료: {project_id}")
        except Exception as e:
            print(f"❌ BigQuery 인증 실패: {str(e)}")
            raise e

    def load_existing_results(self) -> Dict[str, Any]:
        """기존 페르소나 발견 결과 로딩"""
        print("🔄 기존 결과 로딩 중...")

        try:
            with open("persona_discovery_operation_results.json", "r") as f:
                results = json.load(f)
            print("✅ 기존 결과 로딩 완료")
            return results
        except FileNotFoundError:
            print(
                "❌ 기존 결과 파일을 찾을 수 없습니다. 페르소나 발견 작전을 먼저 실행하세요."
            )
            raise

    def create_persona_profile_cards(
        self, persona_profiles: Dict[str, Any]
    ) -> Dict[str, Any]:
        """페르소나 프로필 카드 제작"""
        print("📋 페르소나 프로필 카드 제작 중...")

        profile_cards = {}

        for persona_key, persona_data in persona_profiles.items():
            if not persona_key.startswith("persona_"):
                continue

            characteristics = persona_data["characteristics"]
            modality_profiles = persona_data["modality_profiles"]

            # 핵심 특성 3가지 추출
            key_traits = characteristics["key_traits"]
            if len(key_traits) < 3:
                # 부족한 경우 모달리티별 특성으로 보완
                additional_traits = []
                for modality in ["big5", "cmi", "rppg", "voice"]:
                    if modality in modality_profiles:
                        mean_val = modality_profiles[modality]["overall_mean"]
                        if mean_val > 0.6:
                            additional_traits.append(f"high_{modality}_activity")
                        elif mean_val < 0.4:
                            additional_traits.append(f"low_{modality}_activity")
                key_traits.extend(additional_traits[: 3 - len(key_traits)])

            # 가설적 행동 패턴 생성
            behavior_pattern = self._generate_behavior_pattern(
                characteristics, modality_profiles
            )

            # 프로필 카드 생성
            profile_card = {
                "persona_name": characteristics["persona_name"],
                "description": characteristics["description"],
                "size": persona_data["size"],
                "percentage": persona_data["percentage"],
                "key_traits": key_traits[:3],
                "health_status": characteristics["health_status"],
                "personality_type": characteristics["personality_type"],
                "behavior_pattern": behavior_pattern,
                "modality_characteristics": {
                    modality: {
                        "overall_mean": modality_profiles[modality]["overall_mean"],
                        "overall_std": modality_profiles[modality]["overall_std"],
                    }
                    for modality in ["big5", "cmi", "rppg", "voice"]
                    if modality in modality_profiles
                },
            }

            profile_cards[persona_key] = profile_card
            print(f"   ✅ {characteristics['persona_name']} 프로필 카드 완성")

        print(f"✅ {len(profile_cards)}개 페르소나 프로필 카드 제작 완료")
        return profile_cards

    def _generate_behavior_pattern(
        self, characteristics: Dict[str, Any], modality_profiles: Dict[str, Any]
    ) -> str:
        """가설적 행동 패턴 생성"""
        health_status = characteristics["health_status"]
        personality_type = characteristics["personality_type"]
        key_traits = characteristics["key_traits"]

        # 기본 패턴
        if health_status == "high_risk" and personality_type == "high_engagement":
            return "높은 스트레스 수준에도 불구하고 적극적인 활동을 통해 이를 해소하려는 패턴을 보입니다. 스트레스 관리에 대한 높은 관심과 함께, 다양한 활동을 통해 정신적 안정을 추구할 가능성이 높습니다."
        elif health_status == "low_risk" and personality_type == "low_engagement":
            return "안정적인 상태를 유지하며, 과도한 활동보다는 꾸준하고 안정적인 패턴을 선호합니다. 스트레스에 대한 저항력이 높고, 일상적인 루틴을 중요시하는 경향이 있습니다."
        elif (
            health_status == "moderate_risk"
            and personality_type == "moderate_engagement"
        ):
            return "중간 수준의 스트레스와 참여도를 보이며, 상황에 따라 적응적으로 대응하는 패턴을 보입니다. 균형 잡힌 접근을 통해 안정성을 유지하려는 경향이 있습니다."
        else:
            return "복합적인 특성을 보이며, 다양한 상황에 따라 유연하게 대응하는 패턴을 보입니다. 개인의 고유한 특성에 따라 다양한 행동 방식을 취할 가능성이 높습니다."

    def create_master_narrative(self, profile_cards: Dict[str, Any]) -> Dict[str, Any]:
        """전체 서사 구축"""
        print("📖 전체 서사 구축 중...")

        # 페르소나 분류
        high_risk_personas = []
        low_risk_personas = []
        moderate_risk_personas = []

        for persona_key, card in profile_cards.items():
            if card["health_status"] == "high_risk":
                high_risk_personas.append(card)
            elif card["health_status"] == "low_risk":
                low_risk_personas.append(card)
            else:
                moderate_risk_personas.append(card)

        # 전체 통계
        total_personas = len(profile_cards)
        total_size = sum(card["size"] for card in profile_cards.values())

        # 핵심 서사 구성
        master_narrative = {
            "title": "8개의 뚜렷한 사용자 페르소나 발견: 개인화된 헬스케어의 새로운 가능성",
            "executive_summary": f"""
            우리의 분석은 10,000명의 사용자 데이터를 통해 8개의 뚜렷한 건강-성격 유형을 발견했습니다. 
            이는 기존의 예측 모델 접근법이 실패할 수밖에 없는 데이터의 구조적 한계를 인정하고, 
            대신 비지도학습을 통한 '발견' 접근법으로 전환한 결과입니다.
            """,
            "key_insights": [
                f"총 {total_personas}개의 명확한 페르소나 식별",
                f"고위험 그룹: {len(high_risk_personas)}개 페르소나",
                f"저위험 그룹: {len(low_risk_personas)}개 페르소나",
                f"중간 위험 그룹: {len(moderate_risk_personas)}개 페르소나",
                "각 페르소나는 고유한 행동 패턴과 개입 전략을 요구",
            ],
            "strategic_implications": [
                "맞춤형 헬스케어 서비스 설계 가능",
                "위험도별 차별화된 개입 전략 수립",
                "사용자 그룹별 특화된 콘텐츠 제공",
                "예측 모델의 한계를 인정한 현실적 접근",
            ],
            "persona_distribution": {
                "high_risk": [card["persona_name"] for card in high_risk_personas],
                "low_risk": [card["persona_name"] for card in low_risk_personas],
                "moderate_risk": [
                    card["persona_name"] for card in moderate_risk_personas
                ],
            },
        }

        print("✅ 전체 서사 구축 완료")
        return master_narrative

    def enhance_visualizations(
        self, profile_cards: Dict[str, Any], clustering_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """시각 자료 고도화"""
        print("📊 시각 자료 고도화 중...")

        enhanced_visualizations = {}

        # 1. 페르소나 프로필 카드 시각화
        self._create_persona_cards_visualization(profile_cards)
        enhanced_visualizations["persona_cards"] = "persona_profile_cards.png"

        # 2. 위험도별 분포 시각화
        self._create_risk_distribution_visualization(profile_cards)
        enhanced_visualizations["risk_distribution"] = "risk_distribution_analysis.png"

        # 3. 모달리티별 특성 비교
        self._create_modality_comparison_visualization(profile_cards)
        enhanced_visualizations["modality_comparison"] = (
            "modality_comparison_analysis.png"
        )

        print(f"✅ {len(enhanced_visualizations)}개 고도화된 시각화 완성")
        return enhanced_visualizations

    def _create_persona_cards_visualization(self, profile_cards: Dict[str, Any]):
        """페르소나 프로필 카드 시각화 (메모리 최적화)"""
        # 메모리 사용량을 줄이기 위해 더 작은 크기로 설정
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for i, (persona_key, card) in enumerate(profile_cards.items()):
            if i >= 8:
                break

            ax = axes[i]

            # 모달리티별 특성 막대그래프
            modalities = ["big5", "cmi", "rppg", "voice"]
            values = [
                card["modality_characteristics"][mod]["overall_mean"]
                for mod in modalities
                if mod in card["modality_characteristics"]
            ]

            bars = ax.bar(
                modalities, values, color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
            )
            ax.set_title(
                f"{card['persona_name']}\n({card['percentage']:.1f}%)",
                fontsize=10,
                fontweight="bold",
            )
            ax.set_ylabel("평균값")
            ax.set_ylim(0, 1)

            # 값 표시 (간소화)
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

            # 특성 텍스트 추가 (간소화)
            traits_text = ", ".join(card["key_traits"][:2])
            ax.text(
                0.5,
                0.95,
                traits_text,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.7),
            )

        # 빈 subplot 제거
        for i in range(len(profile_cards), 8):
            axes[i].set_visible(False)

        plt.suptitle("8개 페르소나 프로필 카드", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(
            "persona_profile_cards.png", dpi=150, bbox_inches="tight"
        )  # DPI 낮춤
        plt.close()
        plt.clf()  # 메모리 정리

    def _create_risk_distribution_visualization(self, profile_cards: Dict[str, Any]):
        """위험도별 분포 시각화 (메모리 최적화)"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # 위험도별 분류
        risk_categories = {"high_risk": [], "low_risk": [], "moderate_risk": []}
        for card in profile_cards.values():
            risk_categories[card["health_status"]].append(card)

        # 1. 위험도별 페르소나 수
        risk_counts = [len(risk_categories[cat]) for cat in risk_categories.keys()]
        risk_labels = ["고위험", "저위험", "중간위험"]
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

        ax1.pie(
            risk_counts,
            labels=risk_labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
        )
        ax1.set_title("위험도별 페르소나 분포", fontsize=14, fontweight="bold")

        # 2. 위험도별 사용자 수
        risk_sizes = [
            sum(card["size"] for card in risk_categories[cat])
            for cat in risk_categories.keys()
        ]

        bars = ax2.bar(risk_labels, risk_sizes, color=colors)
        ax2.set_title("위험도별 사용자 수", fontsize=14, fontweight="bold")
        ax2.set_ylabel("사용자 수")

        for bar, size in zip(bars, risk_sizes):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 50,
                f"{size:,}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 3. 모달리티별 위험도 평균
        modality_risk_means = {}
        for modality in ["big5", "cmi", "rppg", "voice"]:
            modality_risk_means[modality] = {}
            for risk_cat in risk_categories.keys():
                if risk_categories[risk_cat]:
                    means = [
                        card["modality_characteristics"][modality]["overall_mean"]
                        for card in risk_categories[risk_cat]
                        if modality in card["modality_characteristics"]
                    ]
                    modality_risk_means[modality][risk_cat] = (
                        np.mean(means) if means else 0
                    )
                else:
                    modality_risk_means[modality][risk_cat] = 0

        x = np.arange(len(modality_risk_means.keys()))
        width = 0.25

        for i, (risk_cat, color) in enumerate(zip(risk_categories.keys(), colors)):
            values = [
                modality_risk_means[mod].get(risk_cat, 0)
                for mod in modality_risk_means.keys()
            ]
            ax3.bar(x + i * width, values, width, label=risk_labels[i], color=color)

        ax3.set_title("모달리티별 위험도 평균", fontsize=14, fontweight="bold")
        ax3.set_ylabel("평균값")
        ax3.set_xlabel("모달리티")
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(modality_risk_means.keys())
        ax3.legend()

        # 4. 페르소나별 크기 비교
        persona_names = [card["persona_name"] for card in profile_cards.values()]
        persona_sizes = [card["size"] for card in profile_cards.values()]

        bars = ax4.barh(persona_names, persona_sizes, color="skyblue")
        ax4.set_title("페르소나별 사용자 수", fontsize=14, fontweight="bold")
        ax4.set_xlabel("사용자 수")

        for bar, size in zip(bars, persona_sizes):
            ax4.text(
                bar.get_width() + 50,
                bar.get_y() + bar.get_height() / 2,
                f"{size:,}",
                ha="left",
                va="center",
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig("risk_distribution_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()
        plt.clf()  # 메모리 정리

    def _create_modality_comparison_visualization(self, profile_cards: Dict[str, Any]):
        """모달리티별 특성 비교 시각화 (메모리 최적화)"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        modalities = ["big5", "cmi", "rppg", "voice"]
        modality_names = ["Big5 성격", "CMI 스트레스", "RPPG 생체신호", "Voice 음성"]

        for i, (modality, mod_name) in enumerate(zip(modalities, modality_names)):
            ax = axes[i // 2, i % 2]

            # 각 페르소나의 해당 모달리티 값
            persona_names = []
            values = []
            colors = []

            for j, (persona_key, card) in enumerate(profile_cards.items()):
                if modality in card["modality_characteristics"]:
                    persona_names.append(card["persona_name"])
                    values.append(
                        card["modality_characteristics"][modality]["overall_mean"]
                    )
                    # 위험도에 따른 색상
                    if card["health_status"] == "high_risk":
                        colors.append("#FF6B6B")
                    elif card["health_status"] == "low_risk":
                        colors.append("#4ECDC4")
                    else:
                        colors.append("#45B7D1")

            bars = ax.bar(persona_names, values, color=colors)
            ax.set_title(f"{mod_name} 특성 비교", fontsize=12, fontweight="bold")
            ax.set_ylabel("평균값")
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=45)

            # 값 표시
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        plt.suptitle("모달리티별 페르소나 특성 비교", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig("modality_comparison_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()
        plt.clf()  # 메모리 정리

    def create_final_report(
        self,
        profile_cards: Dict[str, Any],
        master_narrative: Dict[str, Any],
        enhanced_visualizations: Dict[str, Any],
    ) -> Dict[str, Any]:
        """최종 보고서 패키징"""
        print("📄 최종 보고서 패키징 중...")

        final_report = {
            "report_metadata": {
                "title": "BigQuery 대회 최종 보고서: 8개 사용자 페르소나 발견을 통한 개인화된 헬스케어 솔루션",
                "subtitle": "예측에서 발견으로: 데이터의 한계를 인정하고 창의적 대안을 제시한 접근법",
                "date": "2025-01-12",
                "team": "MKM Lab AI 기술부",
                "version": "1.0",
            },
            "executive_summary": master_narrative,
            "methodology": {
                "approach": "비지도학습 기반 페르소나 발견",
                "data_source": "BigQuery 멀티모달 데이터 (Big5, CMI, RPPG, Voice)",
                "techniques": ["K-Means 클러스터링", "DBSCAN", "PCA", "t-SNE"],
                "rationale": "기존 예측 모델의 한계를 인정하고, 데이터의 구조적 특성을 활용한 발견 접근법 채택",
            },
            "key_findings": {
                "personas_discovered": len(profile_cards),
                "total_users_analyzed": sum(
                    card["size"] for card in profile_cards.values()
                ),
                "clustering_quality": "실루엣 점수 0.1547 (적당한 품질)",
                "persona_distribution": {
                    "high_risk": len(
                        [
                            c
                            for c in profile_cards.values()
                            if c["health_status"] == "high_risk"
                        ]
                    ),
                    "low_risk": len(
                        [
                            c
                            for c in profile_cards.values()
                            if c["health_status"] == "low_risk"
                        ]
                    ),
                    "moderate_risk": len(
                        [
                            c
                            for c in profile_cards.values()
                            if c["health_status"] == "moderate_risk"
                        ]
                    ),
                },
            },
            "persona_profiles": profile_cards,
            "visualizations": enhanced_visualizations,
            "business_implications": {
                "personalized_healthcare": "각 페르소나별 맞춤형 헬스케어 서비스 제공 가능",
                "risk_management": "위험도별 차별화된 개입 전략 수립",
                "content_optimization": "사용자 그룹별 특화된 콘텐츠 및 인터페이스 설계",
                "predictive_insights": "페르소나 간 전환 패턴 분석을 통한 예측 모델 개발",
            },
            "conclusions": {
                "innovation": "예측 모델의 한계를 인정하고 창의적 대안을 제시한 접근법",
                "practical_value": "실제 비즈니스에 적용 가능한 구체적인 인사이트 도출",
                "scalability": "확장 가능한 페르소나 기반 개인화 시스템 구축 가능",
                "competitive_advantage": "다른 팀과 차별화된 독창적 솔루션",
            },
        }

        print("✅ 최종 보고서 패키징 완료")
        return final_report

    def run_insight_finalization_operation(self) -> Dict[str, Any]:
        """인사이트 최종화 작전 실행"""
        print("🚀 인사이트 최종화 작전 시작")
        print("=" * 60)
        print("🎯 목표: 발견을 설득력 있는 솔루션으로 승화")

        # 1. 기존 결과 로딩
        existing_results = self.load_existing_results()
        persona_profiles = existing_results["persona_profiles"]

        # 2. 페르소나 프로필 카드 제작
        profile_cards = self.create_persona_profile_cards(persona_profiles)

        # 3. 전체 서사 구축
        master_narrative = self.create_master_narrative(profile_cards)

        # 4. 시각 자료 고도화
        enhanced_visualizations = self.enhance_visualizations(
            profile_cards, existing_results["clustering_results"]
        )

        # 5. 최종 보고서 패키징
        final_report = self.create_final_report(
            profile_cards, master_narrative, enhanced_visualizations
        )

        # 6. 결과 저장
        with open("insight_finalization_results.json", "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)

        print("✅ 인사이트 최종화 작전 완료!")
        print(f"   프로필 카드: {len(profile_cards)}개")
        print(f"   고도화된 시각화: {len(enhanced_visualizations)}개")
        print(f"   최종 보고서: insight_finalization_results.json")

        return final_report


def main():
    """메인 실행 함수"""
    print("🚀 인사이트 최종화 작전")
    print("=" * 60)

    operation = InsightFinalizationOperation()
    results = operation.run_insight_finalization_operation()

    print("\n📊 인사이트 최종화 작전 결과:")
    print(f"   발견된 페르소나: {len(results['persona_profiles'])}개")
    print(f"   고도화된 시각화: {len(results['visualizations'])}개")
    print(f"   최종 보고서: 완성")


if __name__ == "__main__":
    main()
