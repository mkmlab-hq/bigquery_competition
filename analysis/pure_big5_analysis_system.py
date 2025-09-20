#!/usr/bin/env python3
"""
순수 Big Five 이론 기반 분석 시스템
- 사상체질 및 MKM12 이론 제외
- 서양 Big Five 이론 중심
- BigQuery 대회 최적화
"""

import warnings
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


class PureBig5AnalysisSystem:
    def __init__(self, project_id: str = "persona-diary-service"):
        self.project_id = project_id
        self.scaler = StandardScaler()
        self.results_dir = "pure_big5_results"
        import os

        os.makedirs(self.results_dir, exist_ok=True)

    def load_big5_data(self, limit: int = 2000):
        """Big Five 데이터 로드"""
        print("📊 Big Five 데이터 로드 중...")

        from vector_search_system import Big5VectorSearch

        vs = Big5VectorSearch(project_id=self.project_id)
        data = vs.load_data(limit=limit)

        # Big Five 특성 추출
        big5_traits = ["EXT", "EST", "AGR", "CSN", "OPN"]
        big5_data = []
        for trait in big5_traits:
            trait_cols = [col for col in data.columns if col.startswith(trait)]
            if trait_cols:
                trait_mean = data[trait_cols].mean(axis=1)
                big5_data.append(trait_mean)

        big5_df = pd.DataFrame(np.array(big5_data).T, columns=big5_traits)
        big5_df["country"] = data["country"].values

        print(f"✅ Big Five 데이터 로드 완료: {len(big5_df)}명")
        return big5_df

    def analyze_big5_characteristics(self, big5_df):
        """Big Five 특성 분석"""
        print("🔍 Big Five 특성 분석 중...")

        big5_traits = ["EXT", "EST", "AGR", "CSN", "OPN"]
        trait_names = {
            "EXT": "Extraversion (외향성)",
            "EST": "Neuroticism (신경증)",
            "AGR": "Agreeableness (친화성)",
            "CSN": "Conscientiousness (성실성)",
            "OPN": "Openness (개방성)",
        }

        analysis_results = {}

        for trait in big5_traits:
            trait_data = big5_df[trait]

            # 기본 통계
            stats = {
                "mean": trait_data.mean(),
                "std": trait_data.std(),
                "min": trait_data.min(),
                "max": trait_data.max(),
                "median": trait_data.median(),
            }

            # 분포 분석
            high_count = np.sum(trait_data > 4.0)
            low_count = np.sum(trait_data < 2.0)
            medium_count = len(trait_data) - high_count - low_count

            distribution = {
                "high": {
                    "count": high_count,
                    "percentage": high_count / len(trait_data) * 100,
                },
                "medium": {
                    "count": medium_count,
                    "percentage": medium_count / len(trait_data) * 100,
                },
                "low": {
                    "count": low_count,
                    "percentage": low_count / len(trait_data) * 100,
                },
            }

            analysis_results[trait] = {
                "name": trait_names[trait],
                "stats": stats,
                "distribution": distribution,
            }

            print(f"\\n{trait_names[trait]}:")
            print(f"  평균: {stats['mean']:.3f}")
            print(f"  표준편차: {stats['std']:.3f}")
            print(
                f"  높은 점수: {high_count}명 ({distribution['high']['percentage']:.1f}%)"
            )
            print(
                f"  낮은 점수: {low_count}명 ({distribution['low']['percentage']:.1f}%)"
            )

        return analysis_results

    def analyze_correlations(self, big5_df):
        """Big Five 특성 간 상관관계 분석"""
        print("\\n🔗 Big Five 특성 간 상관관계 분석 중...")

        big5_traits = ["EXT", "EST", "AGR", "CSN", "OPN"]
        correlation_matrix = big5_df[big5_traits].corr()

        print("\\n상관관계 매트릭스:")
        print(correlation_matrix.round(3))

        # 강한 상관관계 찾기
        strong_correlations = []
        for i in range(len(big5_traits)):
            for j in range(i + 1, len(big5_traits)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.3:
                    strong_correlations.append(
                        {
                            "trait1": big5_traits[i],
                            "trait2": big5_traits[j],
                            "correlation": corr_val,
                        }
                    )

        print(f"\\n강한 상관관계 (|r| > 0.3): {len(strong_correlations)}개")
        for corr in strong_correlations:
            print(f"  {corr['trait1']}-{corr['trait2']}: {corr['correlation']:.3f}")

        return correlation_matrix, strong_correlations

    def create_personality_clusters(self, big5_df):
        """Big Five 기반 성격 클러스터 생성"""
        print("\\n🎭 Big Five 기반 성격 클러스터 생성 중...")

        big5_traits = ["EXT", "EST", "AGR", "CSN", "OPN"]
        big5_data = big5_df[big5_traits].values

        # 정규화
        big5_scaled = self.scaler.fit_transform(big5_data)

        # K-means 클러스터링 (순수 Big Five 이론 기반 - 5개 클러스터)
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(big5_scaled)

        big5_df["personality_cluster"] = clusters

        print(f"✅ 성격 클러스터 생성 완료: 5개 클러스터 (순수 Big Five 이론)")

        # 클러스터별 특성 분석
        cluster_analysis = {}
        for i in range(5):
            cluster_data = big5_df[big5_df["personality_cluster"] == i]
            cluster_means = cluster_data[big5_traits].mean()

            cluster_analysis[i] = {
                "size": len(cluster_data),
                "percentage": len(cluster_data) / len(big5_df) * 100,
                "means": cluster_means.to_dict(),
            }

            print(
                f"\\n클러스터 {i+1} ({len(cluster_data)}명, {len(cluster_data)/len(big5_df)*100:.1f}%):"
            )
            for trait in big5_traits:
                print(f"  {trait}: {cluster_means[trait]:.3f}")

        return cluster_analysis

    def generate_personality_insights(self, big5_df, cluster_analysis):
        """성격 인사이트 생성"""
        print("\\n💡 성격 인사이트 생성 중...")

        big5_traits = ["EXT", "EST", "AGR", "CSN", "OPN"]
        trait_names = {
            "EXT": "외향성",
            "EST": "신경증",
            "AGR": "친화성",
            "CSN": "성실성",
            "OPN": "개방성",
        }

        insights = []

        for cluster_id, cluster_info in cluster_analysis.items():
            means = cluster_info["means"]

            # 각 특성의 수준 판단
            trait_levels = {}
            for trait in big5_traits:
                if trait == "EST":  # 신경증은 역방향
                    if means[trait] < 2.5:
                        trait_levels[trait] = "낮음 (안정적)"
                    elif means[trait] < 3.5:
                        trait_levels[trait] = "중간"
                    else:
                        trait_levels[trait] = "높음 (불안정)"
                else:
                    if means[trait] > 4.0:
                        trait_levels[trait] = "높음"
                    elif means[trait] > 3.0:
                        trait_levels[trait] = "중간"
                    else:
                        trait_levels[trait] = "낮음"

            # 클러스터 특성 요약
            cluster_summary = f"클러스터 {cluster_id+1}: "
            for trait in big5_traits:
                cluster_summary += f"{trait_names[trait]}({trait_levels[trait]}) "

            insights.append(
                {
                    "cluster_id": cluster_id,
                    "size": cluster_info["size"],
                    "percentage": cluster_info["percentage"],
                    "summary": cluster_summary,
                    "trait_levels": trait_levels,
                }
            )

        print("\\n성격 클러스터 요약:")
        for insight in insights:
            print(f"  {insight['summary']}")

        return insights

    def create_visualizations(self, big5_df, correlation_matrix, cluster_analysis):
        """시각화 생성"""
        print("\\n📊 시각화 생성 중...")

        big5_traits = ["EXT", "EST", "AGR", "CSN", "OPN"]

        # 1. Big Five 특성 분포
        plt.figure(figsize=(15, 10))

        # 서브플롯 1: Big Five 특성 분포
        plt.subplot(2, 3, 1)
        big5_df[big5_traits].boxplot()
        plt.title("Big Five 특성 분포")
        plt.xticks(rotation=45)

        # 서브플롯 2: 상관관계 히트맵
        plt.subplot(2, 3, 2)
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
        plt.title("Big Five 특성 상관관계")

        # 서브플롯 3: 클러스터 분포
        plt.subplot(2, 3, 3)
        cluster_counts = big5_df["personality_cluster"].value_counts().sort_index()
        plt.bar(cluster_counts.index, cluster_counts.values)
        plt.title("성격 클러스터 분포")
        plt.xlabel("클러스터")
        plt.ylabel("인원 수")

        # 서브플롯 4-6: 클러스터별 Big Five 특성 (상위 3개 클러스터)
        cluster_counts = big5_df["personality_cluster"].value_counts()
        top_clusters = cluster_counts.head(3).index

        for i, cluster_id in enumerate(top_clusters):
            plt.subplot(2, 3, 4 + i)
            cluster_data = big5_df[big5_df["personality_cluster"] == cluster_id]
            if len(cluster_data) > 0:
                cluster_means = cluster_data[big5_traits].mean()
                plt.bar(big5_traits, cluster_means.values)
                plt.title(f"클러스터 {cluster_id+1} 특성 ({len(cluster_data)}명)")
                plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(
            f"{self.results_dir}/pure_big5_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print(f"✅ 시각화 완료: {self.results_dir}/pure_big5_analysis.png")

    def run_pure_big5_analysis(self, limit: int = 2000):
        """순수 Big Five 분석 실행"""
        print("🚀 순수 Big Five 이론 기반 분석 시작")
        print("=" * 60)

        # 1. 데이터 로드
        big5_df = self.load_big5_data(limit)

        # 2. Big Five 특성 분석
        trait_analysis = self.analyze_big5_characteristics(big5_df)

        # 3. 상관관계 분석
        correlation_matrix, strong_correlations = self.analyze_correlations(big5_df)

        # 4. 성격 클러스터 생성
        cluster_analysis = self.create_personality_clusters(big5_df)

        # 5. 성격 인사이트 생성
        insights = self.generate_personality_insights(big5_df, cluster_analysis)

        # 6. 시각화 생성
        self.create_visualizations(big5_df, correlation_matrix, cluster_analysis)

        return {
            "big5_df": big5_df,
            "trait_analysis": trait_analysis,
            "correlation_matrix": correlation_matrix,
            "strong_correlations": strong_correlations,
            "cluster_analysis": cluster_analysis,
            "insights": insights,
        }


def main():
    """메인 실행 함수"""
    print("🧠 순수 Big Five 이론 기반 분석 시스템")

    analyzer = PureBig5AnalysisSystem()
    results = analyzer.run_pure_big5_analysis(limit=2000)

    print("\\n🎉 순수 Big Five 분석 완료!")
    print(f"   결과 저장: {analyzer.results_dir}/")


if __name__ == "__main__":
    main()
