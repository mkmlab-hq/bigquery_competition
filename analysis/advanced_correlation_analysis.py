#!/usr/bin/env python3
"""
고급 Big Five 상관관계 분석 시스템
- Mutual Information 분석
- Spearman Rank Correlation
- 비선형 관계 탐지
- 상관관계 시각화
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


class AdvancedCorrelationAnalysis:
    def __init__(self, project_id: str = "persona-diary-service"):
        self.project_id = project_id
        self.scaler = StandardScaler()
        self.results_dir = "correlation_analysis_results"
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

    def analyze_pearson_correlations(self, data):
        """Pearson 상관관계 분석"""
        print("\n📊 Pearson 상관관계 분석 중...")

        big5_traits = ["EXT", "EST", "AGR", "CSN", "OPN"]
        correlation_matrix = data[big5_traits].corr(method="pearson")

        print("Pearson 상관관계 매트릭스:")
        print(correlation_matrix.round(3))

        # 강한 상관관계 찾기
        strong_correlations = []
        for i in range(len(big5_traits)):
            for j in range(i + 1, len(big5_traits)):
                corr_val = correlation_matrix.iloc[i, j]
                p_val = pearsonr(data[big5_traits[i]], data[big5_traits[j]])[1]

                if abs(corr_val) > 0.1:  # 약한 상관관계도 포함
                    strong_correlations.append(
                        {
                            "trait1": big5_traits[i],
                            "trait2": big5_traits[j],
                            "pearson_correlation": corr_val,
                            "p_value": p_val,
                            "significance": (
                                "significant" if p_val < 0.05 else "not significant"
                            ),
                        }
                    )

        print(f"\n상관관계 분석 결과 (|r| > 0.1): {len(strong_correlations)}개")
        for corr in strong_correlations:
            print(
                f"  {corr['trait1']}-{corr['trait2']}: r={corr['pearson_correlation']:.3f}, p={corr['p_value']:.3f} ({corr['significance']})"
            )

        return correlation_matrix, strong_correlations

    def analyze_spearman_correlations(self, data):
        """Spearman Rank 상관관계 분석"""
        print("\n📊 Spearman Rank 상관관계 분석 중...")

        big5_traits = ["EXT", "EST", "AGR", "CSN", "OPN"]
        spearman_matrix = np.zeros((len(big5_traits), len(big5_traits)))
        p_values = np.zeros((len(big5_traits), len(big5_traits)))

        for i in range(len(big5_traits)):
            for j in range(len(big5_traits)):
                if i == j:
                    spearman_matrix[i, j] = 1.0
                    p_values[i, j] = 0.0
                else:
                    corr, p_val = spearmanr(data[big5_traits[i]], data[big5_traits[j]])
                    spearman_matrix[i, j] = corr
                    p_values[i, j] = p_val

        spearman_df = pd.DataFrame(
            spearman_matrix, index=big5_traits, columns=big5_traits
        )
        p_values_df = pd.DataFrame(p_values, index=big5_traits, columns=big5_traits)

        print("Spearman 상관관계 매트릭스:")
        print(spearman_df.round(3))

        # 비선형 관계 탐지
        nonlinear_relationships = []
        for i in range(len(big5_traits)):
            for j in range(i + 1, len(big5_traits)):
                pearson_corr = data[big5_traits[i]].corr(
                    data[big5_traits[j]], method="pearson"
                )
                spearman_corr = spearman_matrix[i, j]

                # Pearson과 Spearman의 차이가 크면 비선형 관계 가능성
                diff = abs(pearson_corr - spearman_corr)
                if diff > 0.1:
                    nonlinear_relationships.append(
                        {
                            "trait1": big5_traits[i],
                            "trait2": big5_traits[j],
                            "pearson": pearson_corr,
                            "spearman": spearman_corr,
                            "difference": diff,
                        }
                    )

        print(f"\n비선형 관계 탐지 (차이 > 0.1): {len(nonlinear_relationships)}개")
        for rel in nonlinear_relationships:
            print(
                f"  {rel['trait1']}-{rel['trait2']}: Pearson={rel['pearson']:.3f}, Spearman={rel['spearman']:.3f}, 차이={rel['difference']:.3f}"
            )

        return spearman_df, p_values_df, nonlinear_relationships

    def analyze_mutual_information(self, data):
        """Mutual Information 분석"""
        print("\n📊 Mutual Information 분석 중...")

        big5_traits = ["EXT", "EST", "AGR", "CSN", "OPN"]
        mi_matrix = np.zeros((len(big5_traits), len(big5_traits)))

        for i in range(len(big5_traits)):
            for j in range(len(big5_traits)):
                if i == j:
                    mi_matrix[i, j] = 1.0
                else:
                    # Mutual Information 계산
                    mi_score = mutual_info_regression(
                        data[[big5_traits[i]]], data[big5_traits[j]], random_state=42
                    )[0]
                    mi_matrix[i, j] = mi_score

        mi_df = pd.DataFrame(mi_matrix, index=big5_traits, columns=big5_traits)

        print("Mutual Information 매트릭스:")
        print(mi_df.round(3))

        # 높은 MI 값 찾기
        high_mi_relationships = []
        for i in range(len(big5_traits)):
            for j in range(i + 1, len(big5_traits)):
                mi_score = mi_matrix[i, j]
                if mi_score > 0.1:  # 임계값 설정
                    high_mi_relationships.append(
                        {
                            "trait1": big5_traits[i],
                            "trait2": big5_traits[j],
                            "mi_score": mi_score,
                        }
                    )

        print(f"\n높은 Mutual Information (MI > 0.1): {len(high_mi_relationships)}개")
        for rel in high_mi_relationships:
            print(f"  {rel['trait1']}-{rel['trait2']}: MI={rel['mi_score']:.3f}")

        return mi_df, high_mi_relationships

    def analyze_categorical_correlations(self, data):
        """범주형 변수와의 상관관계 분석"""
        print("\n📊 범주형 변수 상관관계 분석 중...")

        big5_traits = ["EXT", "EST", "AGR", "CSN", "OPN"]

        # 국가별 Big Five 특성 분석
        country_analysis = {}
        countries = data["country"].value_counts().head(10).index

        for country in countries:
            country_data = data[data["country"] == country]
            if len(country_data) > 10:  # 최소 10명 이상
                country_means = country_data[big5_traits].mean()
                country_analysis[country] = {
                    "count": len(country_data),
                    "means": country_means.to_dict(),
                }

        print("상위 10개 국가별 Big Five 특성 평균:")
        for country, info in country_analysis.items():
            print(f"\n{country} ({info['count']}명):")
            for trait in big5_traits:
                print(f"  {trait}: {info['means'][trait]:.3f}")

        return country_analysis

    def create_correlation_visualizations(
        self, pearson_matrix, spearman_matrix, mi_matrix, country_analysis
    ):
        """상관관계 시각화 생성"""
        print("\n📊 상관관계 시각화 생성 중...")

        big5_traits = ["EXT", "EST", "AGR", "CSN", "OPN"]

        plt.figure(figsize=(20, 15))

        # 서브플롯 1: Pearson 상관관계 히트맵
        plt.subplot(2, 4, 1)
        sns.heatmap(
            pearson_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Pearson 상관관계")

        # 서브플롯 2: Spearman 상관관계 히트맵
        plt.subplot(2, 4, 2)
        sns.heatmap(
            spearman_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Spearman 상관관계")

        # 서브플롯 3: Mutual Information 히트맵
        plt.subplot(2, 4, 3)
        sns.heatmap(
            mi_matrix, annot=True, cmap="viridis", square=True, cbar_kws={"shrink": 0.8}
        )
        plt.title("Mutual Information")

        # 서브플롯 4: Pearson vs Spearman 비교
        plt.subplot(2, 4, 4)
        pearson_values = []
        spearman_values = []
        trait_pairs = []

        for i in range(len(big5_traits)):
            for j in range(i + 1, len(big5_traits)):
                pearson_values.append(pearson_matrix.iloc[i, j])
                spearman_values.append(spearman_matrix.iloc[i, j])
                trait_pairs.append(f"{big5_traits[i]}-{big5_traits[j]}")

        plt.scatter(pearson_values, spearman_values, alpha=0.7)
        plt.plot([-1, 1], [-1, 1], "r--", alpha=0.5)
        plt.xlabel("Pearson 상관계수")
        plt.ylabel("Spearman 상관계수")
        plt.title("Pearson vs Spearman")
        plt.grid(True, alpha=0.3)

        # 서브플롯 5-8: 국가별 Big Five 특성 비교
        countries = list(country_analysis.keys())[:4]
        for i, country in enumerate(countries):
            plt.subplot(2, 4, 5 + i)
            country_means = list(country_analysis[country]["means"].values())
            plt.bar(big5_traits, country_means)
            plt.title(f'{country} ({country_analysis[country]["count"]}명)')
            plt.xticks(rotation=45)
            plt.ylabel("평균 점수")

        plt.tight_layout()
        plt.savefig(
            f"{self.results_dir}/correlation_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print(f"✅ 상관관계 시각화 완료: {self.results_dir}/correlation_analysis.png")

    def run_advanced_correlation_analysis(self, limit: int = 2000):
        """고급 상관관계 분석 실행"""
        print("🚀 고급 Big Five 상관관계 분석 시작")
        print("=" * 60)

        # 1. 데이터 로드
        big5_df = self.load_big5_data(limit)

        # 2. Pearson 상관관계 분석
        pearson_matrix, pearson_correlations = self.analyze_pearson_correlations(
            big5_df
        )

        # 3. Spearman 상관관계 분석
        spearman_matrix, p_values, nonlinear_relationships = (
            self.analyze_spearman_correlations(big5_df)
        )

        # 4. Mutual Information 분석
        mi_matrix, high_mi_relationships = self.analyze_mutual_information(big5_df)

        # 5. 범주형 변수 상관관계 분석
        country_analysis = self.analyze_categorical_correlations(big5_df)

        # 6. 시각화 생성
        self.create_correlation_visualizations(
            pearson_matrix, spearman_matrix, mi_matrix, country_analysis
        )

        return {
            "big5_df": big5_df,
            "pearson_matrix": pearson_matrix,
            "pearson_correlations": pearson_correlations,
            "spearman_matrix": spearman_matrix,
            "p_values": p_values,
            "nonlinear_relationships": nonlinear_relationships,
            "mi_matrix": mi_matrix,
            "high_mi_relationships": high_mi_relationships,
            "country_analysis": country_analysis,
        }


def main():
    """메인 실행 함수"""
    print("🧠 고급 Big Five 상관관계 분석 시스템")

    analyzer = AdvancedCorrelationAnalysis()
    results = analyzer.run_advanced_correlation_analysis(limit=2000)

    print("\n🎉 고급 상관관계 분석 완료!")
    print(f"   결과 저장: {analyzer.results_dir}/")


if __name__ == "__main__":
    main()
