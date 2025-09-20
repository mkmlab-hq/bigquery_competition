#!/usr/bin/env python3
"""
고급 Big Five 시각화 시스템
- Radar Chart (클러스터별 특성)
- Heatmap (국가별 특성 차이)
- SHAP 기반 모델 해석성 시각화
- Interactive Dashboard
"""

import warnings
from math import pi

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
import seaborn as sns
import shap
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


class AdvancedBig5Visualization:
    def __init__(self, project_id: str = "persona-diary-service"):
        self.project_id = project_id
        self.scaler = StandardScaler()
        self.results_dir = "advanced_visualization_results"
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

    def create_radar_chart(self, big5_df, cluster_labels=None):
        """클러스터별 Radar Chart 생성"""
        print("📊 Radar Chart 생성 중...")

        big5_traits = ["EXT", "EST", "AGR", "CSN", "OPN"]
        trait_names = ["외향성", "신경증", "친화성", "성실성", "개방성"]

        if cluster_labels is None:
            # K-means 클러스터링 수행
            from sklearn.cluster import KMeans

            big5_data = big5_df[big5_traits].values
            big5_scaled = self.scaler.fit_transform(big5_data)
            kmeans = KMeans(n_clusters=6, random_state=42)
            cluster_labels = kmeans.fit_predict(big5_scaled)

        # 클러스터별 평균 계산
        big5_df["cluster"] = cluster_labels
        cluster_means = big5_df.groupby("cluster")[big5_traits].mean()

        # Radar Chart 생성
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection="polar"))

        # 각도 설정
        angles = [n / float(len(big5_traits)) * 2 * pi for n in range(len(big5_traits))]
        angles += angles[:1]  # 닫힌 다각형을 위해

        colors = ["red", "blue", "green", "orange", "purple", "brown"]

        for i, (cluster_id, means) in enumerate(cluster_means.iterrows()):
            values = means.values.tolist()
            values += values[:1]  # 닫힌 다각형을 위해

            ax.plot(
                angles,
                values,
                "o-",
                linewidth=2,
                label=f"클러스터 {cluster_id+1}",
                color=colors[i],
            )
            ax.fill(angles, values, alpha=0.25, color=colors[i])

        # 축 레이블 설정
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(trait_names)
        ax.set_ylim(0, 6)
        ax.set_title("Big Five 클러스터별 특성 Radar Chart", size=16, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/radar_chart.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✅ Radar Chart 저장: {self.results_dir}/radar_chart.png")
        return cluster_means

    def create_country_heatmap(self, big5_df):
        """국가별 Big Five 특성 Heatmap 생성"""
        print("📊 국가별 Heatmap 생성 중...")

        big5_traits = ["EXT", "EST", "AGR", "CSN", "OPN"]
        trait_names = ["외향성", "신경증", "친화성", "성실성", "개방성"]

        # 상위 10개 국가 선택
        top_countries = big5_df["country"].value_counts().head(10).index
        country_data = big5_df[big5_df["country"].isin(top_countries)]

        # 국가별 평균 계산
        country_means = country_data.groupby("country")[big5_traits].mean()
        country_means.columns = trait_names

        # Heatmap 생성
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            country_means.T,
            annot=True,
            cmap="RdYlBu_r",
            center=3.5,
            square=True,
            cbar_kws={"shrink": 0.8},
            fmt=".2f",
        )
        plt.title("국가별 Big Five 특성 Heatmap", fontsize=16, pad=20)
        plt.xlabel("국가")
        plt.ylabel("Big Five 특성")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig(
            f"{self.results_dir}/country_heatmap.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print(f"✅ 국가별 Heatmap 저장: {self.results_dir}/country_heatmap.png")
        return country_means

    def create_shap_visualization(self, big5_df):
        """SHAP 기반 모델 해석성 시각화"""
        print("📊 SHAP 시각화 생성 중...")

        big5_traits = ["EXT", "EST", "AGR", "CSN", "OPN"]

        # 타겟 변수 생성 (만족도 시뮬레이션)
        big5_df["satisfaction"] = (
            big5_df["EXT"] * 0.2
            + big5_df["OPN"] * 0.3
            - big5_df["EST"] * 0.1
            + big5_df["AGR"] * 0.25
            + big5_df["CSN"] * 0.15
        )

        # 모델 훈련
        X = big5_df[big5_traits]
        y = big5_df["satisfaction"]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # SHAP 분석
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # SHAP 시각화 생성
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 특성 중요도 (SHAP 평균 절댓값)
        feature_importance = np.abs(shap_values).mean(0)
        axes[0, 0].bar(big5_traits, feature_importance)
        axes[0, 0].set_title("SHAP 특성 중요도")
        axes[0, 0].set_ylabel("평균 |SHAP 값|")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # 2. 특성별 SHAP 값 분포
        for i, trait in enumerate(big5_traits):
            axes[0, 1].scatter(X[trait], shap_values[:, i], alpha=0.6, label=trait)
        axes[0, 1].set_xlabel("특성 값")
        axes[0, 1].set_ylabel("SHAP 값")
        axes[0, 1].set_title("특성별 SHAP 값 분포")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. SHAP 값 히트맵 (샘플별)
        sample_indices = np.random.choice(
            len(shap_values), min(100, len(shap_values)), replace=False
        )
        shap_sample = shap_values[sample_indices]
        im = axes[1, 0].imshow(shap_sample.T, aspect="auto", cmap="RdBu_r")
        axes[1, 0].set_title("SHAP 값 히트맵 (샘플별)")
        axes[1, 0].set_xlabel("샘플")
        axes[1, 0].set_ylabel("특성")
        axes[1, 0].set_yticks(range(len(big5_traits)))
        axes[1, 0].set_yticklabels(big5_traits)
        plt.colorbar(im, ax=axes[1, 0])

        # 4. 모델 성능
        from sklearn.metrics import mean_squared_error, r2_score

        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        axes[1, 1].text(
            0.5, 0.7, f"R² Score: {r2:.3f}", ha="center", va="center", fontsize=14
        )
        axes[1, 1].text(
            0.5, 0.5, f"RMSE: {rmse:.3f}", ha="center", va="center", fontsize=14
        )
        axes[1, 1].text(
            0.5,
            0.3,
            f"특성 수: {len(big5_traits)}",
            ha="center",
            va="center",
            fontsize=14,
        )
        axes[1, 1].set_title("모델 성능 지표")
        axes[1, 1].axis("off")

        plt.tight_layout()
        plt.savefig(
            f"{self.results_dir}/shap_visualization.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print(f"✅ SHAP 시각화 저장: {self.results_dir}/shap_visualization.png")

        # SHAP 값 요약
        feature_importance = np.abs(shap_values).mean(0)
        feature_importance_df = pd.DataFrame(
            {"trait": big5_traits, "importance": feature_importance}
        ).sort_values("importance", ascending=False)

        print("\n📊 특성 중요도 (SHAP 기준):")
        for _, row in feature_importance_df.iterrows():
            print(f"  {row['trait']}: {row['importance']:.3f}")

        return feature_importance_df

    def create_interactive_dashboard(self, big5_df, cluster_means, country_means):
        """Interactive Dashboard 생성 (Plotly)"""
        print("📊 Interactive Dashboard 생성 중...")

        big5_traits = ["EXT", "EST", "AGR", "CSN", "OPN"]
        trait_names = ["외향성", "신경증", "친화성", "성실성", "개방성"]

        # 1. 클러스터별 Radar Chart (Interactive)
        fig_radar = go.Figure()

        for i, (cluster_id, means) in enumerate(cluster_means.iterrows()):
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=means.values.tolist() + [means.values[0]],  # 닫힌 다각형
                    theta=trait_names + [trait_names[0]],
                    fill="toself",
                    name=f"클러스터 {cluster_id+1}",
                    line_color=px.colors.qualitative.Set1[i],
                )
            )

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 6])),
            showlegend=True,
            title="Big Five 클러스터별 특성 (Interactive)",
        )

        # 2. 국가별 Heatmap (Interactive)
        fig_heatmap = px.imshow(
            country_means.T,
            labels=dict(x="국가", y="특성", color="점수"),
            x=country_means.index,
            y=country_means.columns,
            color_continuous_scale="RdYlBu_r",
            aspect="auto",
        )
        fig_heatmap.update_layout(title="국가별 Big Five 특성 Heatmap (Interactive)")

        # 3. 클러스터 분포
        cluster_counts = big5_df["cluster"].value_counts().sort_index()
        fig_cluster = px.bar(
            x=[f"클러스터 {i+1}" for i in cluster_counts.index],
            y=cluster_counts.values,
            title="클러스터 분포",
            labels={"x": "클러스터", "y": "인원 수"},
        )

        # 4. 특성별 분포
        fig_dist = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=trait_names,
            specs=[
                [{"type": "histogram"} for _ in range(3)],
                [{"type": "histogram"} for _ in range(3)],
            ],
        )

        for i, trait in enumerate(big5_traits):
            row = i // 3 + 1
            col = i % 3 + 1
            fig_dist.add_trace(
                go.Histogram(x=big5_df[trait], name=trait, showlegend=False),
                row=row,
                col=col,
            )

        fig_dist.update_layout(title="Big Five 특성별 분포")

        # Dashboard HTML 생성
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Big Five 분석 대시보드</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>🧠 Big Five 성격 분석 대시보드</h1>
            
            <div id="radar-chart"></div>
            <div id="heatmap"></div>
            <div id="cluster-dist"></div>
            <div id="trait-dist"></div>
            
            <script>
                {fig_radar.to_html(include_plotlyjs=False, div_id="radar-chart")}
                {fig_heatmap.to_html(include_plotlyjs=False, div_id="heatmap")}
                {fig_cluster.to_html(include_plotlyjs=False, div_id="cluster-dist")}
                {fig_dist.to_html(include_plotlyjs=False, div_id="trait-dist")}
            </script>
        </body>
        </html>
        """

        with open(
            f"{self.results_dir}/interactive_dashboard.html", "w", encoding="utf-8"
        ) as f:
            f.write(dashboard_html)

        print(
            f"✅ Interactive Dashboard 저장: {self.results_dir}/interactive_dashboard.html"
        )

    def create_comprehensive_visualization(self, limit: int = 2000):
        """종합 시각화 생성"""
        print("🚀 종합 시각화 생성 시작")
        print("=" * 60)

        # 데이터 로드
        big5_df = self.load_big5_data(limit)

        # 1. 클러스터링 수행
        from sklearn.cluster import KMeans

        big5_traits = ["EXT", "EST", "AGR", "CSN", "OPN"]
        big5_data = big5_df[big5_traits].values
        big5_scaled = self.scaler.fit_transform(big5_data)

        kmeans = KMeans(n_clusters=6, random_state=42)
        cluster_labels = kmeans.fit_predict(big5_scaled)

        # 2. Radar Chart 생성
        cluster_means = self.create_radar_chart(big5_df, cluster_labels)

        # 3. 국가별 Heatmap 생성
        country_means = self.create_country_heatmap(big5_df)

        # 4. SHAP 시각화 생성
        feature_importance = self.create_shap_visualization(big5_df)

        # 5. Interactive Dashboard 생성
        self.create_interactive_dashboard(big5_df, cluster_means, country_means)

        print("\n🎉 종합 시각화 완료!")
        print(f"   결과 저장: {self.results_dir}/")

        return {
            "cluster_means": cluster_means,
            "country_means": country_means,
            "feature_importance": feature_importance,
        }


def main():
    """메인 실행 함수"""
    print("🧠 고급 Big Five 시각화 시스템")

    visualizer = AdvancedBig5Visualization()
    results = visualizer.create_comprehensive_visualization(limit=2000)

    print("\n🎉 고급 시각화 완료!")
    print(f"   결과 저장: {visualizer.results_dir}/")


if __name__ == "__main__":
    main()
