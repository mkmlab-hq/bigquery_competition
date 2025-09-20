#!/usr/bin/env python3
"""
고급 Big Five 클러스터링 시스템
- 다양한 클러스터링 알고리즘 비교
- 최적 클러스터 수 자동 결정
- 클러스터 품질 평가
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


class AdvancedBig5Clustering:
    def __init__(self, project_id: str = "persona-diary-service"):
        self.project_id = project_id
        self.scaler = StandardScaler()
        self.results_dir = "advanced_clustering_results"
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

    def find_optimal_clusters(self, data, max_clusters=10):
        """최적 클러스터 수 찾기"""
        print("🔍 최적 클러스터 수 분석 중...")

        cluster_range = range(2, max_clusters + 1)
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        davies_bouldin_scores = []

        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(data)

            inertias.append(kmeans.inertia_)
            silhouette_avg = silhouette_score(data, cluster_labels)
            silhouette_scores.append(silhouette_avg)

            calinski_avg = calinski_harabasz_score(data, cluster_labels)
            calinski_scores.append(calinski_avg)

            davies_avg = davies_bouldin_score(data, cluster_labels)
            davies_bouldin_scores.append(davies_avg)

            print(
                f"클러스터 {k}개: Silhouette={silhouette_avg:.3f}, Calinski={calinski_avg:.1f}, Davies-Bouldin={davies_avg:.3f}"
            )

        # 최적 클러스터 수 결정
        optimal_k_silhouette = cluster_range[np.argmax(silhouette_scores)]
        optimal_k_calinski = cluster_range[np.argmax(calinski_scores)]
        optimal_k_davies = cluster_range[np.argmin(davies_bouldin_scores)]

        print(f"\n📊 최적 클러스터 수 분석 결과:")
        print(f"  Silhouette Score 기준: {optimal_k_silhouette}개")
        print(f"  Calinski-Harabasz Score 기준: {optimal_k_calinski}개")
        print(f"  Davies-Bouldin Score 기준: {optimal_k_davies}개")

        return {
            "cluster_range": list(cluster_range),
            "inertias": inertias,
            "silhouette_scores": silhouette_scores,
            "calinski_scores": calinski_scores,
            "davies_bouldin_scores": davies_bouldin_scores,
            "optimal_k_silhouette": optimal_k_silhouette,
            "optimal_k_calinski": optimal_k_calinski,
            "optimal_k_davies": optimal_k_davies,
        }

    def compare_clustering_algorithms(self, data, n_clusters=5):
        """다양한 클러스터링 알고리즘 비교"""
        print(f"\n🔄 클러스터링 알고리즘 비교 (클러스터 {n_clusters}개)...")

        algorithms = {
            "K-Means": KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
            "Gaussian Mixture": GaussianMixture(
                n_components=n_clusters, random_state=42
            ),
            "Agglomerative": AgglomerativeClustering(n_clusters=n_clusters),
            "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
        }

        results = {}

        for name, algorithm in algorithms.items():
            try:
                if name == "DBSCAN":
                    cluster_labels = algorithm.fit_predict(data)
                    n_clusters_found = len(set(cluster_labels)) - (
                        1 if -1 in cluster_labels else 0
                    )
                    print(f"  {name}: {n_clusters_found}개 클러스터 발견")
                else:
                    cluster_labels = algorithm.fit_predict(data)
                    n_clusters_found = n_clusters

                if n_clusters_found > 1:
                    silhouette_avg = silhouette_score(data, cluster_labels)
                    calinski_avg = calinski_harabasz_score(data, cluster_labels)
                    davies_avg = davies_bouldin_score(data, cluster_labels)

                    results[name] = {
                        "labels": cluster_labels,
                        "n_clusters": n_clusters_found,
                        "silhouette": silhouette_avg,
                        "calinski": calinski_avg,
                        "davies_bouldin": davies_avg,
                    }

                    print(
                        f"    Silhouette: {silhouette_avg:.3f}, Calinski: {calinski_avg:.1f}, Davies-Bouldin: {davies_avg:.3f}"
                    )
                else:
                    print(f"    {name}: 클러스터링 실패 (클러스터 수 부족)")

            except Exception as e:
                print(f"    {name}: 오류 발생 - {e}")

        return results

    def create_hierarchical_clustering(self, data, max_clusters=10):
        """계층적 클러스터링 분석"""
        print("\n🌳 계층적 클러스터링 분석 중...")

        # 연결 행렬 계산
        linkage_matrix = linkage(data, method="ward")

        # 덴드로그램 생성
        plt.figure(figsize=(15, 8))
        dendrogram(linkage_matrix, truncate_mode="level", p=5)
        plt.title("Big Five 특성 계층적 클러스터링 덴드로그램")
        plt.xlabel("샘플 인덱스")
        plt.ylabel("거리")
        plt.tight_layout()
        plt.savefig(
            f"{self.results_dir}/hierarchical_dendrogram.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(f"✅ 덴드로그램 저장: {self.results_dir}/hierarchical_dendrogram.png")

        return linkage_matrix

    def analyze_cluster_characteristics(self, data, labels, algorithm_name):
        """클러스터 특성 분석"""
        print(f"\n📊 {algorithm_name} 클러스터 특성 분석...")

        big5_traits = ["EXT", "EST", "AGR", "CSN", "OPN"]
        unique_labels = np.unique(labels)

        cluster_analysis = {}

        for cluster_id in unique_labels:
            if cluster_id == -1:  # DBSCAN의 노이즈 포인트
                continue

            cluster_mask = labels == cluster_id
            cluster_data = data[cluster_mask]

            cluster_means = np.mean(cluster_data, axis=0)
            cluster_std = np.std(cluster_data, axis=0)

            cluster_analysis[cluster_id] = {
                "size": len(cluster_data),
                "percentage": len(cluster_data) / len(data) * 100,
                "means": dict(zip(big5_traits, cluster_means)),
                "std": dict(zip(big5_traits, cluster_std)),
            }

            print(
                f"\n클러스터 {cluster_id} ({len(cluster_data)}명, {len(cluster_data)/len(data)*100:.1f}%):"
            )
            for trait in big5_traits:
                trait_idx = big5_traits.index(trait)
                print(
                    f"  {trait}: {cluster_means[trait_idx]:.3f} ± {cluster_std[trait_idx]:.3f}"
                )

        return cluster_analysis

    def create_clustering_visualizations(self, data, results, optimal_analysis):
        """클러스터링 시각화 생성"""
        print("\n📊 클러스터링 시각화 생성 중...")

        # 1. 최적 클러스터 수 분석 그래프
        plt.figure(figsize=(20, 12))

        # 서브플롯 1: Elbow Method
        plt.subplot(2, 4, 1)
        plt.plot(optimal_analysis["cluster_range"], optimal_analysis["inertias"], "bo-")
        plt.title("Elbow Method")
        plt.xlabel("클러스터 수")
        plt.ylabel("Inertia")
        plt.grid(True)

        # 서브플롯 2: Silhouette Score
        plt.subplot(2, 4, 2)
        plt.plot(
            optimal_analysis["cluster_range"],
            optimal_analysis["silhouette_scores"],
            "ro-",
        )
        plt.title("Silhouette Score")
        plt.xlabel("클러스터 수")
        plt.ylabel("Silhouette Score")
        plt.grid(True)

        # 서브플롯 3: Calinski-Harabasz Score
        plt.subplot(2, 4, 3)
        plt.plot(
            optimal_analysis["cluster_range"],
            optimal_analysis["calinski_scores"],
            "go-",
        )
        plt.title("Calinski-Harabasz Score")
        plt.xlabel("클러스터 수")
        plt.ylabel("Calinski-Harabasz Score")
        plt.grid(True)

        # 서브플롯 4: Davies-Bouldin Score
        plt.subplot(2, 4, 4)
        plt.plot(
            optimal_analysis["cluster_range"],
            optimal_analysis["davies_bouldin_scores"],
            "mo-",
        )
        plt.title("Davies-Bouldin Score")
        plt.xlabel("클러스터 수")
        plt.ylabel("Davies-Bouldin Score")
        plt.grid(True)

        # 서브플롯 5-8: 알고리즘별 클러스터 품질 비교
        algorithm_names = list(results.keys())
        silhouette_scores = [results[alg]["silhouette"] for alg in algorithm_names]
        calinski_scores = [results[alg]["calinski"] for alg in algorithm_names]
        davies_scores = [results[alg]["davies_bouldin"] for alg in algorithm_names]

        plt.subplot(2, 4, 5)
        plt.bar(algorithm_names, silhouette_scores)
        plt.title("Silhouette Score 비교")
        plt.xticks(rotation=45)
        plt.ylabel("Silhouette Score")

        plt.subplot(2, 4, 6)
        plt.bar(algorithm_names, calinski_scores)
        plt.title("Calinski-Harabasz Score 비교")
        plt.xticks(rotation=45)
        plt.ylabel("Calinski-Harabasz Score")

        plt.subplot(2, 4, 7)
        plt.bar(algorithm_names, davies_scores)
        plt.title("Davies-Bouldin Score 비교")
        plt.xticks(rotation=45)
        plt.ylabel("Davies-Bouldin Score")

        # 서브플롯 8: 최고 성능 알고리즘 선택
        best_algorithm = max(algorithm_names, key=lambda x: results[x]["silhouette"])
        plt.subplot(2, 4, 8)
        plt.text(
            0.5,
            0.5,
            f'최고 성능 알고리즘:\n{best_algorithm}\nSilhouette: {results[best_algorithm]["silhouette"]:.3f}',
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
        )
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(
            f"{self.results_dir}/clustering_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print(
            f"✅ 클러스터링 분석 시각화 완료: {self.results_dir}/clustering_analysis.png"
        )

        return best_algorithm

    def run_advanced_clustering_analysis(self, limit: int = 2000):
        """고급 클러스터링 분석 실행"""
        print("🚀 고급 Big Five 클러스터링 분석 시작")
        print("=" * 60)

        # 1. 데이터 로드
        big5_df = self.load_big5_data(limit)
        big5_traits = ["EXT", "EST", "AGR", "CSN", "OPN"]
        big5_data = big5_df[big5_traits].values

        # 2. 데이터 정규화
        big5_scaled = self.scaler.fit_transform(big5_data)

        # 3. 최적 클러스터 수 찾기
        optimal_analysis = self.find_optimal_clusters(big5_scaled)

        # 4. 다양한 클러스터링 알고리즘 비교
        n_clusters = optimal_analysis["optimal_k_silhouette"]
        clustering_results = self.compare_clustering_algorithms(big5_scaled, n_clusters)

        # 5. 계층적 클러스터링 분석
        linkage_matrix = self.create_hierarchical_clustering(big5_scaled)

        # 6. 시각화 생성
        best_algorithm = self.create_clustering_visualizations(
            big5_scaled, clustering_results, optimal_analysis
        )

        # 7. 최고 성능 알고리즘으로 클러스터 특성 분석
        if best_algorithm in clustering_results:
            best_labels = clustering_results[best_algorithm]["labels"]
            cluster_characteristics = self.analyze_cluster_characteristics(
                big5_scaled, best_labels, best_algorithm
            )

            # 결과를 원본 데이터에 추가
            big5_df["cluster"] = best_labels
            big5_df["algorithm"] = best_algorithm

            return {
                "big5_df": big5_df,
                "optimal_analysis": optimal_analysis,
                "clustering_results": clustering_results,
                "best_algorithm": best_algorithm,
                "cluster_characteristics": cluster_characteristics,
                "linkage_matrix": linkage_matrix,
            }
        else:
            print("❌ 클러스터링 분석 실패")
            return None


def main():
    """메인 실행 함수"""
    print("🧠 고급 Big Five 클러스터링 분석 시스템")

    analyzer = AdvancedBig5Clustering()
    results = analyzer.run_advanced_clustering_analysis(limit=2000)

    if results:
        print("\n🎉 고급 클러스터링 분석 완료!")
        print(f"   최고 성능 알고리즘: {results['best_algorithm']}")
        print(f"   결과 저장: {analyzer.results_dir}/")
    else:
        print("\n❌ 분석 실패")


if __name__ == "__main__":
    main()
