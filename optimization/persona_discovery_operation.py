#!/usr/bin/env python3
"""
페르소나 발견 작전 (Persona Discovery Operation)
- 예측 모델 → 비지도학습 전환
- 회귀 → 클러스터링 전환
- 성능 → 인사이트 전환
- BigQuery 대회 규정 완전 준수
"""

import json
import os
import warnings
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from google.cloud import bigquery
from plotly.subplots import make_subplots
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


class PersonaDiscoveryOperation:
    """페르소나 발견 작전 - 클러스터링 기반 사용자 그룹 분석"""

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

    def load_multimodal_data(self, limit: int = 10000) -> Dict[str, np.ndarray]:
        """멀티모달 데이터 로딩"""
        print("🔄 멀티모달 데이터 로딩 중...")

        try:
            # Big5 데이터 로딩
            big5_query = f"""
            SELECT * FROM `persona-diary-service.big5_dataset.big5_preprocessed` LIMIT {limit}
            """
            big5_df = self.client.query(big5_query).to_dataframe()
            big5_numeric = big5_df.select_dtypes(include=[np.number])

            # CMI 데이터 로딩
            cmi_query = f"""
            SELECT * FROM `persona-diary-service.cmi_dataset.cmi_preprocessed` LIMIT {limit}
            """
            cmi_df = self.client.query(cmi_query).to_dataframe()
            cmi_numeric = cmi_df.select_dtypes(include=[np.number])

            # RPPG 데이터 로딩
            rppg_query = f"""
            SELECT * FROM `persona-diary-service.rppg_dataset.rppg_preprocessed` LIMIT {limit}
            """
            rppg_df = self.client.query(rppg_query).to_dataframe()
            rppg_numeric = rppg_df.select_dtypes(include=[np.number])

            # Voice 데이터 로딩
            voice_query = f"""
            SELECT * FROM `persona-diary-service.voice_dataset.voice_preprocessed` LIMIT {limit}
            """
            voice_df = self.client.query(voice_query).to_dataframe()
            voice_numeric = voice_df.select_dtypes(include=[np.number])

            multimodal_data = {
                "big5": big5_numeric.values.astype(np.float64),
                "cmi": cmi_numeric.values.astype(np.float64),
                "rppg": rppg_numeric.values.astype(np.float64),
                "voice": voice_numeric.values.astype(np.float64),
            }

            print(f"✅ 멀티모달 데이터 로딩 완료:")
            print(f"   Big5: {big5_numeric.shape}")
            print(f"   CMI: {cmi_numeric.shape}")
            print(f"   RPPG: {rppg_numeric.shape}")
            print(f"   Voice: {voice_numeric.shape}")

            return multimodal_data

        except Exception as e:
            print(f"❌ BigQuery 데이터 로딩 실패: {str(e)}")
            raise e

    def create_unified_latent_space(
        self, multimodal_data: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """통합 잠재 공간 생성"""
        print("🔧 통합 잠재 공간 생성 중...")

        # 각 모달리티별 통계 특성 추출
        features = []
        modality_info = []

        for modality_name, data in multimodal_data.items():
            print(f"   특성 추출 중: {modality_name}")

            # 기본 통계 특성
            mean_features = np.mean(data, axis=1, keepdims=True)
            std_features = np.std(data, axis=1, keepdims=True)
            max_features = np.max(data, axis=1, keepdims=True)
            min_features = np.min(data, axis=1, keepdims=True)
            median_features = np.median(data, axis=1, keepdims=True)

            # 고급 통계 특성
            q25_features = np.percentile(data, 25, axis=1, keepdims=True)
            q75_features = np.percentile(data, 75, axis=1, keepdims=True)
            range_features = max_features - min_features
            iqr_features = q75_features - q25_features

            # 모달리티별 특성 결합
            modality_features = np.concatenate(
                [
                    mean_features,
                    std_features,
                    max_features,
                    min_features,
                    median_features,
                    q25_features,
                    q75_features,
                    range_features,
                    iqr_features,
                ],
                axis=1,
            )

            features.append(modality_features)
            modality_info.extend([modality_name] * modality_features.shape[1])

            print(f"     {modality_name} 특성 수: {modality_features.shape[1]}")

        # 모든 모달리티 특성 결합
        X_combined = np.concatenate(features, axis=1)

        # 데이터 정규화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combined)

        # PCA를 통한 차원 축소
        pca = PCA(n_components=min(50, X_scaled.shape[1]))
        X_pca = pca.fit_transform(X_scaled)

        # t-SNE를 통한 비선형 차원 축소
        if X_scaled.shape[0] > 1000:
            # 큰 데이터셋의 경우 샘플링
            sample_indices = np.random.choice(X_scaled.shape[0], 1000, replace=False)
            X_sample = X_scaled[sample_indices]
        else:
            X_sample = X_scaled
            sample_indices = np.arange(X_scaled.shape[0])

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_sample)

        latent_space_info = {
            "original_features": X_combined.shape[1],
            "pca_components": X_pca.shape[1],
            "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "pca_cumulative_variance": np.cumsum(
                pca.explained_variance_ratio_
            ).tolist(),
            "tsne_components": X_tsne.shape[0],
            "sample_indices": sample_indices.tolist(),
            "modality_info": modality_info,
            "scaler": scaler,
            "pca": pca,
            "tsne": tsne,
        }

        print(f"✅ 통합 잠재 공간 생성 완료:")
        print(f"   원본 특성 수: {X_combined.shape[1]}")
        print(f"   PCA 차원: {X_pca.shape[1]}")
        print(f"   PCA 설명 분산: {pca.explained_variance_ratio_[:5]}")
        print(f"   t-SNE 샘플 수: {X_tsne.shape[0]}")

        return X_pca, latent_space_info

    def discover_personas(
        self, X_latent: np.ndarray, latent_space_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """페르소나 발견 (클러스터링)"""
        print("🎯 페르소나 발견 중...")

        # 최적 클러스터 수 찾기 (K-Means)
        k_range = range(2, 11)
        silhouette_scores = []
        inertias = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_latent)
            silhouette_avg = silhouette_score(X_latent, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            inertias.append(kmeans.inertia_)

        # 최적 클러스터 수 선택
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(
            f"   최적 클러스터 수: {optimal_k} (실루엣 점수: {max(silhouette_scores):.4f})"
        )

        # K-Means 클러스터링
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_latent)

        # DBSCAN 클러스터링 (비선형 클러스터 탐지)
        # 최적 eps 찾기
        neighbors = NearestNeighbors(n_neighbors=min(20, X_latent.shape[0] // 10))
        neighbors_fit = neighbors.fit(X_latent)
        distances, indices = neighbors_fit.kneighbors(X_latent)
        distances = np.sort(distances[:, -1], axis=0)

        # 엘보우 방법으로 최적 eps 찾기
        eps_candidates = np.linspace(distances[0], distances[-1], 20)
        dbscan_scores = []

        for eps in eps_candidates:
            dbscan = DBSCAN(eps=eps, min_samples=5)
            dbscan_labels = dbscan.fit_predict(X_latent)
            n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            if n_clusters > 1:
                silhouette_avg = silhouette_score(X_latent, dbscan_labels)
                dbscan_scores.append((eps, silhouette_avg, n_clusters))
            else:
                dbscan_scores.append((eps, 0, n_clusters))

        # 최적 DBSCAN 파라미터
        if dbscan_scores:
            best_eps, best_score, best_n_clusters = max(
                dbscan_scores, key=lambda x: x[1]
            )
            dbscan = DBSCAN(eps=best_eps, min_samples=5)
            dbscan_labels = dbscan.fit_predict(X_latent)
            print(
                f"   DBSCAN 최적 eps: {best_eps:.4f} (클러스터 수: {best_n_clusters}, 실루엣 점수: {best_score:.4f})"
            )
        else:
            dbscan_labels = np.zeros(X_latent.shape[0])
            print("   DBSCAN 클러스터링 실패 - 모든 점을 노이즈로 분류")

        # 클러스터링 결과 통합
        clustering_results = {
            "kmeans": {
                "labels": kmeans_labels.tolist(),
                "n_clusters": optimal_k,
                "silhouette_score": max(silhouette_scores),
                "inertia": kmeans.inertia_,
                "centers": kmeans.cluster_centers_.tolist(),
            },
            "dbscan": {
                "labels": dbscan_labels.tolist(),
                "n_clusters": len(set(dbscan_labels))
                - (1 if -1 in dbscan_labels else 0),
                "silhouette_score": best_score if dbscan_scores else 0,
                "eps": best_eps if dbscan_scores else 0,
                "noise_points": int(np.sum(dbscan_labels == -1)),
            },
            "optimal_k": optimal_k,
            "silhouette_scores": silhouette_scores,
            "inertias": inertias,
            "k_range": list(k_range),
        }

        print(f"✅ 페르소나 발견 완료:")
        print(f"   K-Means 클러스터 수: {optimal_k}")
        print(
            f"   DBSCAN 클러스터 수: {len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)}"
        )
        print(f"   노이즈 점 수: {int(np.sum(dbscan_labels == -1))}")

        return clustering_results

    def profile_personas(
        self, multimodal_data: Dict[str, np.ndarray], clustering_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """페르소나 프로파일링"""
        print("👥 페르소나 프로파일링 중...")

        kmeans_labels = np.array(clustering_results["kmeans"]["labels"])
        dbscan_labels = np.array(clustering_results["dbscan"]["labels"])

        persona_profiles = {}

        # K-Means 기반 페르소나 프로파일링
        for cluster_id in range(clustering_results["kmeans"]["n_clusters"]):
            cluster_mask = kmeans_labels == cluster_id
            cluster_size = np.sum(cluster_mask)

            if cluster_size == 0:
                continue

            persona_profile = {
                "cluster_id": cluster_id,
                "size": int(cluster_size),
                "percentage": float(cluster_size / len(kmeans_labels) * 100),
                "modality_profiles": {},
            }

            # 각 모달리티별 특성 분석
            for modality_name, data in multimodal_data.items():
                cluster_data = data[cluster_mask]

                modality_profile = {
                    "mean": np.mean(cluster_data, axis=0).tolist(),
                    "std": np.std(cluster_data, axis=0).tolist(),
                    "median": np.median(cluster_data, axis=0).tolist(),
                    "min": np.min(cluster_data, axis=0).tolist(),
                    "max": np.max(cluster_data, axis=0).tolist(),
                    "shape": cluster_data.shape,
                }

                # 모달리티별 대표 특성 (평균)
                modality_mean = np.mean(cluster_data, axis=1)
                modality_profile["overall_mean"] = float(np.mean(modality_mean))
                modality_profile["overall_std"] = float(np.std(modality_mean))

                persona_profile["modality_profiles"][modality_name] = modality_profile

            # 페르소나 특성 정의
            persona_characteristics = self._define_persona_characteristics(
                persona_profile
            )
            persona_profile["characteristics"] = persona_characteristics

            persona_profiles[f"persona_{cluster_id}"] = persona_profile

        # DBSCAN 기반 페르소나 프로파일링 (노이즈 제외)
        unique_dbscan_labels = [label for label in set(dbscan_labels) if label != -1]

        for cluster_id in unique_dbscan_labels:
            cluster_mask = dbscan_labels == cluster_id
            cluster_size = np.sum(cluster_mask)

            if cluster_size < 10:  # 너무 작은 클러스터는 제외
                continue

            persona_profile = {
                "cluster_id": int(cluster_id),
                "size": int(cluster_size),
                "percentage": float(cluster_size / len(dbscan_labels) * 100),
                "modality_profiles": {},
                "method": "dbscan",
            }

            # 각 모달리티별 특성 분석
            for modality_name, data in multimodal_data.items():
                cluster_data = data[cluster_mask]

                modality_profile = {
                    "mean": np.mean(cluster_data, axis=0).tolist(),
                    "std": np.std(cluster_data, axis=0).tolist(),
                    "median": np.median(cluster_data, axis=0).tolist(),
                    "min": np.min(cluster_data, axis=0).tolist(),
                    "max": np.max(cluster_data, axis=0).tolist(),
                    "shape": cluster_data.shape,
                }

                # 모달리티별 대표 특성 (평균)
                modality_mean = np.mean(cluster_data, axis=1)
                modality_profile["overall_mean"] = float(np.mean(modality_mean))
                modality_profile["overall_std"] = float(np.std(modality_mean))

                persona_profile["modality_profiles"][modality_name] = modality_profile

            # 페르소나 특성 정의
            persona_characteristics = self._define_persona_characteristics(
                persona_profile
            )
            persona_profile["characteristics"] = persona_characteristics

            persona_profiles[f"dbscan_persona_{cluster_id}"] = persona_profile

        print(f"✅ 페르소나 프로파일링 완료:")
        print(
            f"   K-Means 페르소나 수: {len([k for k in persona_profiles.keys() if k.startswith('persona_')])}"
        )
        print(
            f"   DBSCAN 페르소나 수: {len([k for k in persona_profiles.keys() if k.startswith('dbscan_persona_')])}"
        )

        return persona_profiles

    def _define_persona_characteristics(
        self, persona_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """페르소나 특성 정의"""
        characteristics = {
            "persona_name": "",
            "description": "",
            "key_traits": [],
            "risk_level": "unknown",
            "health_status": "unknown",
            "personality_type": "unknown",
        }

        # Big5 특성 분석
        if "big5" in persona_profile["modality_profiles"]:
            big5_profile = persona_profile["modality_profiles"]["big5"]
            big5_mean = big5_profile["overall_mean"]

            # Big5 기반 성격 유형 추정
            if big5_mean > 0.6:
                characteristics["personality_type"] = "high_engagement"
            elif big5_mean < 0.4:
                characteristics["personality_type"] = "low_engagement"
            else:
                characteristics["personality_type"] = "moderate_engagement"

        # CMI 특성 분석
        if "cmi" in persona_profile["modality_profiles"]:
            cmi_profile = persona_profile["modality_profiles"]["cmi"]
            cmi_mean = cmi_profile["overall_mean"]

            if cmi_mean > 0.7:
                characteristics["health_status"] = "high_risk"
                characteristics["key_traits"].append("high_stress")
            elif cmi_mean < 0.3:
                characteristics["health_status"] = "low_risk"
                characteristics["key_traits"].append("low_stress")
            else:
                characteristics["health_status"] = "moderate_risk"

        # RPPG 특성 분석
        if "rppg" in persona_profile["modality_profiles"]:
            rppg_profile = persona_profile["modality_profiles"]["rppg"]
            rppg_mean = rppg_profile["overall_mean"]

            if rppg_mean > 0.6:
                characteristics["key_traits"].append("high_physiological_activity")
            elif rppg_mean < 0.4:
                characteristics["key_traits"].append("low_physiological_activity")

        # Voice 특성 분석
        if "voice" in persona_profile["modality_profiles"]:
            voice_profile = persona_profile["modality_profiles"]["voice"]
            voice_mean = voice_profile["overall_mean"]

            if voice_mean > 0.6:
                characteristics["key_traits"].append("high_vocal_activity")
            elif voice_mean < 0.4:
                characteristics["key_traits"].append("low_vocal_activity")

        # 페르소나 이름 생성
        if (
            characteristics["health_status"] == "high_risk"
            and characteristics["personality_type"] == "high_engagement"
        ):
            characteristics["persona_name"] = "고위험 고활동형"
            characteristics["description"] = (
                "높은 스트레스와 높은 참여도를 보이는 사용자 그룹"
            )
        elif (
            characteristics["health_status"] == "low_risk"
            and characteristics["personality_type"] == "low_engagement"
        ):
            characteristics["persona_name"] = "저위험 저활동형"
            characteristics["description"] = (
                "낮은 스트레스와 낮은 참여도를 보이는 사용자 그룹"
            )
        elif (
            characteristics["health_status"] == "moderate_risk"
            and characteristics["personality_type"] == "moderate_engagement"
        ):
            characteristics["persona_name"] = "균형형"
            characteristics["description"] = (
                "중간 수준의 스트레스와 참여도를 보이는 사용자 그룹"
            )
        else:
            characteristics["persona_name"] = "혼합형"
            characteristics["description"] = "복합적 특성을 보이는 사용자 그룹"

        return characteristics

    def create_visualizations(
        self,
        X_latent: np.ndarray,
        latent_space_info: Dict[str, Any],
        clustering_results: Dict[str, Any],
        persona_profiles: Dict[str, Any],
    ) -> Dict[str, Any]:
        """시각화 자료 생성"""
        print("📊 시각화 자료 생성 중...")

        visualizations = {}

        # 1. PCA 2D 시각화
        if X_latent.shape[1] >= 2:
            plt.figure(figsize=(12, 8))

            # K-Means 클러스터 시각화
            plt.subplot(2, 2, 1)
            kmeans_labels = np.array(clustering_results["kmeans"]["labels"])
            scatter = plt.scatter(
                X_latent[:, 0],
                X_latent[:, 1],
                c=kmeans_labels,
                cmap="viridis",
                alpha=0.6,
            )
            plt.title("K-Means 클러스터링 (PCA)")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.colorbar(scatter)

            # DBSCAN 클러스터 시각화
            plt.subplot(2, 2, 2)
            dbscan_labels = np.array(clustering_results["dbscan"]["labels"])
            scatter = plt.scatter(
                X_latent[:, 0],
                X_latent[:, 1],
                c=dbscan_labels,
                cmap="viridis",
                alpha=0.6,
            )
            plt.title("DBSCAN 클러스터링 (PCA)")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.colorbar(scatter)

            # 실루엣 점수 비교
            plt.subplot(2, 2, 3)
            k_range = clustering_results["k_range"]
            silhouette_scores = clustering_results["silhouette_scores"]
            plt.plot(k_range, silhouette_scores, "bo-")
            plt.xlabel("클러스터 수")
            plt.ylabel("실루엣 점수")
            plt.title("K-Means 실루엣 점수")
            plt.grid(True)

            # 엘보우 곡선
            plt.subplot(2, 2, 4)
            inertias = clustering_results["inertias"]
            plt.plot(k_range, inertias, "ro-")
            plt.xlabel("클러스터 수")
            plt.ylabel("Inertia")
            plt.title("K-Means 엘보우 곡선")
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(
                "persona_discovery_clustering.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

            visualizations["clustering_plot"] = "persona_discovery_clustering.png"

        # 2. t-SNE 시각화
        if "sample_indices" in latent_space_info:
            sample_indices = latent_space_info["sample_indices"]
            X_tsne = latent_space_info["tsne"].embedding_

            plt.figure(figsize=(15, 5))

            # t-SNE + K-Means
            plt.subplot(1, 3, 1)
            sample_kmeans_labels = kmeans_labels[sample_indices]
            scatter = plt.scatter(
                X_tsne[:, 0],
                X_tsne[:, 1],
                c=sample_kmeans_labels,
                cmap="viridis",
                alpha=0.6,
            )
            plt.title("t-SNE + K-Means 클러스터링")
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.colorbar(scatter)

            # t-SNE + DBSCAN
            plt.subplot(1, 3, 2)
            sample_dbscan_labels = dbscan_labels[sample_indices]
            scatter = plt.scatter(
                X_tsne[:, 0],
                X_tsne[:, 1],
                c=sample_dbscan_labels,
                cmap="viridis",
                alpha=0.6,
            )
            plt.title("t-SNE + DBSCAN 클러스터링")
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.colorbar(scatter)

            # 클러스터 크기 분포
            plt.subplot(1, 3, 3)
            kmeans_counts = np.bincount(kmeans_labels)
            dbscan_counts = np.bincount(dbscan_labels[dbscan_labels >= 0])

            x_pos = np.arange(len(kmeans_counts))
            plt.bar(x_pos - 0.2, kmeans_counts, 0.4, label="K-Means", alpha=0.7)
            if len(dbscan_counts) > 0:
                plt.bar(
                    x_pos[: len(dbscan_counts)] + 0.2,
                    dbscan_counts,
                    0.4,
                    label="DBSCAN",
                    alpha=0.7,
                )
            plt.xlabel("클러스터 ID")
            plt.ylabel("클러스터 크기")
            plt.title("클러스터 크기 분포")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig("persona_discovery_tsne.png", dpi=300, bbox_inches="tight")
            plt.close()

            visualizations["tsne_plot"] = "persona_discovery_tsne.png"

        # 3. 페르소나 특성 히트맵
        if persona_profiles:
            plt.figure(figsize=(15, 10))

            # K-Means 페르소나들만 선택
            kmeans_personas = {
                k: v for k, v in persona_profiles.items() if k.startswith("persona_")
            }

            if kmeans_personas:
                # 각 페르소나의 모달리티별 평균값 추출
                persona_matrix = []
                persona_names = []

                for persona_name, persona_data in kmeans_personas.items():
                    persona_names.append(
                        persona_data["characteristics"]["persona_name"]
                    )
                    persona_row = []

                    for modality in ["big5", "cmi", "rppg", "voice"]:
                        if modality in persona_data["modality_profiles"]:
                            persona_row.append(
                                persona_data["modality_profiles"][modality][
                                    "overall_mean"
                                ]
                            )
                        else:
                            persona_row.append(0)

                    persona_matrix.append(persona_row)

                persona_matrix = np.array(persona_matrix)

                # 히트맵 생성
                plt.subplot(2, 2, 1)
                sns.heatmap(
                    persona_matrix,
                    xticklabels=["Big5", "CMI", "RPPG", "Voice"],
                    yticklabels=persona_names,
                    annot=True,
                    fmt=".3f",
                    cmap="RdYlBu_r",
                )
                plt.title("페르소나별 모달리티 특성 히트맵")
                plt.xlabel("모달리티")
                plt.ylabel("페르소나")

                # 페르소나 크기 분포
                plt.subplot(2, 2, 2)
                persona_sizes = [
                    persona_data["size"] for persona_data in kmeans_personas.values()
                ]
                persona_labels = [
                    persona_data["characteristics"]["persona_name"]
                    for persona_data in kmeans_personas.values()
                ]
                plt.pie(
                    persona_sizes,
                    labels=persona_labels,
                    autopct="%1.1f%%",
                    startangle=90,
                )
                plt.title("페르소나 크기 분포")

                # 모달리티별 분산
                plt.subplot(2, 2, 3)
                modality_vars = np.var(persona_matrix, axis=0)
                plt.bar(["Big5", "CMI", "RPPG", "Voice"], modality_vars)
                plt.title("모달리티별 분산")
                plt.ylabel("분산")
                plt.xticks(rotation=45)

                # 페르소나별 특성 요약
                plt.subplot(2, 2, 4)
                persona_summary = []
                for persona_data in kmeans_personas.values():
                    summary = f"{persona_data['characteristics']['persona_name']}\n"
                    summary += f"크기: {persona_data['size']} ({persona_data['percentage']:.1f}%)\n"
                    summary += f"특성: {', '.join(persona_data['characteristics']['key_traits'])}"
                    persona_summary.append(summary)

                plt.text(
                    0.1,
                    0.9,
                    "\n\n".join(persona_summary),
                    transform=plt.gca().transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
                )
                plt.axis("off")
                plt.title("페르소나 요약")

            plt.tight_layout()
            plt.savefig("persona_discovery_analysis.png", dpi=300, bbox_inches="tight")
            plt.close()

            visualizations["analysis_plot"] = "persona_discovery_analysis.png"

        print(f"✅ 시각화 자료 생성 완료:")
        for plot_name, plot_file in visualizations.items():
            print(f"   {plot_name}: {plot_file}")

        return visualizations

    def run_persona_discovery_operation(self, limit: int = 10000) -> Dict[str, Any]:
        """페르소나 발견 작전 실행"""
        print("🚀 페르소나 발견 작전 시작")
        print("=" * 60)
        print("🎯 목표: 예측 → 발견, 회귀 → 클러스터링, 성능 → 인사이트")

        # 1. 멀티모달 데이터 로딩
        multimodal_data = self.load_multimodal_data(limit)

        # 2. 통합 잠재 공간 생성
        X_latent, latent_space_info = self.create_unified_latent_space(multimodal_data)

        # 3. 페르소나 발견 (클러스터링)
        clustering_results = self.discover_personas(X_latent, latent_space_info)

        # 4. 페르소나 프로파일링
        persona_profiles = self.profile_personas(multimodal_data, clustering_results)

        # 5. 시각화 자료 생성
        visualizations = self.create_visualizations(
            X_latent, latent_space_info, clustering_results, persona_profiles
        )

        # 6. 결과 통합
        results = {
            "operation_summary": {
                "total_samples": len(multimodal_data["big5"]),
                "n_features_original": latent_space_info["original_features"],
                "n_features_pca": latent_space_info["pca_components"],
                "kmeans_clusters": clustering_results["kmeans"]["n_clusters"],
                "dbscan_clusters": clustering_results["dbscan"]["n_clusters"],
                "n_personas_discovered": len(persona_profiles),
                "visualizations_created": len(visualizations),
            },
            "latent_space_info": latent_space_info,
            "clustering_results": clustering_results,
            "persona_profiles": persona_profiles,
            "visualizations": visualizations,
            "insights": self._generate_insights(persona_profiles, clustering_results),
        }

        # 7. 결과 저장 (JSON 직렬화 가능한 객체만 저장)
        json_safe_results = self._convert_to_json_serializable(results)
        with open("persona_discovery_operation_results.json", "w") as f:
            json.dump(json_safe_results, f, indent=2)

        print("✅ 페르소나 발견 작전 완료!")
        print(f"   발견된 페르소나 수: {len(persona_profiles)}")
        print(f"   생성된 시각화: {len(visualizations)}")
        print(f"   결과 파일: persona_discovery_operation_results.json")

        return results

    def _generate_insights(
        self, persona_profiles: Dict[str, Any], clustering_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """인사이트 생성"""
        insights = {"key_findings": [], "persona_summary": {}, "recommendations": []}

        # 주요 발견사항
        kmeans_personas = {
            k: v for k, v in persona_profiles.items() if k.startswith("persona_")
        }
        dbscan_personas = {
            k: v for k, v in persona_profiles.items() if k.startswith("dbscan_persona_")
        }

        insights["key_findings"].append(f"총 {len(persona_profiles)}개의 페르소나 발견")
        insights["key_findings"].append(
            f"K-Means: {len(kmeans_personas)}개, DBSCAN: {len(dbscan_personas)}개"
        )

        # 가장 큰 페르소나 식별
        if kmeans_personas:
            largest_persona = max(kmeans_personas.values(), key=lambda x: x["size"])
            insights["key_findings"].append(
                f"가장 큰 페르소나: {largest_persona['characteristics']['persona_name']} ({largest_persona['size']}명, {largest_persona['percentage']:.1f}%)"
            )

        # 페르소나 요약
        for persona_name, persona_data in kmeans_personas.items():
            insights["persona_summary"][
                persona_data["characteristics"]["persona_name"]
            ] = {
                "size": persona_data["size"],
                "percentage": persona_data["percentage"],
                "key_traits": persona_data["characteristics"]["key_traits"],
                "health_status": persona_data["characteristics"]["health_status"],
                "personality_type": persona_data["characteristics"]["personality_type"],
            }

        # 권장사항
        insights["recommendations"].append("각 페르소나별 맞춤형 개입 전략 수립 필요")
        insights["recommendations"].append(
            "고위험 페르소나에 대한 우선적 모니터링 강화"
        )
        insights["recommendations"].append(
            "페르소나 간 전환 패턴 분석을 통한 예측 모델 개발"
        )

        return insights

    def _convert_to_json_serializable(self, obj):
        """JSON 직렬화 가능한 객체로 변환"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, bool):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif hasattr(obj, "__dict__"):
            # 객체의 경우 문자열로 변환
            return str(obj)
        else:
            return obj


def main():
    """메인 실행 함수"""
    print("🚀 페르소나 발견 작전")
    print("=" * 60)

    operation = PersonaDiscoveryOperation()
    results = operation.run_persona_discovery_operation(limit=10000)

    print("\n📊 페르소나 발견 작전 결과:")
    print(
        f"   발견된 페르소나 수: {results['operation_summary']['n_personas_discovered']}"
    )
    print(f"   K-Means 클러스터: {results['operation_summary']['kmeans_clusters']}")
    print(f"   DBSCAN 클러스터: {results['operation_summary']['dbscan_clusters']}")
    print(f"   생성된 시각화: {results['operation_summary']['visualizations_created']}")

    print("\n🎯 주요 인사이트:")
    for finding in results["insights"]["key_findings"]:
        print(f"   • {finding}")


if __name__ == "__main__":
    main()
