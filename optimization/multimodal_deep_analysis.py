#!/usr/bin/env python3
"""
멀티모달 특성 심층 분석
- 각 모달리티 통계적 특성 심층 조사
- 교차 상관관계 정밀 분석
- 잠재 공통 변수 발견
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
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


class MultimodalDeepAnalyzer:
    """멀티모달 특성 심층 분석기"""

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
                "big5": big5_numeric.values,
                "cmi": cmi_numeric.values,
                "rppg": rppg_numeric.values,
                "voice": voice_numeric.values,
            }

            print(f"✅ 멀티모달 데이터 로딩 완료:")
            print(f"   Big5: {big5_numeric.shape}")
            print(f"   CMI: {cmi_numeric.shape}")
            print(f"   RPPG: {rppg_numeric.values.shape}")
            print(f"   Voice: {voice_numeric.shape}")

            return multimodal_data

        except Exception as e:
            print(f"❌ BigQuery 데이터 로딩 실패: {str(e)}")
            raise e

    def analyze_modality_statistics(
        self, multimodal_data: Dict[str, np.ndarray]
    ) -> Dict[str, Dict]:
        """각 모달리티 통계적 특성 심층 조사"""
        print("🔍 각 모달리티 통계적 특성 분석 중...")

        modality_stats = {}

        for modality_name, data in multimodal_data.items():
            print(f"   분석 중: {modality_name}")

            # 데이터 타입 확인 및 변환
            if data.dtype != np.float64:
                data = data.astype(np.float64)

            # 기본 통계
            stats = {
                "shape": data.shape,
                "mean": np.mean(data, axis=0),
                "std": np.std(data, axis=0),
                "min": np.min(data, axis=0),
                "max": np.max(data, axis=0),
                "median": np.median(data, axis=0),
                "skewness": self._calculate_skewness(data),
                "kurtosis": self._calculate_kurtosis(data),
                "missing_values": np.isnan(data).sum(),
                "zero_values": (data == 0).sum(),
                "negative_values": (data < 0).sum(),
            }

            # 분포 특성
            stats["distribution_type"] = self._analyze_distribution(data)
            stats["outlier_ratio"] = self._calculate_outlier_ratio(data)
            stats["variance_ratio"] = np.var(data, axis=0) / (
                np.mean(data, axis=0) + 1e-8
            )

            modality_stats[modality_name] = stats

            print(f"     분포 유형: {stats['distribution_type']}")
            print(f"     이상치 비율: {stats['outlier_ratio']:.4f}")
            print(f"     결측값: {stats['missing_values']}")

        return modality_stats

    def analyze_cross_correlations(
        self, multimodal_data: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """교차 상관관계 정밀 분석"""
        print("🔗 교차 상관관계 분석 중...")

        # 각 모달리티의 대표 특성 추출 (평균값)
        modality_representatives = {}
        for modality_name, data in multimodal_data.items():
            if data.shape[1] > 0:
                # 데이터 타입 확인 및 변환
                if data.dtype != np.float64:
                    data = data.astype(np.float64)
                # 각 모달리티의 평균값을 대표값으로 사용
                modality_representatives[modality_name] = np.mean(data, axis=1)

        # 상관관계 매트릭스 계산
        correlation_matrix = np.corrcoef(list(modality_representatives.values()))
        modality_names = list(modality_representatives.keys())

        # 상호정보량 계산
        mutual_info_matrix = np.zeros((len(modality_names), len(modality_names)))
        for i, (name1, data1) in enumerate(modality_representatives.items()):
            for j, (name2, data2) in enumerate(modality_representatives.items()):
                if i != j:
                    mi = mutual_info_regression(data1.reshape(-1, 1), data2)[0]
                    mutual_info_matrix[i, j] = mi

        # 피어슨 상관계수
        pearson_correlations = {}
        for i, name1 in enumerate(modality_names):
            for j, name2 in enumerate(modality_names):
                if i != j:
                    corr, p_value = pearsonr(
                        modality_representatives[name1], modality_representatives[name2]
                    )
                    pearson_correlations[f"{name1}_vs_{name2}"] = {
                        "correlation": corr,
                        "p_value": p_value,
                        "significant": p_value < 0.05,
                    }

        # 스피어만 상관계수
        spearman_correlations = {}
        for i, name1 in enumerate(modality_names):
            for j, name2 in enumerate(modality_names):
                if i != j:
                    corr, p_value = spearmanr(
                        modality_representatives[name1], modality_representatives[name2]
                    )
                    spearman_correlations[f"{name1}_vs_{name2}"] = {
                        "correlation": corr,
                        "p_value": p_value,
                        "significant": p_value < 0.05,
                    }

        cross_correlation_results = {
            "correlation_matrix": correlation_matrix.tolist(),
            "modality_names": modality_names,
            "pearson_correlations": pearson_correlations,
            "spearman_correlations": spearman_correlations,
            "mutual_info_matrix": mutual_info_matrix.tolist(),
        }

        print("✅ 교차 상관관계 분석 완료")
        for pair, stats in pearson_correlations.items():
            print(
                f"   {pair}: r={stats['correlation']:.4f}, p={stats['p_value']:.4f}, sig={stats['significant']}"
            )

        return cross_correlation_results

    def discover_latent_common_variables(
        self, multimodal_data: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """잠재 공통 변수 발견"""
        print("🔍 잠재 공통 변수 발견 중...")

        # 모든 모달리티 데이터 결합
        all_data = []
        modality_info = []

        for modality_name, data in multimodal_data.items():
            if data.shape[1] > 0:
                all_data.append(data)
                modality_info.extend([modality_name] * data.shape[1])

        if not all_data:
            return {"error": "유효한 데이터가 없습니다"}

        X_combined = np.concatenate(all_data, axis=1)

        # PCA를 통한 차원 축소
        pca = PCA(n_components=min(10, X_combined.shape[1]))
        X_pca = pca.fit_transform(X_combined)

        # t-SNE를 통한 비선형 차원 축소
        if X_combined.shape[0] > 1000:
            # 큰 데이터셋의 경우 샘플링
            sample_indices = np.random.choice(X_combined.shape[0], 1000, replace=False)
            X_sample = X_combined[sample_indices]
        else:
            X_sample = X_combined

        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X_sample)

        # 각 모달리티별 PCA 성분 분석
        modality_pca_results = {}
        for modality_name, data in multimodal_data.items():
            if data.shape[1] > 1:
                pca_modality = PCA(n_components=min(3, data.shape[1]))
                X_modality_pca = pca_modality.fit_transform(data)

                modality_pca_results[modality_name] = {
                    "explained_variance_ratio": pca_modality.explained_variance_ratio_.tolist(),
                    "cumulative_variance": np.cumsum(
                        pca_modality.explained_variance_ratio_
                    ).tolist(),
                    "components": X_modality_pca.tolist(),
                }

        latent_variables = {
            "combined_pca": {
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "cumulative_variance": np.cumsum(
                    pca.explained_variance_ratio_
                ).tolist(),
                "components": X_pca.tolist(),
            },
            "tsne_embedding": X_tsne.tolist(),
            "modality_pca": modality_pca_results,
            "feature_importance": self._calculate_feature_importance(
                X_combined, modality_info
            ),
        }

        print("✅ 잠재 공통 변수 발견 완료")
        print(f"   PCA 설명 분산 비율: {pca.explained_variance_ratio_[:5]}")
        print(f"   누적 설명 분산: {np.cumsum(pca.explained_variance_ratio_)[:5]}")

        return latent_variables

    def _calculate_skewness(self, data: np.ndarray) -> np.ndarray:
        """왜도 계산"""
        from scipy.stats import skew

        return skew(data, axis=0)

    def _calculate_kurtosis(self, data: np.ndarray) -> np.ndarray:
        """첨도 계산"""
        from scipy.stats import kurtosis

        return kurtosis(data, axis=0)

    def _analyze_distribution(self, data: np.ndarray) -> str:
        """분포 유형 분석"""
        if data.shape[1] == 0:
            return "empty"

        # 첫 번째 특성의 분포 분석
        first_feature = data[:, 0]

        # 정규성 검정 (Shapiro-Wilk)
        from scipy.stats import shapiro

        if len(first_feature) > 5000:
            # 큰 데이터셋의 경우 샘플링
            sample = np.random.choice(first_feature, 5000, replace=False)
        else:
            sample = first_feature

        try:
            stat, p_value = shapiro(sample)
            if p_value > 0.05:
                return "normal"
            else:
                return "non_normal"
        except:
            return "unknown"

    def _calculate_outlier_ratio(self, data: np.ndarray) -> float:
        """이상치 비율 계산 (IQR 방법)"""
        if data.shape[1] == 0:
            return 0.0

        outlier_count = 0
        total_count = data.size

        for i in range(data.shape[1]):
            Q1 = np.percentile(data[:, i], 25)
            Q3 = np.percentile(data[:, i], 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = (data[:, i] < lower_bound) | (data[:, i] > upper_bound)
            outlier_count += np.sum(outliers)

        return outlier_count / total_count

    def _calculate_feature_importance(
        self, X: np.ndarray, modality_info: List[str]
    ) -> Dict[str, float]:
        """특성 중요도 계산"""
        if X.shape[1] == 0:
            return {}

        # 분산 기반 중요도
        variances = np.var(X, axis=0)

        # 모달리티별 평균 분산
        modality_variances = {}
        for modality in set(modality_info):
            modality_indices = [i for i, m in enumerate(modality_info) if m == modality]
            if modality_indices:
                modality_variances[modality] = np.mean(variances[modality_indices])

        return modality_variances

    def run_deep_analysis(self, limit: int = 10000) -> Dict[str, Any]:
        """심층 분석 실행"""
        print("🚀 멀티모달 심층 분석 시작")
        print("=" * 60)

        # 1. 데이터 로딩
        multimodal_data = self.load_multimodal_data(limit)

        # 2. 각 모달리티 통계적 특성 분석
        modality_stats = self.analyze_modality_statistics(multimodal_data)

        # 3. 교차 상관관계 분석
        cross_correlations = self.analyze_cross_correlations(multimodal_data)

        # 4. 잠재 공통 변수 발견
        latent_variables = self.discover_latent_common_variables(multimodal_data)

        # 5. 결과 통합
        results = {
            "modality_statistics": modality_stats,
            "cross_correlations": cross_correlations,
            "latent_variables": latent_variables,
            "analysis_summary": self._generate_analysis_summary(
                modality_stats, cross_correlations, latent_variables
            ),
        }

        # 6. 결과 저장
        try:
            with open("multimodal_deep_analysis_results.json", "w") as f:
                json.dump(self._convert_to_json_serializable(results), f, indent=2)
        except Exception as e:
            print(f"⚠️ JSON 저장 실패: {str(e)}")
            # 간단한 요약만 저장
            summary_only = {
                "analysis_summary": results["analysis_summary"],
                "modality_shapes": {
                    name: stats["shape"]
                    for name, stats in results["modality_statistics"].items()
                },
                "correlation_summary": {
                    "strong_correlations": len(
                        [
                            c
                            for c in results["cross_correlations"][
                                "pearson_correlations"
                            ].values()
                            if abs(c["correlation"]) > 0.5
                        ]
                    ),
                    "weak_correlations": len(
                        [
                            c
                            for c in results["cross_correlations"][
                                "pearson_correlations"
                            ].values()
                            if abs(c["correlation"]) < 0.1
                        ]
                    ),
                },
            }
            with open("multimodal_deep_analysis_summary.json", "w") as f:
                json.dump(summary_only, f, indent=2)

        print("✅ 멀티모달 심층 분석 완료!")
        self._print_analysis_summary(results["analysis_summary"])

        return results

    def _generate_analysis_summary(
        self, modality_stats: Dict, cross_correlations: Dict, latent_variables: Dict
    ) -> Dict[str, Any]:
        """분석 요약 생성"""
        summary = {
            "total_modalities": len(modality_stats),
            "modality_shapes": {
                name: stats["shape"] for name, stats in modality_stats.items()
            },
            "strong_correlations": [],
            "weak_correlations": [],
            "pca_variance_explained": latent_variables.get("combined_pca", {}).get(
                "explained_variance_ratio", []
            )[:5],
            "key_insights": [],
        }

        # 강한 상관관계 식별
        for pair, stats in cross_correlations.get("pearson_correlations", {}).items():
            if abs(stats["correlation"]) > 0.5:
                summary["strong_correlations"].append(
                    {
                        "pair": pair,
                        "correlation": stats["correlation"],
                        "p_value": stats["p_value"],
                    }
                )
            elif abs(stats["correlation"]) < 0.1:
                summary["weak_correlations"].append(
                    {
                        "pair": pair,
                        "correlation": stats["correlation"],
                        "p_value": stats["p_value"],
                    }
                )

        # 핵심 인사이트
        if summary["strong_correlations"]:
            summary["key_insights"].append("일부 모달리티 간 강한 상관관계 발견")
        else:
            summary["key_insights"].append("모든 모달리티 간 상관관계가 약함")

        if latent_variables.get("combined_pca", {}).get("explained_variance_ratio", []):
            first_pc_variance = latent_variables["combined_pca"][
                "explained_variance_ratio"
            ][0]
            if first_pc_variance > 0.3:
                summary["key_insights"].append("첫 번째 주성분이 높은 분산 설명")
            else:
                summary["key_insights"].append("주성분들이 고르게 분산됨")

        return summary

    def _print_analysis_summary(self, summary: Dict[str, Any]):
        """분석 요약 출력"""
        print("\n📊 분석 요약:")
        print(f"   총 모달리티 수: {summary['total_modalities']}")
        print(f"   모달리티 크기: {summary['modality_shapes']}")
        print(f"   강한 상관관계: {len(summary['strong_correlations'])}개")
        print(f"   약한 상관관계: {len(summary['weak_correlations'])}개")
        print(f"   PCA 설명 분산: {summary['pca_variance_explained']}")
        print("   핵심 인사이트:")
        for insight in summary["key_insights"]:
            print(f"     - {insight}")

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
        else:
            return obj


def main():
    """메인 실행 함수"""
    print("🚀 멀티모달 특성 심층 분석")
    print("=" * 60)

    analyzer = MultimodalDeepAnalyzer()
    results = analyzer.run_deep_analysis(limit=10000)

    print("\n📊 심층 분석 결과:")
    print(f"   분석 완료: multimodal_deep_analysis_results.json")
    print(f"   모달리티 수: {len(results['modality_statistics'])}")
    print(f"   상관관계 분석: 완료")
    print(f"   잠재 변수 발견: 완료")


if __name__ == "__main__":
    main()
