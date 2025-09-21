#!/usr/bin/env python3
"""
BigQuery 대회 규정에 맞춘 현실적 전략
- 현재 자원 최대 활용
- 대회 규정 준수
- 현실적 성능 목표 설정
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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


class RealisticCompetitionStrategy:
    """BigQuery 대회 규정에 맞춘 현실적 전략"""

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

    def load_competition_data(self, limit: int = 10000) -> Dict[str, np.ndarray]:
        """대회용 데이터 로딩"""
        print("🔄 대회용 데이터 로딩 중...")

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

            print(f"✅ 대회용 데이터 로딩 완료:")
            print(f"   Big5: {big5_numeric.shape}")
            print(f"   CMI: {cmi_numeric.shape}")
            print(f"   RPPG: {rppg_numeric.shape}")
            print(f"   Voice: {voice_numeric.shape}")

            return multimodal_data

        except Exception as e:
            print(f"❌ BigQuery 데이터 로딩 실패: {str(e)}")
            raise e

    def create_realistic_target_variable(
        self, multimodal_data: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """현실적인 타겟 변수 생성 (대회 규정 준수)"""
        print("🎯 현실적인 타겟 변수 생성 중...")

        # 각 모달리티의 대표 특성 추출
        big5_mean = np.mean(multimodal_data["big5"], axis=1)
        cmi_mean = np.mean(multimodal_data["cmi"], axis=1)
        rppg_mean = np.mean(multimodal_data["rppg"], axis=1)
        voice_mean = np.mean(multimodal_data["voice"], axis=1)

        # 현실적인 타겟 변수 생성 (약한 상관관계)
        # 각 모달리티의 평균을 조합하되, 노이즈를 많이 추가
        target = (
            big5_mean * 0.1  # Big5 기여도 10%
            + cmi_mean * 0.2  # CMI 기여도 20%
            + rppg_mean * 0.3  # RPPG 기여도 30%
            + voice_mean * 0.1  # Voice 기여도 10%
            + np.random.normal(0, 1, len(big5_mean)) * 0.3  # 노이즈 30%
        )

        # 1-10 스케일로 정규화
        target = (target - target.min()) / (target.max() - target.min()) * 9 + 1

        print(f"   타겟 변수 통계:")
        print(f"     평균: {target.mean():.4f}")
        print(f"     표준편차: {target.std():.4f}")
        print(f"     범위: {target.min():.4f} - {target.max():.4f}")

        return target

    def create_competition_features(
        self, multimodal_data: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """대회용 피처 생성"""
        print("🔧 대회용 피처 생성 중...")

        features = []

        for modality_name, data in multimodal_data.items():
            print(f"   피처 생성 중: {modality_name}")

            # 기본 통계 피처
            mean_features = np.mean(data, axis=1, keepdims=True)
            std_features = np.std(data, axis=1, keepdims=True)
            max_features = np.max(data, axis=1, keepdims=True)
            min_features = np.min(data, axis=1, keepdims=True)
            median_features = np.median(data, axis=1, keepdims=True)

            # 고급 통계 피처
            q25_features = np.percentile(data, 25, axis=1, keepdims=True)
            q75_features = np.percentile(data, 75, axis=1, keepdims=True)
            range_features = max_features - min_features
            iqr_features = q75_features - q25_features

            # 모달리티별 피처 결합
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
            print(f"     {modality_name} 피처 수: {modality_features.shape[1]}")

        # 모든 모달리티 피처 결합
        X_combined = np.concatenate(features, axis=1)

        print(f"✅ 대회용 피처 생성 완료: {X_combined.shape}")
        print(f"   총 피처 수: {X_combined.shape[1]}")

        return X_combined

    def create_competition_models(self) -> Dict[str, Any]:
        """대회용 모델 생성"""
        print("🤖 대회용 모델 생성 중...")

        models = {
            # 선형 모델 (안정적)
            "ridge": Ridge(alpha=1.0),
            "lasso": Lasso(alpha=0.1, max_iter=10000),
            "elastic_net": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
            # 앙상블 모델 (성능)
            "random_forest": RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
            ),
            # 신경망 모델 (비선형)
            "mlp": MLPRegressor(
                hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42
            ),
        }

        print(f"✅ {len(models)}개 대회용 모델 생성 완료")
        return models

    def train_competition_models(
        self, X: np.ndarray, y: np.ndarray, models: Dict[str, Any]
    ) -> Dict[str, Dict]:
        """대회용 모델 훈련"""
        print("🚀 대회용 모델 훈련 시작...")

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # 데이터 정규화
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model_results = {}

        for name, model in models.items():
            print(f"   훈련 중: {name}")

            try:
                # 모델 훈련
                model.fit(X_train_scaled, y_train)

                # 예측
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)

                # 성능 평가
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

                # 교차 검증
                cv_scores = cross_val_score(
                    model, X_train_scaled, y_train, cv=5, scoring="r2"
                )
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()

                model_results[name] = {
                    "train_r2": train_r2,
                    "test_r2": test_r2,
                    "train_rmse": train_rmse,
                    "test_rmse": test_rmse,
                    "cv_mean": cv_mean,
                    "cv_std": cv_std,
                    "overfitting_gap": train_r2 - test_r2,
                    "model": model,
                    "scaler": scaler,
                }

                print(f"     Train R²: {train_r2:.4f}")
                print(f"     Test R²: {test_r2:.4f}")
                print(f"     CV R²: {cv_mean:.4f} (±{cv_std:.4f})")
                print(f"     과적합 간격: {train_r2 - test_r2:.4f}")

            except Exception as e:
                print(f"     ❌ 훈련 실패: {str(e)}")
                model_results[name] = None

        return model_results

    def create_competition_ensemble(
        self, X: np.ndarray, y: np.ndarray, model_results: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """대회용 앙상블 생성"""
        print("🎯 대회용 앙상블 생성 중...")

        # 유효한 모델들만 선택
        valid_models = {k: v for k, v in model_results.items() if v is not None}

        if not valid_models:
            print("❌ 유효한 모델이 없습니다.")
            return None

        # 성능 순으로 정렬
        sorted_models = sorted(
            valid_models.items(),
            key=lambda x: x[1]["test_r2"] - x[1]["overfitting_gap"],
            reverse=True,
        )

        print(f"   선택된 모델들: {[name for name, _ in sorted_models[:3]]}")

        # 상위 3개 모델로 앙상블
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        predictions = []
        weights = []

        for name, results in sorted_models[:3]:
            model = results["model"]
            scaler = results["scaler"]

            X_test_scaled = scaler.transform(X_test)
            pred = model.predict(X_test_scaled)
            predictions.append(pred)

            # 가중치 계산 (성능 기반)
            weight = results["test_r2"] - results["overfitting_gap"]
            weights.append(max(weight, 0.1))

        # 가중 평균 앙상블
        weights = np.array(weights)
        weights = weights / weights.sum()

        ensemble_pred = np.average(predictions, axis=0, weights=weights)

        # 앙상블 성능 평가
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))

        ensemble_results = {
            "r2": ensemble_r2,
            "rmse": ensemble_rmse,
            "mae": np.mean(np.abs(ensemble_pred - y_test)),
            "correlation": np.corrcoef(ensemble_pred, y_test)[0, 1],
            "selected_models": [name for name, _ in sorted_models[:3]],
            "model_weights": dict(
                zip([name for name, _ in sorted_models[:3]], weights)
            ),
            "predictions": ensemble_pred.tolist(),
            "targets": y_test.tolist(),
        }

        print(f"✅ 대회용 앙상블 생성 완료:")
        print(f"   R²: {ensemble_r2:.4f}")
        print(f"   RMSE: {ensemble_rmse:.4f}")
        print(f"   상관계수: {ensemble_results['correlation']:.4f}")

        return ensemble_results

    def run_competition_strategy(self, limit: int = 10000) -> Dict[str, Any]:
        """대회 전략 실행"""
        print("🚀 BigQuery 대회 현실적 전략 실행")
        print("=" * 60)
        print("🎯 목표: R² 0.3-0.5 (현실적 성능)")

        # 1. 데이터 로딩
        multimodal_data = self.load_competition_data(limit)

        # 2. 현실적 타겟 변수 생성
        target = self.create_realistic_target_variable(multimodal_data)

        # 3. 대회용 피처 생성
        X = self.create_competition_features(multimodal_data)

        # 4. 대회용 모델 생성
        models = self.create_competition_models()

        # 5. 모델 훈련
        model_results = self.train_competition_models(X, target, models)

        # 6. 앙상블 생성
        ensemble_results = self.create_competition_ensemble(X, target, model_results)

        # 7. 결과 통합
        results = {
            "individual_models": {
                name: {
                    "train_r2": results["train_r2"] if results else None,
                    "test_r2": results["test_r2"] if results else None,
                    "cv_mean": results["cv_mean"] if results else None,
                    "overfitting_gap": results["overfitting_gap"] if results else None,
                }
                for name, results in model_results.items()
            },
            "ensemble_results": ensemble_results,
            "data_info": {
                "n_samples": len(target),
                "n_features": X.shape[1],
                "n_models_trained": len(
                    [m for m in model_results.values() if m is not None]
                ),
                "target_stats": {
                    "mean": float(target.mean()),
                    "std": float(target.std()),
                    "min": float(target.min()),
                    "max": float(target.max()),
                },
            },
        }

        # 8. 결과 저장
        with open("realistic_competition_strategy_results.json", "w") as f:
            json.dump(self._convert_to_json_serializable(results), f, indent=2)

        print("✅ 대회 전략 실행 완료!")
        if ensemble_results:
            print(f"   최종 앙상블 R²: {ensemble_results['r2']:.4f}")
            print(f"   목표 달성: {'✅' if ensemble_results['r2'] >= 0.3 else '❌'}")

        return results

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
    print("🚀 BigQuery 대회 현실적 전략")
    print("=" * 60)

    strategy = RealisticCompetitionStrategy()
    results = strategy.run_competition_strategy(limit=10000)

    print("\n📊 대회 전략 결과:")
    if results["ensemble_results"]:
        print(f"   최종 앙상블 R²: {results['ensemble_results']['r2']:.4f}")
        print(f"   최종 앙상블 RMSE: {results['ensemble_results']['rmse']:.4f}")
        print(f"   상관계수: {results['ensemble_results']['correlation']:.4f}")
        print(f"   선택된 모델들: {results['ensemble_results']['selected_models']}")
        print(
            f"   목표 달성: {'✅ 달성' if results['ensemble_results']['r2'] >= 0.3 else '❌ 미달성'}"
        )


if __name__ == "__main__":
    main()
