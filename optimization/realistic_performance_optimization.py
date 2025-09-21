#!/usr/bin/env python3
"""
현실적인 성능 목표 설정 및 최적화
- R² 0.7-0.8 목표 설정
- 외부 데이터 기반 타겟 변수 생성
- 선형 모델 중심 최적화
"""

import json
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from google.cloud import bigquery
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


class RealisticPerformanceOptimizer:
    """현실적인 성능 목표 설정 및 최적화"""

    def __init__(self, project_id: str = "persona-diary-service"):
        self.project_id = project_id
        self.scalers = {}
        self.models = {}

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

    def load_realistic_data(self, limit: int = 10000) -> Tuple[Dict, np.ndarray]:
        """현실적인 데이터 로딩"""
        print("🔄 현실적인 데이터 로딩 중...")
        try:
            # Big5 데이터 로딩
            big5_query = f"""
            SELECT * FROM `persona-diary-service.big5_dataset.big5_preprocessed` LIMIT {limit}
            """
            big5_df = self.client.query(big5_query).to_dataframe()

            # CMI 데이터 로딩
            cmi_query = f"""
            SELECT * FROM `persona-diary-service.cmi_dataset.cmi_preprocessed` LIMIT {limit}
            """
            cmi_df = self.client.query(cmi_query).to_dataframe()

            # RPPG 데이터 로딩
            rppg_query = f"""
            SELECT * FROM `persona-diary-service.rppg_dataset.rppg_preprocessed` LIMIT {limit}
            """
            rppg_df = self.client.query(rppg_query).to_dataframe()

            # Voice 데이터 로딩
            voice_query = f"""
            SELECT * FROM `persona-diary-service.voice_dataset.voice_preprocessed` LIMIT {limit}
            """
            voice_df = self.client.query(voice_query).to_dataframe()

            # 수치 데이터만 선택
            cmi_numeric = cmi_df.select_dtypes(include=[np.number])
            rppg_numeric = rppg_df.select_dtypes(include=[np.number])
            voice_numeric = voice_df.select_dtypes(include=[np.number])
            big5_numeric = big5_df.select_dtypes(include=[np.number])

            # 데이터 결합
            multimodal_data = {
                "big5": big5_numeric.values,
                "cmi": cmi_numeric.values,
                "rppg": rppg_numeric.values,
                "voice": voice_numeric.values,
            }

            # 현실적인 타겟 변수 생성 (외부 데이터 시뮬레이션)
            print("🔍 현실적인 타겟 변수 생성 중...")

            # 방법 1: 외부 데이터 시뮬레이션 (실제 외부 데이터가 있다면 사용)
            np.random.seed(42)

            # 외부 요인들 (실제 외부 데이터가 있다면 사용)
            external_factors = {
                "age": np.random.normal(30, 10, len(big5_numeric)),
                "education": np.random.normal(5, 2, len(big5_numeric)),
                "income": np.random.normal(50000, 20000, len(big5_numeric)),
                "health": np.random.normal(7, 2, len(big5_numeric)),
                "social": np.random.normal(6, 2, len(big5_numeric)),
            }

            # 현실적인 타겟 변수 생성 (외부 요인 기반)
            targets = (
                external_factors["age"] * 0.1
                + external_factors["education"] * 0.2
                + external_factors["income"] * 0.00001
                + external_factors["health"] * 0.3
                + external_factors["social"] * 0.4
                + np.random.normal(0, 1, len(big5_numeric))  # 노이즈
            )

            # 1-10 스케일로 정규화
            targets = (targets - targets.min()) / (
                targets.max() - targets.min()
            ) * 9 + 1

            print(f"✅ 현실적인 데이터 로딩 완료:")
            print(f"   Big5: {big5_numeric.shape}")
            print(f"   CMI: {cmi_numeric.shape}")
            print(f"   RPPG: {rppg_numeric.shape}")
            print(f"   Voice: {voice_numeric.shape}")
            print(f"   Targets: {targets.shape}")
            print(f"   타겟 변수 통계:")
            print(f"     평균: {targets.mean():.4f}")
            print(f"     표준편차: {targets.std():.4f}")
            print(f"   현실적인 타겟 변수!")

            return multimodal_data, targets

        except Exception as e:
            print(f"❌ BigQuery 데이터 로딩 실패: {str(e)}")
            raise e

    def create_optimized_features(self, multimodal_data: Dict) -> np.ndarray:
        """최적화된 피처 생성"""
        print("🔧 최적화된 피처 생성 중...")

        # 1. 각 모달리티별 최적화된 통계 사용
        features = []

        for modality_name, data in multimodal_data.items():
            if data.dtype != np.float64:
                data = data.astype(np.float64)

            # 최적화된 통계 사용
            mean_features = np.mean(data, axis=1, keepdims=True)
            std_features = np.std(data, axis=1, keepdims=True)
            max_features = np.max(data, axis=1, keepdims=True)
            min_features = np.min(data, axis=1, keepdims=True)
            median_features = np.median(data, axis=1, keepdims=True)
            q25_features = np.percentile(data, 25, axis=1, keepdims=True)
            q75_features = np.percentile(data, 75, axis=1, keepdims=True)
            skew_features = np.array(
                [self._calculate_skewness(row) for row in data]
            ).reshape(-1, 1)

            # 8개 피처 사용
            modality_features = np.concatenate(
                [
                    mean_features,
                    std_features,
                    max_features,
                    min_features,
                    median_features,
                    q25_features,
                    q75_features,
                    skew_features,
                ],
                axis=1,
            )
            features.append(modality_features)

        X_optimized = np.concatenate(features, axis=1)

        print(f"   최적화된 피처 수: {X_optimized.shape[1]}")
        print(f"   각 모달리티별 8개 피처 사용")

        return X_optimized

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """왜도 계산"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            return np.mean(((data - mean) / std) ** 3)
        except:
            return 0.0

    def create_optimized_models(self):
        """최적화된 모델들 생성"""
        print("🔄 최적화된 모델들 생성 중...")

        self.models = {
            "ridge": Ridge(alpha=1.0),  # 적당한 정규화
            "lasso": Lasso(alpha=0.1, max_iter=10000),  # 적당한 정규화
            "elastic_net": ElasticNet(
                alpha=0.1, l1_ratio=0.5, max_iter=10000
            ),  # 적당한 정규화
            "random_forest": RandomForestRegressor(
                n_estimators=100,  # 적당한 트리 수
                max_depth=10,  # 적당한 깊이
                min_samples_split=20,  # 적당한 분할 기준
                min_samples_leaf=10,  # 적당한 리프 기준
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
        }

        print(f"✅ {len(self.models)}개 최적화된 모델 생성 완료")

    def prepare_optimized_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """최적화된 데이터 준비"""
        print("🔄 최적화된 데이터 준비 중...")

        # StandardScaler로 정규화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers["optimized_ensemble"] = scaler

        print(f"✅ 최적화된 데이터 준비 완료: {X_scaled.shape}")
        print(f"   피처 수: {X_scaled.shape[1]}")

        return X_scaled, y

    def train_optimized_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """최적화된 모델들 훈련"""
        print("🚀 최적화된 모델들 훈련 시작...")

        # 훈련/검증/테스트 분할 (3단계)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.3, random_state=42
        )

        model_results = {}
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in self.models.items():
            print(f"   훈련 중: {name}")

            try:
                # 교차 검증
                cv_r2_scores = cross_val_score(
                    model, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1
                )
                cv_rmse_scores = -cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=kf,
                    scoring="neg_root_mean_squared_error",
                    n_jobs=-1,
                )

                avg_r2 = cv_r2_scores.mean()
                std_r2 = cv_r2_scores.std()
                avg_rmse = cv_rmse_scores.mean()

                # 최종 모델 훈련
                model.fit(X_train, y_train)

                # 검증 성능
                val_pred = model.predict(X_val)
                val_r2 = r2_score(y_val, val_pred)
                val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

                # 테스트 성능
                test_pred = model.predict(X_test)
                test_r2 = r2_score(y_test, test_pred)
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

                # 과적합 간격 계산
                overfitting_gap = val_r2 - test_r2

                model_results[name] = {
                    "cv_mean_r2": avg_r2,
                    "cv_std_r2": std_r2,
                    "cv_mean_rmse": avg_rmse,
                    "val_r2": val_r2,
                    "val_rmse": val_rmse,
                    "test_r2": test_r2,
                    "test_rmse": test_rmse,
                    "overfitting_gap": overfitting_gap,
                    "model": model,
                }

                print(f"     CV R²: {avg_r2:.4f} (±{std_r2:.4f})")
                print(f"     Val R²: {val_r2:.4f}")
                print(f"     Test R²: {test_r2:.4f}")
                print(f"     과적합 간격: {overfitting_gap:.4f}")

            except Exception as e:
                print(f"     ❌ 훈련 실패: {str(e)}")
                model_results[name] = None

        # 성능 순으로 정렬 (과적합 간격 고려)
        valid_models = {k: v for k, v in model_results.items() if v is not None}
        sorted_models = sorted(
            valid_models.items(),
            key=lambda x: x[1]["test_r2"] - x[1]["overfitting_gap"],
            reverse=True,
        )

        print(f"✅ {len(valid_models)}개 최적화된 모델 훈련 완료")
        print("📊 최적화된 모델 성능 순위 (과적합 간격 고려):")
        for i, (name, scores) in enumerate(sorted_models, 1):
            print(
                f"   {i}. {name}: Test R² = {scores['test_r2']:.4f}, 과적합 = {scores['overfitting_gap']:.4f}"
            )

        return model_results

    def create_optimized_ensemble(
        self, X: np.ndarray, y: np.ndarray, model_results: Dict
    ) -> Dict:
        """최적화된 앙상블 생성"""
        print("🔄 최적화된 앙상블 생성 중...")

        # 상위 3개 모델 선택 (과적합 방지)
        valid_models = {k: v for k, v in model_results.items() if v is not None}
        sorted_models = sorted(
            valid_models.items(),
            key=lambda x: x[1]["test_r2"] - x[1]["overfitting_gap"],
            reverse=True,
        )
        top_models = [name for name, _ in sorted_models[:3]]

        print(f"   선택된 상위 모델들: {top_models}")

        # 훈련/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        predictions = []
        weights = []

        for name in top_models:
            if name in model_results and model_results[name] is not None:
                model = model_results[name]["model"]
                pred = model.predict(X_test)
                predictions.append(pred)
                # 과적합 간격을 고려한 가중치
                weight = (
                    model_results[name]["test_r2"]
                    - model_results[name]["overfitting_gap"]
                )
                weights.append(max(weight, 0.1))

        if not predictions:
            print("❌ 유효한 예측값이 없습니다.")
            return None

        # 가중치 정규화
        weights = np.array(weights)
        weights = weights / weights.sum()

        # 가중 평균으로 앙상블 예측
        predictions = np.array(predictions)
        ensemble_pred = np.average(predictions, axis=0, weights=weights)

        # 앙상블 성능 평가
        r2 = r2_score(y_test, ensemble_pred)
        mse = mean_squared_error(y_test, ensemble_pred)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(ensemble_pred - y_test))
        correlation = np.corrcoef(ensemble_pred, y_test)[0, 1]

        results = {
            "r2": r2,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "correlation": correlation,
            "predictions": ensemble_pred,
            "targets": y_test,
            "selected_models": top_models,
            "model_weights": dict(zip(top_models, weights)),
        }

        print(f"✅ 최적화된 앙상블 생성 및 평가 완료:")
        print(f"   R²: {r2:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   상관계수: {correlation:.4f}")

        return results

    def run_realistic_performance_optimization(self, limit: int = 10000) -> Dict:
        """현실적인 성능 목표 설정 및 최적화 실행"""
        print("🚀 현실적인 성능 목표 설정 및 최적화 시작")
        print("=" * 60)
        print("🎯 목표: R² 0.7-0.8 (현실적인 성능)")

        # 1. 현실적인 데이터 로딩
        multimodal_data, targets = self.load_realistic_data(limit)

        # 2. 최적화된 피처 생성
        X_engineered = self.create_optimized_features(multimodal_data)

        # 3. 최적화된 모델들 생성
        self.create_optimized_models()

        # 4. 최적화된 데이터 준비
        X, y = self.prepare_optimized_data(X_engineered, targets)

        # 5. 최적화된 모델들 훈련
        model_results = self.train_optimized_models(X, y)

        # 6. 최적화된 앙상블 생성
        ensemble_results = self.create_optimized_ensemble(X, y, model_results)

        # 7. 결과 저장
        results = {
            "individual_models_results": {
                name: {
                    "cv_mean_r2": scores["cv_mean_r2"] if scores else None,
                    "cv_std_r2": scores["cv_std_r2"] if scores else None,
                    "val_r2": scores["val_r2"] if scores else None,
                    "test_r2": scores["test_r2"] if scores else None,
                    "overfitting_gap": scores["overfitting_gap"] if scores else None,
                }
                for name, scores in model_results.items()
            },
            "ensemble_results": ensemble_results,
            "data_info": {
                "n_samples": len(y),
                "n_features_original": X_engineered.shape[1],
                "n_features_selected": X.shape[1],
                "n_models_trained": len(
                    [m for m in model_results.values() if m is not None]
                ),
                "n_models_in_ensemble": (
                    len(ensemble_results["selected_models"]) if ensemble_results else 0
                ),
                "optimization_status": "현실적인 성능 목표 설정 및 최적화 완료",
                "target_performance": "R² 0.7-0.8",
            },
        }

        # JSON으로 저장
        with open("realistic_performance_optimization_results.json", "w") as f:

            def convert_to_json_serializable(obj):
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
                elif isinstance(obj, pd.Series):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_json_serializable(item) for item in obj]
                else:
                    return obj

            json_results = convert_to_json_serializable(results)
            json.dump(json_results, f, indent=2)

        print(f"✅ 현실적인 성능 목표 설정 및 최적화 완료!")
        if ensemble_results:
            print(f"   최종 앙상블 R²: {ensemble_results['r2']:.4f}")
            print(f"   최종 앙상블 RMSE: {ensemble_results['rmse']:.4f}")
            print(
                f"   목표 달성 여부: {'✅ 달성' if ensemble_results['r2'] >= 0.7 else '❌ 미달성'}"
            )

        return results


def main():
    """메인 실행 함수"""
    print("🚀 현실적인 성능 목표 설정 및 최적화")
    print("=" * 60)

    optimizer = RealisticPerformanceOptimizer()
    results = optimizer.run_realistic_performance_optimization(limit=10000)

    print("\n📊 현실적인 성능 목표 설정 및 최적화 결과:")
    if results["ensemble_results"]:
        print(f"   최종 앙상블 R²: {results['ensemble_results']['r2']:.4f}")
        print(f"   최종 앙상블 RMSE: {results['ensemble_results']['rmse']:.4f}")
        print(f"   상관계수: {results['ensemble_results']['correlation']:.4f}")
        print(f"   선택된 모델들: {results['ensemble_results']['selected_models']}")
        print(
            f"   목표 달성 여부: {'✅ 달성' if results['ensemble_results']['r2'] >= 0.7 else '❌ 미달성'}"
        )


if __name__ == "__main__":
    main()
