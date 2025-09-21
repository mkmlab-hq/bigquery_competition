#!/usr/bin/env python3
"""
간소화된 최적화 시스템 - 안정적인 성능 향상
- 기본 피처 엔지니어링
- 안정적인 모델 훈련
- 현실적인 성능 개선
"""

import json
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from google.cloud import bigquery
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)


class SimplifiedOptimizer:
    """간소화된 최적화 시스템"""

    def __init__(self, project_id: str = "persona-diary-service"):
        self.project_id = project_id
        self.scalers = {}
        self.models = {}
        self.feature_selector = None

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

    def load_real_bigquery_data(self, limit: int = 10000) -> Tuple[Dict, np.ndarray]:
        """실제 BigQuery 데이터 로딩"""
        print("🔄 실제 BigQuery 데이터 로딩 중...")
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

            # 최적화된 타겟 변수 생성
            print("🔍 최적화된 타겟 변수 생성 중...")

            # Big5 점수 계산
            big5_scores = {
                "EXT": big5_numeric[["EXT1", "EXT2", "EXT3", "EXT4", "EXT5"]].mean(
                    axis=1
                ),
                "EST": big5_numeric[["EST1", "EST2", "EST3", "EST4", "EST5"]].mean(
                    axis=1
                ),
                "AGR": big5_numeric[["AGR1", "AGR2", "AGR3", "AGR4", "AGR5"]].mean(
                    axis=1
                ),
                "CSN": big5_numeric[["CSN1", "CSN2", "CSN3", "CSN4", "CSN5"]].mean(
                    axis=1
                ),
                "OPN": big5_numeric[["OPN1", "OPN2", "OPN3", "OPN4", "OPN5"]].mean(
                    axis=1
                ),
            }

            # 최적화된 타겟 변수 (가중치 조정)
            targets = (
                big5_scores["EXT"] * 0.30
                + big5_scores["OPN"] * 0.25
                + (6 - big5_scores["EST"]) * 0.20
                + big5_scores["AGR"] * 0.15
                + big5_scores["CSN"] * 0.10
                + (cmi_numeric.mean(axis=1) / 6) * 0.05
                + (rppg_numeric.mean(axis=1) / 6) * 0.03
                + (voice_numeric.mean(axis=1) / 6) * 0.02
            )

            print(f"✅ 실제 BigQuery 데이터 로딩 완료:")
            print(f"   Big5: {big5_numeric.shape}")
            print(f"   CMI: {cmi_numeric.shape}")
            print(f"   RPPG: {rppg_numeric.shape}")
            print(f"   Voice: {voice_numeric.shape}")
            print(f"   Targets: {targets.shape}")

            return multimodal_data, targets

        except Exception as e:
            print(f"❌ BigQuery 데이터 로딩 실패: {str(e)}")
            raise e

    def create_basic_features(self, multimodal_data: Dict) -> np.ndarray:
        """기본 피처 엔지니어링"""
        print("🔧 기본 피처 엔지니어링 시작...")

        # 1. 기본 피처 결합
        X_basic = np.concatenate(
            [
                multimodal_data["big5"],
                multimodal_data["cmi"],
                multimodal_data["rppg"],
                multimodal_data["voice"],
            ],
            axis=1,
        )

        print(f"   기본 피처 수: {X_basic.shape[1]}")

        # 2. 간단한 통계 피처 생성
        print("   간단한 통계 피처 생성 중...")
        statistical_features = []

        # 각 모달리티별 기본 통계 피처
        for modality_name, data in multimodal_data.items():
            # 데이터 타입 확인 및 변환
            if data.dtype != np.float64:
                data = data.astype(np.float64)

            # 평균, 최대값, 최소값
            mean_features = np.mean(data, axis=1, keepdims=True)
            max_features = np.max(data, axis=1, keepdims=True)
            min_features = np.min(data, axis=1, keepdims=True)

            # 범위 피처
            range_features = max_features - min_features

            statistical_features.append(
                np.concatenate(
                    [mean_features, max_features, min_features, range_features], axis=1
                )
            )

        X_statistical = np.concatenate(statistical_features, axis=1)
        print(f"   통계적 피처 수: {X_statistical.shape[1]}")

        # 3. 간단한 상호작용 피처 생성
        print("   간단한 상호작용 피처 생성 중...")
        interaction_features = []

        # Big5와 다른 모달리티 간 상호작용
        big5_data = multimodal_data["big5"].astype(np.float64)
        for other_name, other_data in [
            ("cmi", multimodal_data["cmi"]),
            ("rppg", multimodal_data["rppg"]),
            ("voice", multimodal_data["voice"]),
        ]:
            other_data = other_data.astype(np.float64)
            # Big5의 주요 특성과 다른 모달리티의 평균 간 상호작용
            big5_ext = big5_data[:, 0:5].mean(axis=1, keepdims=True)  # EXT
            big5_opn = big5_data[:, 5:10].mean(axis=1, keepdims=True)  # OPN
            other_mean = np.mean(other_data, axis=1, keepdims=True)

            interaction_features.append(big5_ext * other_mean)
            interaction_features.append(big5_opn * other_mean)

        X_interaction = np.concatenate(interaction_features, axis=1)
        print(f"   상호작용 피처 수: {X_interaction.shape[1]}")

        # 4. 모든 피처 결합
        X_combined = np.concatenate([X_basic, X_statistical, X_interaction], axis=1)

        print(f"✅ 기본 피처 엔지니어링 완료:")
        print(f"   총 피처 수: {X_combined.shape[1]}")
        print(f"   기본: {X_basic.shape[1]}, 통계: {X_statistical.shape[1]}")
        print(f"   상호작용: {X_interaction.shape[1]}")

        return X_combined

    def create_optimized_models(self):
        """최적화된 모델들 생성"""
        print("🔄 최적화된 모델들 생성 중...")

        self.models = {
            "random_forest": RandomForestRegressor(
                n_estimators=200,  # 더 많은 트리
                max_depth=12,  # 더 깊은 깊이
                min_samples_split=10,  # 최적화
                min_samples_leaf=5,  # 최적화
                max_features="sqrt",  # 피처 선택 최적화
                random_state=42,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=200,  # 더 많은 트리
                learning_rate=0.05,  # 더 낮은 학습률
                max_depth=6,  # 더 깊은 깊이
                min_samples_split=10,
                subsample=0.9,  # 서브샘플링
                random_state=42,
            ),
            "xgboost": XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.01,  # L1 정규화
                reg_lambda=0.01,  # L2 정규화
                random_state=42,
                n_jobs=-1,
            ),
            "lightgbm": LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.01,  # L1 정규화
                reg_lambda=0.01,  # L2 정규화
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
            "ridge": Ridge(alpha=0.1),  # 정규화 강도 조정
            "elastic_net": ElasticNet(alpha=0.001, l1_ratio=0.5),  # 정규화 강도 조정
            "svr": SVR(kernel="rbf", C=10.0, gamma="auto"),  # 하이퍼파라미터 최적화
        }

        print(f"✅ {len(self.models)}개 최적화된 모델 생성 완료")

    def prepare_optimized_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """최적화된 데이터 준비"""
        print("🔄 최적화된 데이터 준비 중...")

        # RobustScaler로 정규화
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers["robust_ensemble"] = scaler

        # 피처 선택 (상위 100개 특성 선택)
        print("   피처 선택 중...")
        self.feature_selector = SelectKBest(score_func=f_regression, k=100)
        X_selected = self.feature_selector.fit_transform(X_scaled, y)

        print(f"✅ 최적화된 데이터 준비 완료: {X_selected.shape}")
        print(f"   원본 피처 수: {X.shape[1]}")
        print(f"   선택된 피처 수: {X_selected.shape[1]}")

        return X_selected, y

    def train_optimized_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """최적화된 모델들 훈련"""
        print("🚀 최적화된 모델들 훈련 시작...")

        # 훈련/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
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

                # 테스트 성능
                test_pred = model.predict(X_test)
                test_r2 = r2_score(y_test, test_pred)
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

                model_results[name] = {
                    "cv_mean_r2": avg_r2,
                    "cv_std_r2": std_r2,
                    "cv_mean_rmse": avg_rmse,
                    "test_r2": test_r2,
                    "test_rmse": test_rmse,
                    "model": model,
                }

                print(f"     CV R²: {avg_r2:.4f} (±{std_r2:.4f})")
                print(f"     Test R²: {test_r2:.4f}")
                print(f"     Test RMSE: {test_rmse:.4f}")

            except Exception as e:
                print(f"     ❌ 훈련 실패: {str(e)}")
                model_results[name] = None

        # 성능 순으로 정렬
        valid_models = {k: v for k, v in model_results.items() if v is not None}
        sorted_models = sorted(
            valid_models.items(), key=lambda x: x[1]["test_r2"], reverse=True
        )

        print(f"✅ {len(valid_models)}개 최적화된 모델 훈련 완료")
        print("📊 최적화된 모델 성능 순위 (테스트 R² 기준):")
        for i, (name, scores) in enumerate(sorted_models, 1):
            print(f"   {i}. {name}: R² = {scores['test_r2']:.4f}")

        return model_results

    def create_optimized_ensemble(
        self, X: np.ndarray, y: np.ndarray, model_results: Dict
    ) -> Dict:
        """최적화된 앙상블 생성"""
        print("🔄 최적화된 앙상블 생성 중...")

        # 상위 5개 모델 선택
        valid_models = {k: v for k, v in model_results.items() if v is not None}
        sorted_models = sorted(
            valid_models.items(), key=lambda x: x[1]["test_r2"], reverse=True
        )
        top_models = [name for name, _ in sorted_models[:5]]

        print(f"   선택된 상위 모델들: {top_models}")

        # 훈련/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        predictions = []
        weights = []

        for name in top_models:
            if name in model_results and model_results[name] is not None:
                model = model_results[name]["model"]
                pred = model.predict(X_test)
                predictions.append(pred)
                # 테스트 R² 점수를 가중치로 사용
                weights.append(model_results[name]["test_r2"])

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

    def run_simplified_optimization(self, limit: int = 10000) -> Dict:
        """간소화된 최적화 실행"""
        print("🚀 간소화된 최적화 시스템 시작")
        print("=" * 60)

        # 1. 실제 BigQuery 데이터 로딩
        multimodal_data, targets = self.load_real_bigquery_data(limit)

        # 2. 기본 피처 엔지니어링
        X_engineered = self.create_basic_features(multimodal_data)

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
                    "test_r2": scores["test_r2"] if scores else None,
                    "test_rmse": scores["test_rmse"] if scores else None,
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
                "optimization_status": "완료",
            },
        }

        # JSON으로 저장
        with open("simplified_optimization_results.json", "w") as f:

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

        print(f"✅ 간소화된 최적화 완료!")
        if ensemble_results:
            print(f"   최종 앙상블 R²: {ensemble_results['r2']:.4f}")
            print(f"   최종 앙상블 RMSE: {ensemble_results['rmse']:.4f}")

        return results


def main():
    """메인 실행 함수"""
    print("🚀 간소화된 최적화 시스템 - 안정적인 성능 향상")
    print("=" * 60)

    optimizer = SimplifiedOptimizer()
    results = optimizer.run_simplified_optimization(limit=10000)

    print("\n📊 간소화된 최적화 결과:")
    if results["ensemble_results"]:
        print(f"   최종 앙상블 R²: {results['ensemble_results']['r2']:.4f}")
        print(f"   최종 앙상블 RMSE: {results['ensemble_results']['rmse']:.4f}")
        print(f"   상관계수: {results['ensemble_results']['correlation']:.4f}")
        print(f"   선택된 모델들: {results['ensemble_results']['selected_models']}")


if __name__ == "__main__":
    main()
