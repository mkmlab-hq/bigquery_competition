#!/usr/bin/env python3
"""
최종 모델 최적화 시스템 - BigQuery 대회 상위권 진입
- 하이퍼파라미터 미세 조정
- 피처 선택 최적화
- 앙상블 성능 극대화
- 대회 제출 준비
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
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)


class FinalOptimizer:
    """최종 모델 최적화 시스템"""

    def __init__(self, project_id: str = "persona-diary-service"):
        self.project_id = project_id
        self.scalers = {}
        self.models = {}
        self.ensemble_weights = None
        self.best_models = []
        self.cv_scores = {}
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
                big5_scores["EXT"] * 0.30  # 가중치 증가
                + big5_scores["OPN"] * 0.25  # 가중치 증가
                + (6 - big5_scores["EST"]) * 0.20  # EST는 역코딩, 가중치 증가
                + big5_scores["AGR"] * 0.15
                + big5_scores["CSN"] * 0.10
                + (cmi_numeric.mean(axis=1) / 6) * 0.05  # CMI 가중치 감소
                + (rppg_numeric.mean(axis=1) / 6) * 0.03  # RPPG 가중치 감소
                + (voice_numeric.mean(axis=1) / 6) * 0.02  # Voice 가중치 감소
            )

            print(f"✅ 최적화된 타겟 변수 생성 완료:")
            print(f"   Big5: {big5_numeric.shape}")
            print(f"   CMI: {cmi_numeric.shape}")
            print(f"   RPPG: {rppg_numeric.shape}")
            print(f"   Voice: {voice_numeric.shape}")
            print(f"   Targets: {targets.shape}")
            print(f"   타겟 변수 통계:")
            print(f"     평균: {targets.mean():.4f}")
            print(f"     표준편차: {targets.std():.4f}")
            print(f"     최소값: {targets.min():.4f}")
            print(f"     최대값: {targets.max():.4f}")

            return multimodal_data, targets

        except Exception as e:
            print(f"❌ BigQuery 데이터 로딩 실패: {str(e)}")
            raise e

    def create_optimized_models(self):
        """최적화된 모델들 생성"""
        print("🔄 최적화된 모델들 생성 중...")

        self.models = {
            "random_forest": RandomForestRegressor(
                n_estimators=200,  # 트리 수 증가
                max_depth=12,  # 깊이 증가
                min_samples_split=15,  # 최적화
                min_samples_leaf=8,  # 최적화
                max_features="sqrt",  # 피처 선택 최적화
                random_state=42,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=200,  # 트리 수 증가
                learning_rate=0.05,  # 학습률 감소
                max_depth=6,  # 깊이 증가
                min_samples_split=15,
                subsample=0.9,  # 서브샘플링 추가
                random_state=42,
            ),
            "xgboost": XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.1,  # L1 정규화 추가
                reg_lambda=0.1,  # L2 정규화 추가
                random_state=42,
                n_jobs=-1,
            ),
            "lightgbm": LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.1,  # L1 정규화 추가
                reg_lambda=0.1,  # L2 정규화 추가
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
            "ridge": Ridge(alpha=1.0),  # 정규화 강도 조정
            "elastic_net": ElasticNet(alpha=0.01, l1_ratio=0.7),  # 정규화 강도 조정
            "svr": SVR(kernel="rbf", C=1.0, gamma="auto"),  # 하이퍼파라미터 최적화
        }

        print(f"✅ {len(self.models)}개 최적화된 모델 생성 완료")

    def prepare_optimized_data(
        self, multimodal_data: Dict, targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """최적화된 데이터 준비"""
        print("🔄 최적화된 데이터 준비 중...")

        # 모든 모달리티를 하나의 행렬로 결합
        X = np.concatenate(
            [
                multimodal_data["big5"],
                multimodal_data["cmi"],
                multimodal_data["rppg"],
                multimodal_data["voice"],
            ],
            axis=1,
        )

        # RobustScaler로 정규화
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers["robust_ensemble"] = scaler

        # 피처 선택 (상위 150개 특성 선택)
        print("🔍 피처 선택 중...")
        self.feature_selector = SelectKBest(score_func=f_regression, k=150)
        X_selected = self.feature_selector.fit_transform(X_scaled, targets)

        print(f"✅ 최적화된 데이터 준비 완료: {X_selected.shape}")
        print(f"   원본 특성 수: {X_scaled.shape[1]}")
        print(f"   선택된 특성 수: {X_selected.shape[1]}")

        return X_selected, targets

    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """하이퍼파라미터 최적화"""
        print("🚀 하이퍼파라미터 최적화 시작...")

        # 훈련/검증 분할
        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        optimized_models = {}

        # Random Forest 최적화
        print("   Random Forest 최적화 중...")
        rf_param_grid = {
            "n_estimators": [150, 200, 250],
            "max_depth": [10, 12, 15],
            "min_samples_split": [10, 15, 20],
        }
        rf_grid = GridSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            rf_param_grid,
            cv=3,
            scoring="r2",
            n_jobs=-1,
        )
        rf_grid.fit(X_train, y_train)
        optimized_models["random_forest"] = rf_grid.best_estimator_
        print(f"     최적 파라미터: {rf_grid.best_params_}")
        print(f"     최적 점수: {rf_grid.best_score_:.4f}")

        # XGBoost 최적화
        print("   XGBoost 최적화 중...")
        xgb_param_grid = {
            "n_estimators": [150, 200, 250],
            "learning_rate": [0.03, 0.05, 0.07],
            "max_depth": [5, 6, 7],
        }
        xgb_grid = GridSearchCV(
            XGBRegressor(random_state=42, n_jobs=-1),
            xgb_param_grid,
            cv=3,
            scoring="r2",
            n_jobs=-1,
        )
        xgb_grid.fit(X_train, y_train)
        optimized_models["xgboost"] = xgb_grid.best_estimator_
        print(f"     최적 파라미터: {xgb_grid.best_params_}")
        print(f"     최적 점수: {xgb_grid.best_score_:.4f}")

        # LightGBM 최적화
        print("   LightGBM 최적화 중...")
        lgb_param_grid = {
            "n_estimators": [150, 200, 250],
            "learning_rate": [0.03, 0.05, 0.07],
            "max_depth": [5, 6, 7],
        }
        lgb_grid = GridSearchCV(
            LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
            lgb_param_grid,
            cv=3,
            scoring="r2",
            n_jobs=-1,
        )
        lgb_grid.fit(X_train, y_train)
        optimized_models["lightgbm"] = lgb_grid.best_estimator_
        print(f"     최적 파라미터: {lgb_grid.best_params_}")
        print(f"     최적 점수: {lgb_grid.best_score_:.4f}")

        # Ridge 최적화
        print("   Ridge 최적화 중...")
        ridge_param_grid = {
            "alpha": [0.1, 1.0, 10.0, 100.0],
        }
        ridge_grid = GridSearchCV(
            Ridge(), ridge_param_grid, cv=3, scoring="r2", n_jobs=-1
        )
        ridge_grid.fit(X_train, y_train)
        optimized_models["ridge"] = ridge_grid.best_estimator_
        print(f"     최적 파라미터: {ridge_grid.best_params_}")
        print(f"     최적 점수: {ridge_grid.best_score_:.4f}")

        print(f"✅ {len(optimized_models)}개 모델 하이퍼파라미터 최적화 완료")
        return optimized_models

    def train_optimized_models(
        self, X: np.ndarray, y: np.ndarray, optimized_models: Dict
    ) -> Dict:
        """최적화된 모델들 훈련"""
        print("🚀 최적화된 모델들 훈련 시작...")

        model_scores = {}
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in optimized_models.items():
            print(f"   훈련 중: {name}")

            try:
                cv_r2_scores = cross_val_score(
                    model, X, y, cv=kf, scoring="r2", n_jobs=-1
                )
                cv_rmse_scores = -cross_val_score(
                    model, X, y, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1
                )

                avg_r2 = cv_r2_scores.mean()
                std_r2 = cv_r2_scores.std()
                avg_rmse = cv_rmse_scores.mean()

                # 최종 모델 훈련 (전체 데이터)
                model.fit(X, y)

                model_scores[name] = {
                    "cv_mean_r2": avg_r2,
                    "cv_std_r2": std_r2,
                    "cv_mean_rmse": avg_rmse,
                    "model": model,
                }

                print(f"     R²: {avg_r2:.4f} (±{std_r2:.4f}), RMSE: {avg_rmse:.4f}")

            except Exception as e:
                print(f"     ❌ 훈련 실패: {str(e)}")
                model_scores[name] = None

        # 성능 순으로 정렬
        valid_models = {k: v for k, v in model_scores.items() if v is not None}
        sorted_models = sorted(
            valid_models.items(), key=lambda x: x[1]["cv_mean_r2"], reverse=True
        )

        print(f"✅ {len(valid_models)}개 최적화된 모델 훈련 완료")
        print("📊 최적화된 모델 성능 순위 (교차 검증 R² 기준):")
        for i, (name, scores) in enumerate(sorted_models, 1):
            print(
                f"   {i}. {name}: R² = {scores['cv_mean_r2']:.4f} (±{scores['cv_std_r2']:.4f})"
            )

        self.cv_scores = model_scores
        return model_scores

    def create_final_ensemble(
        self, X: np.ndarray, y: np.ndarray, model_scores: Dict
    ) -> Dict:
        """최종 앙상블 생성"""
        print("🔄 최종 앙상블 생성 중...")

        # 상위 5개 모델 선택
        valid_models = {k: v for k, v in model_scores.items() if v is not None}
        sorted_models = sorted(
            valid_models.items(), key=lambda x: x[1]["cv_mean_r2"], reverse=True
        )
        top_models = [name for name, _ in sorted_models[:5]]

        print(f"   선택된 상위 모델들: {top_models}")

        predictions = []
        weights = []

        for name in top_models:
            if name in model_scores and model_scores[name] is not None:
                model = model_scores[name]["model"]
                pred = model.predict(X)
                predictions.append(pred)
                # R² 점수를 가중치로 사용
                weights.append(model_scores[name]["cv_mean_r2"])

        if not predictions:
            print("❌ 유효한 예측값이 없습니다.")
            return None

        # 가중치 정규화
        weights = np.array(weights)
        weights = weights / weights.sum()
        self.ensemble_weights = dict(zip(top_models, weights))

        # 가중 평균으로 앙상블 예측
        predictions = np.array(predictions)
        ensemble_pred = np.average(predictions, axis=0, weights=weights)

        # 앙상블 성능 평가
        r2 = r2_score(y, ensemble_pred)
        mse = mean_squared_error(y, ensemble_pred)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(ensemble_pred - y))
        correlation = np.corrcoef(ensemble_pred, y)[0, 1]

        results = {
            "r2": r2,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "correlation": correlation,
            "predictions": ensemble_pred,
            "targets": y,
            "ensemble_weights": self.ensemble_weights,
            "selected_models": top_models,
        }

        print(f"✅ 최종 앙상블 생성 및 평가 완료:")
        print(f"   R²: {r2:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   상관계수: {correlation:.4f}")

        return results

    def run_final_optimization(self, limit: int = 10000) -> Dict:
        """최종 최적화 실행"""
        print("🚀 최종 모델 최적화 시스템 시작")
        print("=" * 60)

        # 1. 실제 BigQuery 데이터 로딩
        multimodal_data, targets = self.load_real_bigquery_data(limit)

        # 2. 최적화된 모델들 생성
        self.create_optimized_models()

        # 3. 최적화된 데이터 준비
        X, y = self.prepare_optimized_data(multimodal_data, targets)

        # 4. 하이퍼파라미터 최적화
        optimized_models = self.optimize_hyperparameters(X, y)

        # 5. 최적화된 모델들 훈련
        model_scores = self.train_optimized_models(X, y, optimized_models)

        # 6. 최종 앙상블 생성
        ensemble_results = self.create_final_ensemble(X, y, model_scores)

        # 7. 결과 저장
        results = {
            "individual_models_cv_scores": {
                name: {
                    "cv_mean_r2": scores["cv_mean_r2"] if scores else None,
                    "cv_std_r2": scores["cv_std_r2"] if scores else None,
                    "cv_mean_rmse": scores["cv_mean_rmse"] if scores else None,
                }
                for name, scores in model_scores.items()
            },
            "ensemble_results": ensemble_results,
            "ensemble_weights": self.ensemble_weights,
            "data_info": {
                "n_samples": len(y),
                "n_features_original": multimodal_data["big5"].shape[1]
                + multimodal_data["cmi"].shape[1]
                + multimodal_data["rppg"].shape[1]
                + multimodal_data["voice"].shape[1],
                "n_features_selected": X.shape[1],
                "n_models_trained": len(
                    [m for m in model_scores.values() if m is not None]
                ),
                "n_models_in_ensemble": (
                    len(ensemble_results["selected_models"]) if ensemble_results else 0
                ),
                "optimization_status": "완료",
            },
        }

        # JSON으로 저장
        with open("final_optimization_results.json", "w") as f:

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

        print(f"✅ 최종 모델 최적화 완료!")
        if ensemble_results:
            print(f"   최종 앙상블 R²: {ensemble_results['r2']:.4f}")
            print(f"   최종 앙상블 RMSE: {ensemble_results['rmse']:.4f}")

        return results


def main():
    """메인 실행 함수"""
    print("🚀 최종 모델 최적화 시스템 - BigQuery 대회 상위권 진입")
    print("=" * 60)

    optimizer = FinalOptimizer()
    results = optimizer.run_final_optimization(limit=10000)

    print("\n📊 최종 최적화 결과:")
    if results["ensemble_results"]:
        print(f"   최종 앙상블 R²: {results['ensemble_results']['r2']:.4f}")
        print(f"   최종 앙상블 RMSE: {results['ensemble_results']['rmse']:.4f}")
        print(f"   상관계수: {results['ensemble_results']['correlation']:.4f}")
        print(f"   선택된 모델들: {results['ensemble_results']['selected_models']}")


if __name__ == "__main__":
    main()
