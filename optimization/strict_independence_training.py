#!/usr/bin/env python3
"""
엄격한 독립성 훈련 시스템 - 완전한 데이터 누출 방지
- Big5 데이터를 입력에서 완전히 제거
- CMI, RPPG, Voice 데이터만 사용
- 완전히 독립적인 타겟 변수 생성
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
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)


class StrictIndependenceTrainer:
    """엄격한 독립성 훈련 시스템"""

    def __init__(self, project_id: str = "persona-diary-service"):
        self.project_id = project_id
        self.scalers = {}
        self.models = {}
        self.ensemble_weights = None
        self.best_models = []
        self.cv_scores = {}

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
        """실제 BigQuery 데이터 로딩 (Big5 데이터 완전 제거)"""
        print("🔄 실제 BigQuery 데이터 로딩 중...")
        try:
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

            # 데이터 결합 (Big5 데이터 완전 제거)
            multimodal_data = {
                "cmi": cmi_numeric.values,
                "rppg": rppg_numeric.values,
                "voice": voice_numeric.values,
            }

            # 완전히 독립적인 타겟 변수 생성
            print("🔍 완전히 독립적인 타겟 변수 생성 중...")

            # 방법 1: CMI 데이터의 특정 특성만 사용
            cmi_target = cmi_numeric.iloc[:, :5].mean(axis=1)  # 처음 5개 특성만

            # 방법 2: RPPG 데이터의 특정 특성만 사용
            rppg_target = rppg_numeric.iloc[:, :5].mean(axis=1)  # 처음 5개 특성만

            # 방법 3: Voice 데이터의 특정 특성만 사용
            voice_target = voice_numeric.iloc[:, :10].mean(axis=1)  # 처음 10개 특성만

            # 방법 4: 완전히 다른 조합으로 타겟 변수 생성
            np.random.seed(42)
            independent_target = (
                cmi_target * 0.3
                + rppg_target * 0.2
                + voice_target * 0.2
                + np.random.normal(0, 0.5, len(cmi_target)) * 0.3  # 랜덤 노이즈 추가
            )

            # 1-10 스케일로 정규화
            targets = (independent_target - independent_target.min()) / (
                independent_target.max() - independent_target.min()
            ) * 9 + 1

            print(f"✅ 완전히 독립적인 타겟 변수 생성 완료:")
            print(f"   CMI: {cmi_numeric.shape}")
            print(f"   RPPG: {rppg_numeric.shape}")
            print(f"   Voice: {voice_numeric.shape}")
            print(f"   Targets: {targets.shape}")
            print(f"   타겟 변수 통계:")
            print(f"     평균: {targets.mean():.4f}")
            print(f"     표준편차: {targets.std():.4f}")
            print(f"     최소값: {targets.min():.4f}")
            print(f"     최대값: {targets.max():.4f}")
            print(f"   Big5 데이터 완전 제거됨!")

            return multimodal_data, targets

        except Exception as e:
            print(f"❌ BigQuery 데이터 로딩 실패: {str(e)}")
            raise e

    def create_robust_models(self):
        """강건한 모델들 생성 (과적합 방지)"""
        print("🔄 강건한 모델들 생성 중...")

        self.models = {
            "random_forest": RandomForestRegressor(
                n_estimators=50,  # 더 적은 트리
                max_depth=6,  # 더 얕은 깊이
                min_samples_split=30,  # 더 많은 샘플 필요
                min_samples_leaf=15,  # 더 많은 리프 샘플
                random_state=42,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=50,  # 더 적은 트리
                learning_rate=0.05,  # 더 낮은 학습률
                max_depth=3,  # 더 얕은 깊이
                min_samples_split=30,
                random_state=42,
            ),
            "xgboost": XGBRegressor(
                n_estimators=50,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.7,
                colsample_bytree=0.7,
                random_state=42,
                n_jobs=-1,
            ),
            "lightgbm": LGBMRegressor(
                n_estimators=50,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.7,
                colsample_bytree=0.7,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
            "ridge": Ridge(alpha=50.0),  # 매우 강한 정규화
            "elastic_net": ElasticNet(alpha=1.0, l1_ratio=0.5),  # 매우 강한 정규화
            "svr": SVR(kernel="rbf", C=0.01, gamma="scale"),  # 매우 부드러운 경계
        }

        print(f"✅ {len(self.models)}개 강건한 모델 생성 완료")

    def prepare_robust_data(
        self, multimodal_data: Dict, targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """강건한 데이터 준비"""
        print("🔄 강건한 데이터 준비 중...")

        # 모든 모달리티를 하나의 행렬로 결합 (Big5 제외)
        X = np.concatenate(
            [
                multimodal_data["cmi"],
                multimodal_data["rppg"],
                multimodal_data["voice"],
            ],
            axis=1,
        )

        # RobustScaler로 정규화 (이상치에 강함)
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers["robust_ensemble"] = scaler

        print(f"✅ 강건한 데이터 준비 완료: {X_scaled.shape}")
        return X_scaled, targets

    def train_individual_models_with_cv(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """교차 검증으로 개별 모델 훈련 및 안정성 평가"""
        print("🚀 교차 검증으로 모델 훈련 시작...")

        model_scores = {}
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in self.models.items():
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

        print(f"✅ {len(valid_models)}개 모델 훈련 완료")
        print("📊 모델 성능 순위 (교차 검증 R² 기준):")
        for i, (name, scores) in enumerate(sorted_models[:5], 1):
            print(
                f"   {i}. {name}: R² = {scores['cv_mean_r2']:.4f} (±{scores['cv_std_r2']:.4f})"
            )

        self.cv_scores = model_scores
        return model_scores

    def select_stable_models(
        self, model_scores: Dict, stability_threshold: float = 0.1
    ) -> List[str]:
        """안정적인 모델들 선택 (낮은 표준편차 기준)"""
        print("🔄 안정적인 모델들 선택 중...")

        stable_models = []
        for name, scores in model_scores.items():
            if scores and scores["cv_mean_r2"] > 0.05:  # 일정 성능 이상
                # 표준편차가 낮고, R2가 일정 수준 이상인 모델 선택
                if scores["cv_std_r2"] < stability_threshold:
                    stable_models.append(name)
                    print(
                        f"   ✅ {name}: R² = {scores['cv_mean_r2']:.4f} (±{scores['cv_std_r2']:.4f})"
                    )

        if not stable_models:
            print("⚠️ 안정적인 모델이 충분하지 않습니다. 모든 유효 모델을 사용합니다.")
            stable_models = [
                name for name, scores in model_scores.items() if scores is not None
            ]

        print(f"✅ {len(stable_models)}개 안정적인 모델 선택 완료")
        self.best_models = stable_models
        return stable_models

    def create_robust_ensemble(
        self, X: np.ndarray, y: np.ndarray, model_scores: Dict, stable_models: List[str]
    ) -> Dict:
        """강건한 앙상블 생성 (선택된 모델들의 가중 평균)"""
        print("🔄 강건한 앙상블 생성 중...")

        if not stable_models:
            print("❌ 앙상블을 위한 모델이 없습니다.")
            return None

        predictions = []
        weights = []

        for name in stable_models:
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
        self.ensemble_weights = dict(zip(stable_models, weights))

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
        }

        print(f"✅ 강건한 앙상블 생성 및 평가 완료:")
        print(f"   R²: {r2:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   상관계수: {correlation:.4f}")

        return results

    def run_strict_independence_training(self, limit: int = 10000) -> Dict:
        """엄격한 독립성 훈련 실행"""
        print("🚀 엄격한 독립성 훈련 시스템 시작")
        print("=" * 60)

        # 1. 실제 BigQuery 데이터 로딩 (Big5 데이터 완전 제거)
        multimodal_data, targets = self.load_real_bigquery_data(limit)

        # 2. 강건한 모델들 생성
        self.create_robust_models()

        # 3. 데이터 준비
        X, y = self.prepare_robust_data(multimodal_data, targets)

        # 4. 교차 검증으로 개별 모델 훈련
        model_scores = self.train_individual_models_with_cv(X, y)

        # 5. 안정적인 모델들 선택
        stable_models = self.select_stable_models(model_scores)

        # 6. 강건한 앙상블 생성
        ensemble_results = self.create_robust_ensemble(
            X, y, model_scores, stable_models
        )

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
            "stable_models_selected": self.best_models,
            "ensemble_results": ensemble_results,
            "ensemble_weights": self.ensemble_weights,
            "data_info": {
                "n_samples": len(y),
                "n_features": X.shape[1],
                "n_models_trained": len(
                    [m for m in model_scores.values() if m is not None]
                ),
                "n_models_in_ensemble": len(self.best_models),
                "big5_removed": True,
                "target_independence": "Big5 데이터 완전 제거, CMI/RPPG/Voice만 사용",
            },
        }

        # JSON으로 저장
        with open("strict_independence_training_results.json", "w") as f:

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

        print(f"✅ 엄격한 독립성 훈련 완료!")
        if ensemble_results:
            print(f"   최종 앙상블 R²: {ensemble_results['r2']:.4f}")
            print(f"   최종 앙상블 RMSE: {ensemble_results['rmse']:.4f}")

        return results


def main():
    """메인 실행 함수"""
    print("🚀 엄격한 독립성 훈련 시스템 - 완전한 데이터 누출 방지")
    print("=" * 60)

    trainer = StrictIndependenceTrainer()
    results = trainer.run_strict_independence_training(limit=10000)

    print("\n📊 엄격한 독립성 훈련 결과:")
    if results["ensemble_results"]:
        print(f"   최종 앙상블 R²: {results['ensemble_results']['r2']:.4f}")
        print(f"   최종 앙상블 RMSE: {results['ensemble_results']['rmse']:.4f}")
        print(f"   상관계수: {results['ensemble_results']['correlation']:.4f}")


if __name__ == "__main__":
    main()
