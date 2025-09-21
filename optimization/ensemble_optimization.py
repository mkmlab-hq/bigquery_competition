#!/usr/bin/env python3
"""
앙상블 최적화 시스템 - R² 0.70+ 도전
- 현재 모델 기반 앙상블 구현
- 다양한 알고리즘 조합
- 성능 향상 목표: R² 0.70+
"""

import json
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)


class EnsembleOptimizer:
    """앙상블 최적화 시스템"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.ensemble_weights = None
        self.best_ensemble = None

    def create_diverse_models(self):
        """다양한 모델 생성"""
        print("🔄 다양한 모델 생성 중...")

        self.models = {
            # 트리 기반 모델들
            "random_forest": RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                min_samples_split=5,
                random_state=42,
            ),
            "xgboost": XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
            ),
            "lightgbm": LGBMRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            ),
            # 선형 모델들
            "ridge": Ridge(alpha=1.0),
            "elastic_net": ElasticNet(alpha=0.1, l1_ratio=0.5),
            # 서포트 벡터 머신
            "svr": SVR(kernel="rbf", C=1.0, gamma="scale"),
        }

        print(f"✅ {len(self.models)}개 모델 생성 완료")

    def prepare_ensemble_data(
        self, multimodal_data: Dict, targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """앙상블용 데이터 준비"""
        print("🔄 앙상블 데이터 준비 중...")

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
        self.scalers["ensemble"] = scaler

        print(f"✅ 데이터 준비 완료: {X_scaled.shape}")
        return X_scaled, targets

    def train_individual_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """개별 모델 훈련"""
        print("🚀 개별 모델 훈련 시작...")

        model_scores = {}

        for name, model in self.models.items():
            print(f"   훈련 중: {name}")

            try:
                # 모델 훈련
                model.fit(X, y)

                # 교차 검증 점수
                cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
                avg_score = cv_scores.mean()
                std_score = cv_scores.std()

                model_scores[name] = {
                    "cv_mean": avg_score,
                    "cv_std": std_score,
                    "model": model,
                }

                print(f"     R²: {avg_score:.4f} (±{std_score:.4f})")

            except Exception as e:
                print(f"     ❌ 훈련 실패: {str(e)}")
                model_scores[name] = None

        # 성능 순으로 정렬
        valid_models = {k: v for k, v in model_scores.items() if v is not None}
        sorted_models = sorted(
            valid_models.items(), key=lambda x: x[1]["cv_mean"], reverse=True
        )

        print(f"✅ {len(valid_models)}개 모델 훈련 완료")
        print("📊 모델 성능 순위:")
        for i, (name, scores) in enumerate(sorted_models[:5], 1):
            print(f"   {i}. {name}: R² = {scores['cv_mean']:.4f}")

        return model_scores

    def optimize_ensemble_weights(
        self, X: np.ndarray, y: np.ndarray, model_scores: Dict
    ) -> np.ndarray:
        """앙상블 가중치 최적화"""
        print("🔄 앙상블 가중치 최적화 중...")

        # 유효한 모델들만 선택
        valid_models = {k: v for k, v in model_scores.items() if v is not None}

        if len(valid_models) < 2:
            print("❌ 앙상블을 위한 충분한 모델이 없습니다.")
            return None

        # 각 모델의 예측값 생성
        predictions = {}
        for name, scores in valid_models.items():
            model = scores["model"]
            pred = model.predict(X)
            predictions[name] = pred

        # 가중치 최적화 (간단한 방법: 성능 기반 가중치)
        weights = []
        for name in valid_models.keys():
            score = valid_models[name]["cv_mean"]
            weights.append(score)

        # 정규화
        weights = np.array(weights)
        weights = weights / weights.sum()

        self.ensemble_weights = dict(zip(valid_models.keys(), weights))

        print("✅ 앙상블 가중치 최적화 완료:")
        for name, weight in self.ensemble_weights.items():
            print(f"   {name}: {weight:.4f}")

        return weights

    def create_ensemble_predictions(
        self, X: np.ndarray, model_scores: Dict
    ) -> np.ndarray:
        """앙상블 예측 생성"""
        if self.ensemble_weights is None:
            print("❌ 앙상블 가중치가 없습니다.")
            return None

        # 각 모델의 예측값 생성
        predictions = []
        weights = []

        for name, weight in self.ensemble_weights.items():
            if name in model_scores and model_scores[name] is not None:
                model = model_scores[name]["model"]
                pred = model.predict(X)
                predictions.append(pred)
                weights.append(weight)

        if not predictions:
            print("❌ 유효한 예측값이 없습니다.")
            return None

        # 가중 평균으로 앙상블 예측
        predictions = np.array(predictions)
        weights = np.array(weights)

        ensemble_pred = np.average(predictions, axis=0, weights=weights)

        return ensemble_pred

    def evaluate_ensemble(
        self, X: np.ndarray, y: np.ndarray, model_scores: Dict
    ) -> Dict:
        """앙상블 성능 평가"""
        print("📊 앙상블 성능 평가 중...")

        # 앙상블 예측
        ensemble_pred = self.create_ensemble_predictions(X, model_scores)

        if ensemble_pred is None:
            return None

        # 성능 메트릭 계산
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

        print(f"✅ 앙상블 평가 완료:")
        print(f"   R²: {r2:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   상관계수: {correlation:.4f}")

        return results

    def run_ensemble_optimization(
        self, multimodal_data: Dict, targets: np.ndarray
    ) -> Dict:
        """앙상블 최적화 실행"""
        print("🚀 앙상블 최적화 시스템 시작")
        print("=" * 60)

        # 1. 다양한 모델 생성
        self.create_diverse_models()

        # 2. 데이터 준비
        X, y = self.prepare_ensemble_data(multimodal_data, targets)

        # 3. 개별 모델 훈련
        model_scores = self.train_individual_models(X, y)

        # 4. 앙상블 가중치 최적화
        weights = self.optimize_ensemble_weights(X, y, model_scores)

        # 5. 앙상블 성능 평가
        ensemble_results = self.evaluate_ensemble(X, y, model_scores)

        # 6. 결과 저장
        results = {
            "individual_models": {
                name: {
                    "cv_mean": scores["cv_mean"] if scores else None,
                    "cv_std": scores["cv_std"] if scores else None,
                }
                for name, scores in model_scores.items()
            },
            "ensemble_results": ensemble_results,
            "ensemble_weights": self.ensemble_weights,
            "data_info": {
                "n_samples": len(y),
                "n_features": X.shape[1],
                "n_models": len([m for m in model_scores.values() if m is not None]),
            },
        }

        # JSON으로 저장
        with open("ensemble_optimization_results.json", "w") as f:

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
                elif isinstance(obj, dict):
                    return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_json_serializable(item) for item in obj]
                else:
                    return obj

            json_results = convert_to_json_serializable(results)
            json.dump(json_results, f, indent=2)

        print(f"✅ 앙상블 최적화 완료!")
        if ensemble_results:
            print(f"   최종 R²: {ensemble_results['r2']:.4f}")
            print(f"   최종 RMSE: {ensemble_results['rmse']:.4f}")

        return results


def main():
    """메인 실행 함수"""
    print("🚀 앙상블 최적화 시스템 - R² 0.70+ 도전")
    print("=" * 60)

    # 기존 데이터 로더 사용
    from advanced_multimodal_training import BigQueryDataLoader

    # 데이터 로딩
    data_loader = BigQueryDataLoader()
    multimodal_data, targets = data_loader.load_competition_data(10000)

    # 앙상블 최적화 실행
    optimizer = EnsembleOptimizer()
    results = optimizer.run_ensemble_optimization(multimodal_data, targets)

    print("\n📊 앙상블 최적화 결과:")
    if results["ensemble_results"]:
        print(f"   앙상블 R²: {results['ensemble_results']['r2']:.4f}")
        print(f"   앙상블 RMSE: {results['ensemble_results']['rmse']:.4f}")
        print(f"   상관계수: {results['ensemble_results']['correlation']:.4f}")


if __name__ == "__main__":
    main()
