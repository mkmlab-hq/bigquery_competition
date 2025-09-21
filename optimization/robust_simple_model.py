import json
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class RobustSimpleModel:
    """견고한 단순 모델"""

    def __init__(self, device="cpu"):
        self.device = device
        self.model = None
        self.scalers = {}
        self.model_type = None

    def train_robust_model(
        self, X_train: Dict, y_train: np.ndarray, X_val: Dict, y_val: np.ndarray
    ):
        """견고한 모델 훈련"""

        print("🚀 견고한 단순 모델 훈련 시작!")

        # 데이터 정규화
        for modality in ["big5", "cmi", "rppg", "voice"]:
            scaler = StandardScaler()
            X_train[modality] = scaler.fit_transform(X_train[modality])
            X_val[modality] = scaler.transform(X_val[modality])
            self.scalers[modality] = scaler

        # 모든 모달리티 결합
        X_train_combined = np.concatenate(
            [X_train["big5"], X_train["cmi"], X_train["rppg"], X_train["voice"]], axis=1
        )
        X_val_combined = np.concatenate(
            [X_val["big5"], X_val["cmi"], X_val["rppg"], X_val["voice"]], axis=1
        )

        # 단순하고 견고한 모델들 테스트
        models_to_test = {
            "linear_regression": LinearRegression(),
            "ridge_weak": Ridge(alpha=0.01),
            "ridge_medium": Ridge(alpha=0.1),
            "ridge_strong": Ridge(alpha=1.0),
            "ridge_very_strong": Ridge(alpha=10.0),
            "lasso_weak": Lasso(alpha=0.01, max_iter=1000),
            "lasso_medium": Lasso(alpha=0.1, max_iter=1000),
            "random_forest_simple": RandomForestRegressor(
                n_estimators=50, max_depth=5, random_state=42
            ),
            "random_forest_medium": RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            ),
        }

        best_r2 = -float("inf")
        best_model = None
        best_name = None

        for name, model in models_to_test.items():
            print(f"📚 {name} 모델 테스트 중...")

            try:
                # 모델 훈련
                model.fit(X_train_combined, y_train)

                # 검증 예측
                val_predictions = model.predict(X_val_combined)
                r2 = r2_score(y_val, val_predictions)

                print(f"  {name} R²: {r2:.4f}")

                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model
                    best_name = name

            except Exception as e:
                print(f"  {name} 실패: {str(e)}")
                continue

        self.model = best_model
        self.model_type = best_name

        print(f"✅ 최고 성능 모델: {best_name} (R²: {best_r2:.4f})")

    def predict(self, X: Dict) -> np.ndarray:
        """예측"""
        X_combined = np.concatenate(
            [X["big5"], X["cmi"], X["rppg"], X["voice"]], axis=1
        )

        return self.model.predict(X_combined)


def test_robust_simple_model():
    """견고한 단순 모델 테스트"""

    print("🧪 견고한 단순 모델 테스트 시작!")

    # 데이터 생성 (실제 데이터 시뮬레이션)
    np.random.seed(42)
    n_samples = 5000

    big5_data = np.random.beta(2, 2, (n_samples, 5))
    cmi_data = np.random.beta(1.5, 1.5, (n_samples, 20))
    rppg_data = np.random.normal(0, 1, (n_samples, 10))
    voice_data = np.random.normal(0, 1, (n_samples, 20))

    # 타겟 생성
    targets = (
        0.25 * big5_data[:, 0]
        + 0.20 * big5_data[:, 1]
        + 0.15 * big5_data[:, 2]
        + 0.20 * big5_data[:, 3]
        + 0.10 * big5_data[:, 4]
        + 0.05 * np.mean(cmi_data, axis=1)
        + 0.03 * np.mean(rppg_data, axis=1)
        + 0.02 * np.mean(voice_data, axis=1)
        + np.random.normal(0, 0.15, n_samples)
    )

    targets = (targets - targets.min()) / (targets.max() - targets.min())

    # 데이터 분할
    train_idx, test_idx = train_test_split(
        range(len(targets)), test_size=0.3, random_state=42
    )
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)

    X_train = {
        "big5": big5_data[train_idx],
        "cmi": cmi_data[train_idx],
        "rppg": rppg_data[train_idx],
        "voice": voice_data[train_idx],
    }
    X_val = {
        "big5": big5_data[val_idx],
        "cmi": cmi_data[val_idx],
        "rppg": rppg_data[val_idx],
        "voice": voice_data[val_idx],
    }
    X_test = {
        "big5": big5_data[test_idx],
        "cmi": cmi_data[test_idx],
        "rppg": rppg_data[test_idx],
        "voice": voice_data[test_idx],
    }

    y_train = targets[train_idx]
    y_val = targets[val_idx]
    y_test = targets[test_idx]

    print(f"훈련 데이터: {len(y_train)}개")
    print(f"검증 데이터: {len(y_val)}개")
    print(f"테스트 데이터: {len(y_test)}개")

    # 견고한 모델 생성 및 훈련
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    robust_model = RobustSimpleModel(device)
    robust_model.train_robust_model(X_train, y_train, X_val, y_val)

    # 테스트 예측
    test_predictions = robust_model.predict(X_test)

    # 성능 평가
    test_r2 = r2_score(y_test, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    test_mae = mean_absolute_error(y_test, test_predictions)

    print("\n📊 견고한 단순 모델 테스트 결과:")
    print(f"  모델 타입: {robust_model.model_type}")
    print(f"  테스트 R²: {test_r2:.4f}")
    print(f"  테스트 RMSE: {test_rmse:.4f}")
    print(f"  테스트 MAE: {test_mae:.4f}")

    # 결과 저장
    results = {
        "robust_simple_model_test": {
            "model_type": robust_model.model_type,
            "test_r2": test_r2,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
        }
    }

    with open("robust_simple_model_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("✅ 결과 저장 완료!")
    print("📁 robust_simple_model_test_results.json")

    return results


if __name__ == "__main__":
    results = test_robust_simple_model()


