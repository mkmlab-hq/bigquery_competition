import json
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class BestSingleModel:
    """최고 성능 단일 모델"""

    def __init__(self, device="cpu"):
        self.device = device
        self.model = None
        self.scalers = {}
        self.model_type = None

    def train_best_model(
        self, X_train: Dict, y_train: np.ndarray, X_val: Dict, y_val: np.ndarray
    ):
        """최고 성능 모델 훈련"""

        print("🚀 최고 성능 모델 훈련 시작!")

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

        # 여러 모델 테스트
        models_to_test = {
            "neural_network": self._create_neural_network(),
            "random_forest": RandomForestRegressor(n_estimators=200, random_state=42),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=200, random_state=42
            ),
            "ridge": Ridge(alpha=1.0),
            "ridge_optimized": Ridge(alpha=0.1),
            "ridge_strong": Ridge(alpha=10.0),
        }

        best_r2 = -float("inf")
        best_model = None
        best_name = None

        for name, model in models_to_test.items():
            print(f"📚 {name} 모델 테스트 중...")

            if name == "neural_network":
                # 신경망 모델 훈련
                r2 = self._train_neural_network(model, X_train, y_train, X_val, y_val)
            else:
                # Scikit-learn 모델 훈련
                model.fit(X_train_combined, y_train)
                val_predictions = model.predict(X_val_combined)
                r2 = r2_score(y_val, val_predictions)

            print(f"  {name} R²: {r2:.4f}")

            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_name = name

        self.model = best_model
        self.model_type = best_name

        print(f"✅ 최고 성능 모델: {best_name} (R²: {best_r2:.4f})")

    def _create_neural_network(self):
        """신경망 모델 생성"""
        from transfer_learning_multimodal import TransferLearningMultimodalNet

        return TransferLearningMultimodalNet(
            hidden_dim=256, dropout_rate=0.3, use_pretrained=False
        )

    def _train_neural_network(
        self, model, X_train: Dict, y_train: np.ndarray, X_val: Dict, y_val: np.ndarray
    ):
        """신경망 모델 훈련"""
        model.to(self.device)

        # 옵티마이저 및 손실 함수
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.MSELoss()

        # 데이터셋 생성
        from torch.utils.data import DataLoader

        from transfer_learning_multimodal import TransferLearningMultimodalDataset

        train_dataset = TransferLearningMultimodalDataset(
            X_train["big5"],
            X_train["cmi"],
            X_train["rppg"],
            X_train["voice"],
            y_train,
            augment=False,
        )
        val_dataset = TransferLearningMultimodalDataset(
            X_val["big5"],
            X_val["cmi"],
            X_val["rppg"],
            X_val["voice"],
            y_val,
            augment=False,
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        # 훈련
        best_val_r2 = -float("inf")
        patience = 5
        patience_counter = 0

        for epoch in range(30):
            # 훈련
            model.train()
            train_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()

                big5 = batch["big5"].to(self.device)
                cmi = batch["cmi"].to(self.device)
                rppg = batch["rppg"].to(self.device)
                voice = batch["voice"].to(self.device)
                targets = batch["target"].to(self.device)

                predictions = model(big5, cmi, rppg, voice)
                loss = criterion(predictions, targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()

            # 검증
            model.eval()
            val_predictions = []
            val_targets = []

            with torch.no_grad():
                for batch in val_loader:
                    big5 = batch["big5"].to(self.device)
                    cmi = batch["cmi"].to(self.device)
                    rppg = batch["rppg"].to(self.device)
                    voice = batch["voice"].to(self.device)
                    targets = batch["target"].to(self.device)

                    predictions = model(big5, cmi, rppg, voice)
                    val_predictions.extend(predictions.cpu().numpy().flatten())
                    val_targets.extend(targets.cpu().numpy().flatten())

            val_r2 = r2_score(val_targets, val_predictions)

            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        return best_val_r2

    def predict(self, X: Dict) -> np.ndarray:
        """예측"""
        if self.model_type == "neural_network":
            # 신경망 모델 예측
            self.model.eval()

            from torch.utils.data import DataLoader

            from transfer_learning_multimodal import TransferLearningMultimodalDataset

            dataset = TransferLearningMultimodalDataset(
                X["big5"],
                X["cmi"],
                X["rppg"],
                X["voice"],
                np.zeros(len(X["big5"])),
                augment=False,
            )
            dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

            predictions = []
            with torch.no_grad():
                for batch in dataloader:
                    big5 = batch["big5"].to(self.device)
                    cmi = batch["cmi"].to(self.device)
                    rppg = batch["rppg"].to(self.device)
                    voice = batch["voice"].to(self.device)

                    pred = self.model(big5, cmi, rppg, voice)
                    predictions.extend(pred.cpu().numpy().flatten())

            return np.array(predictions)
        else:
            # Scikit-learn 모델 예측
            X_combined = np.concatenate(
                [X["big5"], X["cmi"], X["rppg"], X["voice"]], axis=1
            )

            return self.model.predict(X_combined)


def test_best_single_model():
    """최고 성능 단일 모델 테스트"""

    print("🧪 최고 성능 단일 모델 테스트 시작!")

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

    # 최고 성능 모델 생성 및 훈련
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = BestSingleModel(device)
    best_model.train_best_model(X_train, y_train, X_val, y_val)

    # 테스트 예측
    test_predictions = best_model.predict(X_test)

    # 성능 평가
    test_r2 = r2_score(y_test, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    test_mae = mean_absolute_error(y_test, test_predictions)

    print("\n📊 최고 성능 단일 모델 테스트 결과:")
    print(f"  모델 타입: {best_model.model_type}")
    print(f"  테스트 R²: {test_r2:.4f}")
    print(f"  테스트 RMSE: {test_rmse:.4f}")
    print(f"  테스트 MAE: {test_mae:.4f}")

    # 결과 저장
    results = {
        "best_single_model_test": {
            "model_type": best_model.model_type,
            "test_r2": test_r2,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
        }
    }

    with open("best_single_model_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("✅ 결과 저장 완료!")
    print("📁 best_single_model_test_results.json")

    return results


if __name__ == "__main__":
    results = test_best_single_model()


