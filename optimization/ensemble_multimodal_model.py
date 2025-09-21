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
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class EnsembleMultimodalModel:
    """앙상블 멀티모달 모델"""

    def __init__(self, device="cpu"):
        self.device = device
        self.models = {}
        self.scalers = {}
        self.weights = {}

    def add_neural_network_model(
        self, name: str, model: nn.Module, weight: float = 1.0
    ):
        """신경망 모델 추가"""
        self.models[name] = {"type": "neural_network", "model": model, "weight": weight}

    def add_sklearn_model(self, name: str, model, weight: float = 1.0):
        """Scikit-learn 모델 추가"""
        self.models[name] = {"type": "sklearn", "model": model, "weight": weight}

    def train_models(
        self, X_train: Dict, y_train: np.ndarray, X_val: Dict, y_val: np.ndarray
    ):
        """모든 모델 훈련"""

        print("🚀 앙상블 모델 훈련 시작!")

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

        # 각 모델 훈련
        for name, model_info in self.models.items():
            print(f"📚 {name} 모델 훈련 중...")

            if model_info["type"] == "neural_network":
                # 신경망 모델 훈련
                self._train_neural_network(
                    name, model_info, X_train, y_train, X_val, y_val
                )
            else:
                # Scikit-learn 모델 훈련
                self._train_sklearn_model(
                    name, model_info, X_train_combined, y_train, X_val_combined, y_val
                )

        # 가중치 최적화
        self._optimize_weights(X_val, y_val)

        print("✅ 앙상블 모델 훈련 완료!")

    def _train_neural_network(
        self,
        name: str,
        model_info: Dict,
        X_train: Dict,
        y_train: np.ndarray,
        X_val: Dict,
        y_val: np.ndarray,
    ):
        """신경망 모델 훈련"""
        model = model_info["model"]
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

        for epoch in range(20):
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
                # 최고 성능 모델 저장
                model_info["best_state"] = model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        # 최고 성능 모델 복원
        if "best_state" in model_info:
            model.load_state_dict(model_info["best_state"])

        print(f"  {name} 최고 R²: {best_val_r2:.4f}")

    def _train_sklearn_model(
        self,
        name: str,
        model_info: Dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ):
        """Scikit-learn 모델 훈련"""
        model = model_info["model"]

        # 훈련
        model.fit(X_train, y_train)

        # 검증
        val_predictions = model.predict(X_val)
        val_r2 = r2_score(y_val, val_predictions)

        print(f"  {name} R²: {val_r2:.4f}")

    def _optimize_weights(self, X_val: Dict, y_val: np.ndarray):
        """앙상블 가중치 최적화"""
        print("⚖️ 앙상블 가중치 최적화 중...")

        # 각 모델의 예측값 수집
        predictions = {}

        for name, model_info in self.models.items():
            if model_info["type"] == "neural_network":
                # 신경망 모델 예측
                model = model_info["model"]
                model.eval()

                from torch.utils.data import DataLoader

                from transfer_learning_multimodal import (
                    TransferLearningMultimodalDataset,
                )

                val_dataset = TransferLearningMultimodalDataset(
                    X_val["big5"],
                    X_val["cmi"],
                    X_val["rppg"],
                    X_val["voice"],
                    y_val,
                    augment=False,
                )
                val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

                val_predictions = []
                with torch.no_grad():
                    for batch in val_loader:
                        big5 = batch["big5"].to(self.device)
                        cmi = batch["cmi"].to(self.device)
                        rppg = batch["rppg"].to(self.device)
                        voice = batch["voice"].to(self.device)

                        pred = model(big5, cmi, rppg, voice)
                        val_predictions.extend(pred.cpu().numpy().flatten())

                predictions[name] = np.array(val_predictions)
            else:
                # Scikit-learn 모델 예측
                X_val_combined = np.concatenate(
                    [X_val["big5"], X_val["cmi"], X_val["rppg"], X_val["voice"]], axis=1
                )

                predictions[name] = model_info["model"].predict(X_val_combined)

        # 가중치 최적화 (간단한 방법: 각 모델의 R² 점수 기반)
        weights = {}
        for name, pred in predictions.items():
            r2 = r2_score(y_val, pred)
            weights[name] = max(0, r2)  # 음수 R²는 0으로 설정

        # 정규화
        total_weight = sum(weights.values())
        if total_weight > 0:
            for name in weights:
                weights[name] /= total_weight
        else:
            # 모든 가중치가 0이면 균등 분배
            for name in weights:
                weights[name] = 1.0 / len(weights)

        self.weights = weights
        print(f"✅ 최적 가중치: {weights}")

    def predict(self, X: Dict) -> np.ndarray:
        """앙상블 예측"""
        predictions = {}

        for name, model_info in self.models.items():
            if model_info["type"] == "neural_network":
                # 신경망 모델 예측
                model = model_info["model"]
                model.eval()

                from torch.utils.data import DataLoader

                from transfer_learning_multimodal import (
                    TransferLearningMultimodalDataset,
                )

                dataset = TransferLearningMultimodalDataset(
                    X["big5"],
                    X["cmi"],
                    X["rppg"],
                    X["voice"],
                    np.zeros(len(X["big5"])),
                    augment=False,
                )
                dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

                pred = []
                with torch.no_grad():
                    for batch in dataloader:
                        big5 = batch["big5"].to(self.device)
                        cmi = batch["cmi"].to(self.device)
                        rppg = batch["rppg"].to(self.device)
                        voice = batch["voice"].to(self.device)

                        p = model(big5, cmi, rppg, voice)
                        pred.extend(p.cpu().numpy().flatten())

                predictions[name] = np.array(pred)
            else:
                # Scikit-learn 모델 예측
                X_combined = np.concatenate(
                    [X["big5"], X["cmi"], X["rppg"], X["voice"]], axis=1
                )

                predictions[name] = model_info["model"].predict(X_combined)

        # 가중 평균
        ensemble_pred = np.zeros(len(predictions[list(predictions.keys())[0]]))
        for name, pred in predictions.items():
            ensemble_pred += self.weights[name] * pred

        return ensemble_pred


def create_ensemble_models():
    """앙상블 모델 생성"""

    print("🎯 앙상블 멀티모달 모델 생성!")

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    # 앙상블 모델 생성
    ensemble = EnsembleMultimodalModel(device)

    # 1. 신경망 모델들 추가
    from transfer_learning_multimodal import TransferLearningMultimodalNet

    # 모델 1: 기본 전이 학습 모델
    model1 = TransferLearningMultimodalNet(
        hidden_dim=256, dropout_rate=0.3, use_pretrained=False
    )
    ensemble.add_neural_network_model("neural_net_1", model1, weight=1.0)

    # 모델 2: 다른 하이퍼파라미터
    model2 = TransferLearningMultimodalNet(
        hidden_dim=128, dropout_rate=0.4, use_pretrained=False
    )
    ensemble.add_neural_network_model("neural_net_2", model2, weight=1.0)

    # 2. Scikit-learn 모델들 추가
    ensemble.add_sklearn_model(
        "random_forest",
        RandomForestRegressor(n_estimators=100, random_state=42),
        weight=1.0,
    )
    ensemble.add_sklearn_model(
        "gradient_boosting",
        GradientBoostingRegressor(n_estimators=100, random_state=42),
        weight=1.0,
    )
    ensemble.add_sklearn_model("ridge", Ridge(alpha=1.0), weight=1.0)
    ensemble.add_sklearn_model("lasso", Lasso(alpha=0.1), weight=1.0)

    return ensemble


def test_ensemble_model():
    """앙상블 모델 테스트"""

    print("🧪 앙상블 모델 테스트 시작!")

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

    # 앙상블 모델 생성 및 훈련
    ensemble = create_ensemble_models()
    ensemble.train_models(X_train, y_train, X_val, y_val)

    # 테스트 예측
    test_predictions = ensemble.predict(X_test)

    # 성능 평가
    test_r2 = r2_score(y_test, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    test_mae = mean_absolute_error(y_test, test_predictions)

    print("\n📊 앙상블 모델 테스트 결과:")
    print(f"  테스트 R²: {test_r2:.4f}")
    print(f"  테스트 RMSE: {test_rmse:.4f}")
    print(f"  테스트 MAE: {test_mae:.4f}")

    # 결과 저장
    results = {
        "ensemble_test": {
            "test_r2": test_r2,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "model_count": len(ensemble.models),
            "weights": ensemble.weights,
        }
    }

    with open("ensemble_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("✅ 결과 저장 완료!")
    print("📁 ensemble_test_results.json")

    return results


if __name__ == "__main__":
    results = test_ensemble_model()


