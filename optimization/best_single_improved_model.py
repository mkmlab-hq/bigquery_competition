import json
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from google.cloud import bigquery
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from torch.utils.data import DataLoader

from transfer_learning_multimodal import (
    TransferLearningMultimodalDataset,
    TransferLearningMultimodalNet,
    TransferLearningTrainer,
)

warnings.filterwarnings("ignore")


class BestSingleImprovedModel:
    """최고 성능 단일 개선 모델"""

    def __init__(self, device="cpu"):
        self.device = device
        self.model = None
        self.scalers = {}
        self.model_type = None

    def train_best_model(
        self, X_train: Dict, y_train: np.ndarray, X_val: Dict, y_val: np.ndarray
    ):
        """최고 성능 모델 훈련"""
        print("🚀 최고 성능 단일 모델 훈련 시작!")

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
            "neural_network_large": self._create_neural_network_large(X_train),
            "neural_network_medium": self._create_neural_network_medium(X_train),
            "random_forest_strong": RandomForestRegressor(
                n_estimators=200, max_depth=15, random_state=42
            ),
            "gradient_boosting_strong": GradientBoostingRegressor(
                n_estimators=200, max_depth=8, random_state=42
            ),
            "ridge_optimized": Ridge(alpha=0.1),
            "ridge_strong": Ridge(alpha=1.0),
            "ridge_weak": Ridge(alpha=0.01),
        }

        best_r2 = -float("inf")
        best_model = None
        best_name = None

        for name, model in models_to_test.items():
            print(f"📚 {name} 모델 테스트 중...")

            if name.startswith("neural_network"):
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

    def _create_neural_network_large(self, X_train: Dict):
        """큰 신경망 모델 생성"""
        return TransferLearningMultimodalNet(
            big5_dim=X_train["big5"].shape[1],
            cmi_dim=X_train["cmi"].shape[1],
            rppg_dim=X_train["rppg"].shape[1],
            voice_dim=X_train["voice"].shape[1],
            hidden_dim=512,
            dropout_rate=0.3,
            use_pretrained=False,
        )

    def _create_neural_network_medium(self, X_train: Dict):
        """중간 신경망 모델 생성"""
        return TransferLearningMultimodalNet(
            big5_dim=X_train["big5"].shape[1],
            cmi_dim=X_train["cmi"].shape[1],
            rppg_dim=X_train["rppg"].shape[1],
            voice_dim=X_train["voice"].shape[1],
            hidden_dim=256,
            dropout_rate=0.4,
            use_pretrained=False,
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
        train_dataset = TransferLearningMultimodalDataset(
            X_train["big5"],
            X_train["cmi"],
            X_train["rppg"],
            X_train["voice"],
            y_train,
            augment=True,  # 데이터 증강 활성화
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
        patience = 10
        patience_counter = 0

        for epoch in range(50):
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
        if self.model_type.startswith("neural_network"):
            # 신경망 모델 예측
            self.model.eval()

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


class FeatureEngineer:
    """특징 엔지니어링 클래스"""

    def __init__(self):
        self.poly_features = PolynomialFeatures(
            degree=2, include_bias=False, interaction_only=True
        )
        self.scalers = {}

    def create_advanced_features(self, data: dict) -> dict:
        """고급 특징 생성"""
        print("🔧 고급 특징 엔지니어링 시작...")

        # 기본 특징
        big5 = data["big5"]
        cmi = data["cmi"]
        rppg = data["rppg"]
        voice = data["voice"]

        # 1. 통계적 특징 생성
        print("  📊 통계적 특징 생성...")
        big5_stats = self._create_statistical_features(big5, "big5")
        cmi_stats = self._create_statistical_features(cmi, "cmi")
        rppg_stats = self._create_statistical_features(rppg, "rppg")
        voice_stats = self._create_statistical_features(voice, "voice")

        # 2. 상호작용 특징 생성
        print("  🔗 상호작용 특징 생성...")
        big5_interactions = self._create_interaction_features(big5, "big5")
        cmi_interactions = self._create_interaction_features(cmi, "cmi")

        # 3. 모든 특징 결합
        enhanced_data = {
            "big5": np.concatenate([big5, big5_stats, big5_interactions], axis=1),
            "cmi": np.concatenate([cmi, cmi_stats, cmi_interactions], axis=1),
            "rppg": np.concatenate([rppg, rppg_stats], axis=1),
            "voice": np.concatenate([voice, voice_stats], axis=1),
            "targets": data["targets"],
        }

        print(f"✅ 특징 엔지니어링 완료:")
        print(f"  Big5: {big5.shape[1]} → {enhanced_data['big5'].shape[1]} 특징")
        print(f"  CMI: {cmi.shape[1]} → {enhanced_data['cmi'].shape[1]} 특징")
        print(f"  RPPG: {rppg.shape[1]} → {enhanced_data['rppg'].shape[1]} 특징")
        print(f"  Voice: {voice.shape[1]} → {enhanced_data['voice'].shape[1]} 특징")

        return enhanced_data

    def _create_statistical_features(self, data: np.ndarray, name: str) -> np.ndarray:
        """통계적 특징 생성"""
        features = []
        features.append(np.mean(data, axis=1, keepdims=True))  # 평균
        features.append(np.std(data, axis=1, keepdims=True))  # 표준편차
        features.append(np.var(data, axis=1, keepdims=True))  # 분산
        features.append(np.max(data, axis=1, keepdims=True))  # 최대값
        features.append(np.min(data, axis=1, keepdims=True))  # 최소값
        features.append(np.median(data, axis=1, keepdims=True))  # 중앙값

        return np.concatenate(features, axis=1)

    def _create_interaction_features(self, data: np.ndarray, name: str) -> np.ndarray:
        """상호작용 특징 생성"""
        if data.shape[1] < 2:
            return np.array([]).reshape(data.shape[0], 0)

        # 상위 3개 특징 간의 상호작용
        n_features = min(3, data.shape[1])
        interactions = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                interactions.append((data[:, i] * data[:, j]).reshape(-1, 1))

        if interactions:
            return np.concatenate(interactions, axis=1)
        else:
            return np.array([]).reshape(data.shape[0], 0)


def test_best_single_improved_model():
    """최고 성능 단일 개선 모델 테스트"""
    print("🧪 최고 성능 단일 개선 모델 테스트 시작!")

    # 1. 데이터 생성 (더 많은 샘플)
    np.random.seed(42)
    n_samples = 20000

    big5_data = np.random.beta(2, 2, (n_samples, 5))
    cmi_data = np.random.beta(1.5, 1.5, (n_samples, 20))
    rppg_data = np.random.normal(0, 1, (n_samples, 10))
    voice_data = np.random.normal(0, 1, (n_samples, 20))

    # 더 복잡한 타겟 생성
    targets = (
        0.20 * big5_data[:, 0]  # Openness
        + 0.18 * big5_data[:, 1]  # Conscientiousness
        + 0.15 * big5_data[:, 2]  # Extraversion
        + 0.17 * big5_data[:, 3]  # Agreeableness
        + 0.12 * big5_data[:, 4]  # Neuroticism
        + 0.08 * np.mean(cmi_data, axis=1)
        + 0.05 * np.mean(rppg_data, axis=1)
        + 0.03 * np.mean(voice_data, axis=1)
        # 비선형 상호작용 추가
        + 0.02 * (big5_data[:, 0] * big5_data[:, 1])  # Openness × Conscientiousness
        + 0.01 * (big5_data[:, 2] * big5_data[:, 3])  # Extraversion × Agreeableness
        + np.random.normal(0, 0.12, n_samples)  # 노이즈
    )

    targets = (targets - targets.min()) / (targets.max() - targets.min())

    # 2. 특징 엔지니어링
    data = {
        "big5": big5_data,
        "cmi": cmi_data,
        "rppg": rppg_data,
        "voice": voice_data,
        "targets": targets,
    }

    feature_engineer = FeatureEngineer()
    enhanced_data = feature_engineer.create_advanced_features(data)

    # 3. 데이터 분할
    train_idx, test_idx = train_test_split(
        range(len(enhanced_data["targets"])), test_size=0.3, random_state=42
    )
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)

    X_train = {
        modality: enhanced_data[modality][train_idx]
        for modality in ["big5", "cmi", "rppg", "voice"]
    }
    X_val = {
        modality: enhanced_data[modality][val_idx]
        for modality in ["big5", "cmi", "rppg", "voice"]
    }
    X_test = {
        modality: enhanced_data[modality][test_idx]
        for modality in ["big5", "cmi", "rppg", "voice"]
    }

    y_train = enhanced_data["targets"][train_idx]
    y_val = enhanced_data["targets"][val_idx]
    y_test = enhanced_data["targets"][test_idx]

    print(f"훈련 데이터: {len(y_train)}개")
    print(f"검증 데이터: {len(y_val)}개")
    print(f"테스트 데이터: {len(y_test)}개")

    # 4. 최고 성능 단일 모델 생성 및 훈련
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = BestSingleImprovedModel(device)
    best_model.train_best_model(X_train, y_train, X_val, y_val)

    # 5. 테스트 예측
    test_predictions = best_model.predict(X_test)

    # 6. 성능 평가
    test_r2 = r2_score(y_test, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    test_mae = mean_absolute_error(y_test, test_predictions)

    print("\n📊 최고 성능 단일 개선 모델 테스트 결과:")
    print(f"  모델 타입: {best_model.model_type}")
    print(f"  테스트 R²: {test_r2:.4f}")
    print(f"  테스트 RMSE: {test_rmse:.4f}")
    print(f"  테스트 MAE: {test_mae:.4f}")

    # 7. 결과 저장
    results = {
        "best_single_improved_model_test": {
            "model_type": best_model.model_type,
            "test_r2": test_r2,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "sample_count": len(enhanced_data["targets"]),
        }
    }

    with open("best_single_improved_model_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("✅ 결과 저장 완료!")
    print("📁 best_single_improved_model_test_results.json")

    return results


if __name__ == "__main__":
    results = test_best_single_improved_model()


