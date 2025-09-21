#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Realistic Multimodal Training System
- 과적합 방지
- 현실적인 데이터
- 일반화 성능 향상
"""

import json
import os
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


class RealisticMultimodalDataset(Dataset):
    """현실적인 멀티모달 데이터셋"""

    def __init__(self, big5_data, cmi_data, rppg_data, voice_data, targets):
        self.big5_data = torch.FloatTensor(big5_data)
        self.cmi_data = torch.FloatTensor(cmi_data)
        self.rppg_data = torch.FloatTensor(rppg_data)
        self.voice_data = torch.FloatTensor(voice_data)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.big5_data)

    def __getitem__(self, idx):
        return {
            "big5": self.big5_data[idx],
            "cmi": self.cmi_data[idx],
            "rppg": self.rppg_data[idx],
            "voice": self.voice_data[idx],
            "target": self.targets[idx],
        }


class RealisticMultimodalNet(nn.Module):
    """과적합 방지 멀티모달 네트워크"""

    def __init__(
        self,
        big5_dim=25,
        cmi_dim=10,
        rppg_dim=15,
        voice_dim=20,
        hidden_dim=64,  # 더 작은 네트워크
        output_dim=1,
        dropout_rate=0.5,  # 더 높은 드롭아웃
    ):
        super(RealisticMultimodalNet, self).__init__()

        # 각 모달리티별 인코더 (더 간단하게)
        self.big5_encoder = nn.Sequential(
            nn.Linear(big5_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.cmi_encoder = nn.Sequential(
            nn.Linear(cmi_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.rppg_encoder = nn.Sequential(
            nn.Linear(rppg_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.voice_encoder = nn.Sequential(
            nn.Linear(voice_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # 융합 레이어 (더 간단하게)
        fusion_input_dim = (hidden_dim // 2) + (hidden_dim // 4) * 3
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        # L2 정규화를 위한 가중치
        self.weight_decay = 1e-4

    def forward(self, big5, cmi, rppg, voice):
        """Forward pass"""
        # 각 모달리티 인코딩
        big5_encoded = self.big5_encoder(big5)
        cmi_encoded = self.cmi_encoder(cmi)
        rppg_encoded = self.rppg_encoder(rppg)
        voice_encoded = self.voice_encoder(voice)

        # 융합
        fused_features = torch.cat(
            [big5_encoded, cmi_encoded, rppg_encoded, voice_encoded], dim=1
        )

        # 최종 예측
        output = self.fusion_layer(fused_features)

        return output

    def l2_regularization(self):
        """L2 정규화 계산"""
        l2_reg = 0
        for param in self.parameters():
            l2_reg += torch.norm(param, 2)
        return self.weight_decay * l2_reg


class RealisticMultimodalTrainer:
    """현실적인 멀티모달 훈련 시스템"""

    def __init__(self, project_id: str = "persona-diary-service"):
        self.project_id = project_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scalers = {}
        self.model = None
        self.training_history = []

        print(f"Device: {self.device}")

    def generate_realistic_data(
        self, n_samples: int = 10000
    ) -> Tuple[Dict, np.ndarray]:
        """현실적인 데이터 생성 (노이즈 추가)"""
        print(f"Generating realistic data... ({n_samples} samples)")

        np.random.seed(42)

        # Big5 데이터 (더 현실적인 분포)
        big5_data = np.random.normal(3.0, 1.2, (n_samples, 25))
        big5_data = np.clip(big5_data, 1.0, 5.0)

        # 노이즈 추가
        big5_noise = np.random.normal(0, 0.3, big5_data.shape)
        big5_data += big5_noise
        big5_data = np.clip(big5_data, 1.0, 5.0)

        # CMI 데이터 (더 현실적인 분포)
        cmi_data = np.random.normal(45, 20, (n_samples, 10))
        cmi_data = np.clip(cmi_data, 0, 100)

        # 노이즈 추가
        cmi_noise = np.random.normal(0, 5, cmi_data.shape)
        cmi_data += cmi_noise
        cmi_data = np.clip(cmi_data, 0, 100)

        # RPPG 데이터 (더 현실적인 분포)
        rppg_data = np.random.normal(65, 15, (n_samples, 15))
        rppg_data = np.clip(rppg_data, 40, 120)

        # 노이즈 추가
        rppg_noise = np.random.normal(0, 3, rppg_data.shape)
        rppg_data += rppg_noise
        rppg_data = np.clip(rppg_data, 40, 120)

        # Voice 데이터 (더 현실적인 분포)
        voice_data = np.random.normal(180, 60, (n_samples, 20))
        voice_data = np.clip(voice_data, 50, 400)

        # 노이즈 추가
        voice_noise = np.random.normal(0, 15, voice_data.shape)
        voice_data += voice_noise
        voice_data = np.clip(voice_data, 50, 400)

        # 타겟 변수 생성 (더 복잡하고 현실적으로)
        big5_scores = {
            "EXT": big5_data[:, :5].mean(axis=1),
            "EST": big5_data[:, 5:10].mean(axis=1),
            "AGR": big5_data[:, 10:15].mean(axis=1),
            "CSN": big5_data[:, 15:20].mean(axis=1),
            "OPN": big5_data[:, 20:25].mean(axis=1),
        }

        # 더 복잡한 타겟 함수 (비선형 관계)
        targets = (
            big5_scores["EXT"] * 0.2
            + big5_scores["OPN"] * 0.15
            + (5 - big5_scores["EST"]) * 0.1
            + big5_scores["AGR"] * 0.1
            + big5_scores["CSN"] * 0.05
            + (cmi_data.mean(axis=1) / 100) * 0.1
            + (rppg_data.mean(axis=1) / 100) * 0.05
            + (voice_data.mean(axis=1) / 300) * 0.05
            + np.sin(big5_scores["EXT"]) * 0.1  # 비선형 항
            + np.cos(big5_scores["OPN"]) * 0.1  # 비선형 항
            + np.random.normal(0, 0.2, n_samples)  # 노이즈
        )

        # 1-10 스케일로 정규화
        targets = (targets - targets.min()) / (targets.max() - targets.min()) * 9 + 1

        multimodal_data = {
            "big5": big5_data,
            "cmi": cmi_data,
            "rppg": rppg_data,
            "voice": voice_data,
        }

        print(f"Realistic data generated:")
        print(f"  Big5: {big5_data.shape}")
        print(f"  CMI: {cmi_data.shape}")
        print(f"  RPPG: {rppg_data.shape}")
        print(f"  Voice: {voice_data.shape}")
        print(f"  Targets: {targets.shape}")

        return multimodal_data, targets

    def prepare_data(
        self,
        multimodal_data: Dict,
        targets: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """데이터 전처리"""
        print("Preprocessing data...")

        # 데이터 정규화
        for modality, data in multimodal_data.items():
            scaler = StandardScaler()
            multimodal_data[modality] = scaler.fit_transform(data)
            self.scalers[modality] = scaler

        # 타겟 정규화
        target_scaler = StandardScaler()
        targets_scaled = target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()
        self.scalers["target"] = target_scaler

        # 데이터 분할
        big5_temp, big5_test, y_temp, y_test = train_test_split(
            multimodal_data["big5"],
            targets_scaled,
            test_size=test_size,
            random_state=42,
        )
        cmi_temp, cmi_test, _, _ = train_test_split(
            multimodal_data["cmi"], targets_scaled, test_size=test_size, random_state=42
        )
        rppg_temp, rppg_test, _, _ = train_test_split(
            multimodal_data["rppg"],
            targets_scaled,
            test_size=test_size,
            random_state=42,
        )
        voice_temp, voice_test, _, _ = train_test_split(
            multimodal_data["voice"],
            targets_scaled,
            test_size=test_size,
            random_state=42,
        )

        # 훈련/검증 분할
        big5_train, big5_val, y_train, y_val = train_test_split(
            big5_temp, y_temp, test_size=val_size / (1 - test_size), random_state=42
        )
        cmi_train, cmi_val, _, _ = train_test_split(
            cmi_temp, y_temp, test_size=val_size / (1 - test_size), random_state=42
        )
        rppg_train, rppg_val, _, _ = train_test_split(
            rppg_temp, y_temp, test_size=val_size / (1 - test_size), random_state=42
        )
        voice_train, voice_val, _, _ = train_test_split(
            voice_temp, y_temp, test_size=val_size / (1 - test_size), random_state=42
        )

        # 딕셔너리로 재구성
        X_train = {
            "big5": big5_train,
            "cmi": cmi_train,
            "rppg": rppg_train,
            "voice": voice_train,
        }
        X_val = {"big5": big5_val, "cmi": cmi_val, "rppg": rppg_val, "voice": voice_val}
        X_test = {
            "big5": big5_test,
            "cmi": cmi_test,
            "rppg": rppg_test,
            "voice": voice_test,
        }

        # DataLoader 생성
        train_dataset = RealisticMultimodalDataset(
            X_train["big5"], X_train["cmi"], X_train["rppg"], X_train["voice"], y_train
        )
        val_dataset = RealisticMultimodalDataset(
            X_val["big5"], X_val["cmi"], X_val["rppg"], X_val["voice"], y_val
        )
        test_dataset = RealisticMultimodalDataset(
            X_test["big5"], X_test["cmi"], X_test["rppg"], X_test["voice"], y_test
        )

        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True
        )  # 더 작은 배치
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        print(f"Data split complete:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val: {len(val_dataset)} samples")
        print(f"  Test: {len(test_dataset)} samples")

        return train_loader, val_loader, test_loader

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        learning_rate: float = 0.001,
    ) -> Dict:
        """모델 훈련 (과적합 방지)"""
        print(f"Training model with overfitting prevention... (Epochs: {epochs})")

        # 모델 초기화
        self.model = RealisticMultimodalNet().to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=10, factor=0.5
        )

        best_val_loss = float("inf")
        patience_counter = 0
        early_stopping_patience = 20

        for epoch in range(epochs):
            # 훈련
            self.model.train()
            train_loss = 0.0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                big5 = batch["big5"].to(self.device)
                cmi = batch["cmi"].to(self.device)
                rppg = batch["rppg"].to(self.device)
                voice = batch["voice"].to(self.device)
                targets = batch["target"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(big5, cmi, rppg, voice)

                # 기본 손실
                basic_loss = criterion(outputs.squeeze(), targets)

                # L2 정규화
                l2_loss = self.model.l2_regularization()

                # 총 손실
                total_loss = basic_loss + l2_loss

                total_loss.backward()

                # 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += total_loss.item()

            # 검증
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    big5 = batch["big5"].to(self.device)
                    cmi = batch["cmi"].to(self.device)
                    rppg = batch["rppg"].to(self.device)
                    voice = batch["voice"].to(self.device)
                    targets = batch["target"].to(self.device)

                    outputs = self.model(big5, cmi, rppg, voice)
                    loss = criterion(outputs.squeeze(), targets)
                    val_loss += loss.item()

            # 평균 손실 계산
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            # 학습률 스케줄링
            scheduler.step(val_loss)

            # 히스토리 저장
            self.training_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

            # 조기 종료 체크
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), "best_realistic_model.pth")
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

            # 진행 상황 출력
            if (epoch + 1) % 20 == 0:
                print(
                    f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

        # 최고 모델 로드
        self.model.load_state_dict(torch.load("best_realistic_model.pth"))

        print(f"Training complete! Best validation loss: {best_val_loss:.4f}")

        return {
            "best_val_loss": best_val_loss,
            "total_epochs": epoch + 1,
            "training_history": self.training_history,
        }

    def evaluate_model(self, test_loader: DataLoader) -> Dict:
        """모델 평가"""
        print("Evaluating model...")

        self.model.eval()
        test_predictions = []
        test_targets = []

        with torch.no_grad():
            for batch in test_loader:
                big5 = batch["big5"].to(self.device)
                cmi = batch["cmi"].to(self.device)
                rppg = batch["rppg"].to(self.device)
                voice = batch["voice"].to(self.device)
                targets = batch["target"].to(self.device)

                outputs = self.model(big5, cmi, rppg, voice)

                test_predictions.extend(outputs.squeeze().cpu().numpy())
                test_targets.extend(targets.cpu().numpy())

        # 메트릭 계산
        test_predictions = np.array(test_predictions)
        test_targets = np.array(test_targets)

        mse = mean_squared_error(test_targets, test_predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(test_targets, test_predictions)
        mae = np.mean(np.abs(test_predictions - test_targets))

        evaluation_results = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "predictions": test_predictions,
            "targets": test_targets,
        }

        print(f"Evaluation complete:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")

        return evaluation_results

    def run_training(self, n_samples: int = 10000, epochs: int = 100) -> Dict:
        """훈련 실행"""
        print("=" * 60)
        print("REALISTIC MULTIMODAL TRAINING SYSTEM")
        print("=" * 60)

        # 1. 데이터 생성
        multimodal_data, targets = self.generate_realistic_data(n_samples)

        # 2. 데이터 준비
        train_loader, val_loader, test_loader = self.prepare_data(
            multimodal_data, targets
        )

        # 3. 모델 훈련
        training_results = self.train_model(train_loader, val_loader, epochs)

        # 4. 모델 평가
        evaluation_results = self.evaluate_model(test_loader)

        # 5. 결과 저장
        results = {
            "training_results": training_results,
            "evaluation_results": evaluation_results,
            "model_info": {
                "device": str(self.device),
                "n_samples": n_samples,
                "epochs_trained": training_results["total_epochs"],
            },
        }

        # JSON으로 저장
        with open("realistic_multimodal_results.json", "w") as f:
            json_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    json_results[key] = {}
                    for k, v in value.items():
                        if isinstance(v, np.ndarray):
                            json_results[key][k] = v.tolist()
                        elif hasattr(v, "item"):  # numpy scalar
                            json_results[key][k] = float(v.item())
                        else:
                            json_results[key][k] = v
                else:
                    json_results[key] = value
            json.dump(json_results, f, indent=2)

        print(f"\nTraining complete!")
        print(f"  Final R²: {evaluation_results['r2']:.4f}")
        print(f"  Final RMSE: {evaluation_results['rmse']:.4f}")

        return results


def main():
    """메인 실행 함수"""
    print("Realistic Multimodal Training System")
    print("=" * 60)

    # 훈련 시스템 초기화
    trainer = RealisticMultimodalTrainer()

    # 훈련 실행
    results = trainer.run_training(n_samples=10000, epochs=100)

    print("\nTraining Summary:")
    print(f"  Device: {results['model_info']['device']}")
    print(f"  Samples: {results['model_info']['n_samples']}")
    print(f"  Epochs: {results['model_info']['epochs_trained']}")
    print(f"  Performance: R² = {results['evaluation_results']['r2']:.4f}")


if __name__ == "__main__":
    main()



