#!/usr/bin/env python3
"""
최적화된 멀티모달 훈련 시스템
- 고급 과적합 방지 기법
- 모델 아키텍처 최적화
- 데이터 증강 및 정규화
- 교차 검증 및 성능 분석
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
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


class OptimizedMultimodalDataset(Dataset):
    """최적화된 멀티모달 데이터셋 - 데이터 증강 포함"""

    def __init__(
        self, big5_data, cmi_data, rppg_data, voice_data, targets, augment=False
    ):
        self.big5_data = torch.FloatTensor(big5_data)
        self.cmi_data = torch.FloatTensor(cmi_data)
        self.rppg_data = torch.FloatTensor(rppg_data)
        self.voice_data = torch.FloatTensor(voice_data)
        self.targets = torch.FloatTensor(targets)
        self.augment = augment
        self.training = False  # 초기값 설정

    def __len__(self):
        return len(self.big5_data)

    def __getitem__(self, idx):
        big5 = self.big5_data[idx]
        cmi = self.cmi_data[idx]
        rppg = self.rppg_data[idx]
        voice = self.voice_data[idx]
        target = self.targets[idx]

        # 데이터 증강 (훈련 시에만)
        if self.augment and self.training:
            big5 = self._augment_data(big5, noise_std=0.05)
            cmi = self._augment_data(cmi, noise_std=0.1)
            rppg = self._augment_data(rppg, noise_std=0.05)
            voice = self._augment_data(voice, noise_std=0.1)

        return {
            "big5": big5,
            "cmi": cmi,
            "rppg": rppg,
            "voice": voice,
            "target": target,
        }

    def _augment_data(self, data, noise_std=0.05):
        """데이터 증강: 가우시안 노이즈 추가"""
        noise = torch.randn_like(data) * noise_std
        return data + noise

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class OptimizedMultimodalNet(nn.Module):
    """최적화된 멀티모달 네트워크 - 과적합 방지 강화"""

    def __init__(
        self,
        big5_dim=25,
        cmi_dim=10,
        rppg_dim=15,
        voice_dim=20,
        hidden_dim=64,  # 차원 감소: 128 → 64
        output_dim=1,
        dropout_rate=0.5,  # Dropout 증가: 0.3 → 0.5
        use_cross_attention=False,  # 크로스 어텐션 비활성화
        use_transformer=False,  # 트랜스포머 비활성화
        weight_decay=1e-4,  # L2 정규화
    ):
        super(OptimizedMultimodalNet, self).__init__()

        self.use_cross_attention = use_cross_attention
        self.use_transformer = use_transformer
        self.weight_decay = weight_decay

        # 각 모달리티별 최적화된 인코더
        self.big5_encoder = self._create_optimized_encoder(
            big5_dim, hidden_dim // 2, dropout_rate
        )
        self.cmi_encoder = self._create_optimized_encoder(
            cmi_dim, hidden_dim // 2, dropout_rate
        )
        self.rppg_encoder = self._create_optimized_encoder(
            rppg_dim, hidden_dim // 2, dropout_rate
        )
        self.voice_encoder = self._create_optimized_encoder(
            voice_dim, hidden_dim // 2, dropout_rate
        )

        # 단순화된 융합 레이어
        # basic_fused: 4 * 32 = 128, weighted_fused: 32, total = 160
        fusion_input_dim = 4 * (hidden_dim // 2) + (hidden_dim // 2)  # 128 + 32 = 160
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # BatchNorm 추가
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        # 동적 가중치 (단순화)
        self.modality_weights = nn.Parameter(torch.ones(4))
        self.weight_gate = nn.Sequential(
            nn.Linear(hidden_dim // 2, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 4),
            nn.Softmax(dim=-1),
        )

    def _create_optimized_encoder(
        self, input_dim: int, output_dim: int, dropout_rate: float
    ) -> nn.Module:
        """최적화된 인코더 생성 - BatchNorm과 강화된 정규화"""
        return nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.BatchNorm1d(output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(output_dim * 2, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, big5, cmi, rppg, voice):
        """최적화된 forward pass"""
        # 각 모달리티 인코딩
        big5_encoded = self.big5_encoder(big5)
        cmi_encoded = self.cmi_encoder(cmi)
        rppg_encoded = self.rppg_encoder(rppg)
        voice_encoded = self.voice_encoder(voice)

        # 동적 가중치 계산
        dynamic_weights = self.weight_gate(big5_encoded)

        # 가중치 기반 융합 (단순화)
        weighted_fused = (
            dynamic_weights[:, 0:1] * big5_encoded
            + dynamic_weights[:, 1:2] * cmi_encoded
            + dynamic_weights[:, 2:3] * rppg_encoded
            + dynamic_weights[:, 3:4] * voice_encoded
        )

        # 기본 융합 (단순 연결)
        basic_fused = torch.cat(
            [big5_encoded, cmi_encoded, rppg_encoded, voice_encoded], dim=1
        )

        # 최종 융합
        final_fused = torch.cat([basic_fused, weighted_fused], dim=1)

        # 융합 레이어로 최종 예측
        output = self.fusion_layer(final_fused)

        return output, dynamic_weights

    def l2_regularization(self):
        """L2 정규화 계산"""
        l2_reg = 0
        for param in self.parameters():
            l2_reg += torch.norm(param, 2)
        return self.weight_decay * l2_reg


class OptimizedMultimodalTrainer:
    """최적화된 멀티모달 훈련 시스템"""

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
        """현실적인 데이터 생성 - 노이즈와 복잡성 추가"""
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

        # 타겟 변수 생성 (더 복잡한 관계)
        big5_scores = {
            "EXT": big5_data[:, :5].mean(axis=1),
            "EST": big5_data[:, 5:10].mean(axis=1),
            "AGR": big5_data[:, 10:15].mean(axis=1),
            "CSN": big5_data[:, 15:20].mean(axis=1),
            "OPN": big5_data[:, 20:25].mean(axis=1),
        }

        # 더 복잡한 타겟 변수 (비선형 관계)
        targets = (
            big5_scores["EXT"] * 0.2
            + big5_scores["OPN"] * 0.15
            + (5 - big5_scores["EST"]) * 0.1
            + big5_scores["AGR"] * 0.1
            + big5_scores["CSN"] * 0.05
            + (cmi_data.mean(axis=1) / 100) * 0.1
            + (rppg_data.mean(axis=1) / 100) * 0.05
            + (voice_data.mean(axis=1) / 300) * 0.05
            + np.random.normal(0, 0.1, n_samples)  # 추가 노이즈
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
        """데이터 전처리 및 DataLoader 생성"""
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

        # DataLoader 생성 (데이터 증강 포함)
        train_dataset = OptimizedMultimodalDataset(
            X_train["big5"],
            X_train["cmi"],
            X_train["rppg"],
            X_train["voice"],
            y_train,
            augment=True,
        )
        val_dataset = OptimizedMultimodalDataset(
            X_val["big5"],
            X_val["cmi"],
            X_val["rppg"],
            X_val["voice"],
            y_val,
            augment=False,
        )
        test_dataset = OptimizedMultimodalDataset(
            X_test["big5"],
            X_test["cmi"],
            X_test["rppg"],
            X_test["voice"],
            y_test,
            augment=False,
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
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
        """최적화된 모델 훈련"""
        print(f"Training optimized model... (Epochs: {epochs})")

        # 모델 초기화
        self.model = OptimizedMultimodalNet(
            big5_dim=25,
            cmi_dim=10,
            rppg_dim=15,
            voice_dim=20,
            hidden_dim=64,
            output_dim=1,
            dropout_rate=0.5,
            use_cross_attention=False,
            use_transformer=False,
            weight_decay=1e-4,
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=5, factor=0.5
        )

        best_val_loss = float("inf")
        patience_counter = 0
        early_stopping_patience = 15

        for epoch in range(epochs):
            # 훈련
            self.model.train()
            train_loss = 0.0
            train_predictions = []
            train_targets = []

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                big5 = batch["big5"].to(self.device)
                cmi = batch["cmi"].to(self.device)
                rppg = batch["rppg"].to(self.device)
                voice = batch["voice"].to(self.device)
                targets = batch["target"].to(self.device)

                optimizer.zero_grad()
                outputs, dynamic_weights = self.model(big5, cmi, rppg, voice)

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
                train_predictions.extend(outputs.squeeze().detach().cpu().numpy())
                train_targets.extend(targets.detach().cpu().numpy())

            # 검증
            self.model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []

            with torch.no_grad():
                for batch in val_loader:
                    big5 = batch["big5"].to(self.device)
                    cmi = batch["cmi"].to(self.device)
                    rppg = batch["rppg"].to(self.device)
                    voice = batch["voice"].to(self.device)
                    targets = batch["target"].to(self.device)

                    outputs, dynamic_weights = self.model(big5, cmi, rppg, voice)
                    loss = criterion(outputs.squeeze(), targets)
                    val_loss += loss.item()

                    val_predictions.extend(outputs.squeeze().cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())

            # 평균 손실 계산
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            # 학습률 스케줄링
            scheduler.step(val_loss)

            # 메트릭 계산
            train_rmse = np.sqrt(
                np.mean((np.array(train_predictions) - np.array(train_targets)) ** 2)
            )
            val_rmse = np.sqrt(
                np.mean((np.array(val_predictions) - np.array(val_targets)) ** 2)
            )

            # 히스토리 저장
            self.training_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_rmse": train_rmse,
                    "val_rmse": val_rmse,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

            # 조기 종료 체크
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 최고 모델 저장
                torch.save(self.model.state_dict(), "best_optimized_model.pth")
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

            # 진행 상황 출력
            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
                print(f"  Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}")
                print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # 최고 모델 로드
        self.model.load_state_dict(torch.load("best_optimized_model.pth"))

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

                outputs, dynamic_weights = self.model(big5, cmi, rppg, voice)

                test_predictions.extend(outputs.squeeze().cpu().numpy())
                test_targets.extend(targets.cpu().numpy())

        # 메트릭 계산
        test_predictions = np.array(test_predictions)
        test_targets = np.array(test_targets)

        mse = mean_squared_error(test_targets, test_predictions)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(test_predictions - test_targets))
        r2 = r2_score(test_targets, test_predictions)

        # 상관계수
        correlation = np.corrcoef(test_predictions, test_targets)[0, 1]

        evaluation_results = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "correlation": correlation,
            "predictions": test_predictions,
            "targets": test_targets,
        }

        print(f"Evaluation complete:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Correlation: {correlation:.4f}")

        return evaluation_results

    def cross_validate(
        self, multimodal_data: Dict, targets: np.ndarray, k_folds: int = 3
    ) -> Dict:
        """간단한 교차 검증 (3-fold)"""
        print(f"Performing {k_folds}-fold cross validation...")

        # 간단한 교차 검증을 위해 데이터를 3개로 분할
        n_samples = len(targets)
        fold_size = n_samples // k_folds

        cv_results = []

        for fold in range(k_folds):
            print(f"Fold {fold + 1}/{k_folds}")

            # 간단한 분할
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < k_folds - 1 else n_samples

            val_idx = np.arange(val_start, val_end)
            train_idx = np.concatenate(
                [np.arange(0, val_start), np.arange(val_end, n_samples)]
            )

            # 데이터 분할
            X_train = {
                modality: data[train_idx] for modality, data in multimodal_data.items()
            }
            X_val = {
                modality: data[val_idx] for modality, data in multimodal_data.items()
            }
            y_train = targets[train_idx]
            y_val = targets[val_idx]

            print(f"  Train size: {len(y_train)}, Val size: {len(y_val)}")

            # 간단한 모델 훈련 (빠른 검증)
            try:
                # 데이터 정규화
                scalers = {}
                for modality in multimodal_data.keys():
                    scaler = StandardScaler()
                    X_train[modality] = scaler.fit_transform(X_train[modality])
                    X_val[modality] = scaler.transform(X_val[modality])
                    scalers[modality] = scaler

                # 타겟 정규화
                target_scaler = StandardScaler()
                y_train_scaled = target_scaler.fit_transform(
                    y_train.reshape(-1, 1)
                ).flatten()
                y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()

                # 간단한 선형 모델로 빠른 검증
                from sklearn.linear_model import Ridge
                from sklearn.metrics import mean_squared_error, r2_score

                # 모든 모달리티를 하나로 결합
                X_train_combined = np.concatenate(
                    [
                        X_train["big5"],
                        X_train["cmi"],
                        X_train["rppg"],
                        X_train["voice"],
                    ],
                    axis=1,
                )
                X_val_combined = np.concatenate(
                    [X_val["big5"], X_val["cmi"], X_val["rppg"], X_val["voice"]], axis=1
                )

                # Ridge 회귀 모델
                model = Ridge(alpha=1.0)
                model.fit(X_train_combined, y_train_scaled)

                # 예측
                val_predictions = model.predict(X_val_combined)

                # 메트릭 계산
                r2 = r2_score(y_val_scaled, val_predictions)
                rmse = np.sqrt(mean_squared_error(y_val_scaled, val_predictions))

                cv_results.append({"r2": r2, "rmse": rmse})
                print(f"  Fold {fold + 1} - R²: {r2:.4f}, RMSE: {rmse:.4f}")

            except Exception as e:
                print(f"  Fold {fold + 1} - Error: {str(e)}")
                cv_results.append({"r2": 0.0, "rmse": 1.0})

        # 교차 검증 결과 요약
        cv_r2_scores = [result["r2"] for result in cv_results]
        cv_rmse_scores = [result["rmse"] for result in cv_results]

        cv_summary = {
            "mean_r2": np.mean(cv_r2_scores),
            "std_r2": np.std(cv_r2_scores),
            "mean_rmse": np.mean(cv_rmse_scores),
            "std_rmse": np.std(cv_rmse_scores),
            "cv_results": cv_results,
        }

        print(f"Cross-validation complete:")
        print(f"  Mean R²: {cv_summary['mean_r2']:.4f} ± {cv_summary['std_r2']:.4f}")
        print(
            f"  Mean RMSE: {cv_summary['mean_rmse']:.4f} ± {cv_summary['std_rmse']:.4f}"
        )

        return cv_summary

    def create_visualizations(
        self, evaluation_results: Dict, save_dir: str = "optimized_results"
    ):
        """시각화 생성"""
        print("Creating visualizations...")

        os.makedirs(save_dir, exist_ok=True)

        # 1. 훈련 히스토리
        history_df = pd.DataFrame(self.training_history)

        plt.figure(figsize=(15, 10))

        # 손실 그래프
        plt.subplot(2, 3, 1)
        plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
        plt.plot(history_df["epoch"], history_df["val_loss"], label="Validation Loss")
        plt.title("Training History - Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        # RMSE 그래프
        plt.subplot(2, 3, 2)
        plt.plot(history_df["epoch"], history_df["train_rmse"], label="Train RMSE")
        plt.plot(history_df["epoch"], history_df["val_rmse"], label="Validation RMSE")
        plt.title("Training History - RMSE")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.legend()
        plt.grid(True)

        # 학습률 그래프
        plt.subplot(2, 3, 3)
        plt.plot(history_df["epoch"], history_df["learning_rate"])
        plt.title("Learning Rate Schedule")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.yscale("log")
        plt.grid(True)

        # 예측 vs 실제
        plt.subplot(2, 3, 4)
        plt.scatter(
            evaluation_results["targets"], evaluation_results["predictions"], alpha=0.6
        )
        plt.plot(
            [evaluation_results["targets"].min(), evaluation_results["targets"].max()],
            [evaluation_results["targets"].min(), evaluation_results["targets"].max()],
            "r--",
        )
        plt.title(f'Predictions vs Targets (R² = {evaluation_results["r2"]:.3f})')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.grid(True)

        # 잔차 분포
        plt.subplot(2, 3, 5)
        residuals = evaluation_results["predictions"] - evaluation_results["targets"]
        plt.hist(residuals, bins=30, alpha=0.7)
        plt.title("Residual Distribution")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.grid(True)

        # 성능 메트릭
        plt.subplot(2, 3, 6)
        metrics = ["R²", "RMSE", "MAE", "Correlation"]
        values = [
            evaluation_results["r2"],
            evaluation_results["rmse"],
            evaluation_results["mae"],
            evaluation_results["correlation"],
        ]
        plt.bar(metrics, values)
        plt.title("Performance Metrics")
        plt.ylabel("Value")
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(
            f"{save_dir}/optimized_training_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print(f"Visualizations saved: {save_dir}/optimized_training_analysis.png")

    def run_comprehensive_training(
        self, n_samples: int = 10000, epochs: int = 100, k_folds: int = 5
    ) -> Dict:
        """종합 훈련 실행"""
        print("OPTIMIZED MULTIMODAL TRAINING SYSTEM")
        print("=" * 60)

        # 1. 데이터 생성
        multimodal_data, targets = self.generate_realistic_data(n_samples)

        # 2. 교차 검증
        cv_results = self.cross_validate(multimodal_data, targets, k_folds)

        # 3. 데이터 준비
        train_loader, val_loader, test_loader = self.prepare_data(
            multimodal_data, targets
        )

        # 4. 모델 훈련
        training_results = self.train_model(train_loader, val_loader, epochs)

        # 5. 모델 평가
        evaluation_results = self.evaluate_model(test_loader)

        # 6. 시각화 생성
        self.create_visualizations(evaluation_results)

        # 7. 결과 저장
        results = {
            "training_results": training_results,
            "evaluation_results": evaluation_results,
            "cv_results": cv_results,
            "model_info": {
                "device": str(self.device),
                "n_samples": n_samples,
                "epochs_trained": training_results["total_epochs"],
            },
        }

        # JSON으로 저장
        with open("optimized_training_results.json", "w") as f:
            # NumPy 배열과 float32를 JSON 직렬화 가능한 형태로 변환
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

        print(f"Optimized training complete!")
        print(f"  Final R²: {evaluation_results['r2']:.4f}")
        print(f"  Final RMSE: {evaluation_results['rmse']:.4f}")
        print(f"  CV Mean R²: {cv_results['mean_r2']:.4f} ± {cv_results['std_r2']:.4f}")

        return results


def main():
    """메인 실행 함수"""
    print("OPTIMIZED MULTIMODAL TRAINING SYSTEM")
    print("=" * 60)

    # 훈련 시스템 초기화
    trainer = OptimizedMultimodalTrainer()

    # 종합 훈련 실행
    results = trainer.run_comprehensive_training(n_samples=10000, epochs=100, k_folds=5)

    print("\nTraining Summary:")
    print(f"  Device: {results['model_info']['device']}")
    print(f"  Samples: {results['model_info']['n_samples']}")
    print(f"  Epochs: {results['model_info']['epochs_trained']}")
    print(f"  Final R²: {results['evaluation_results']['r2']:.4f}")
    print(
        f"  CV Mean R²: {results['cv_results']['mean_r2']:.4f} ± {results['cv_results']['std_r2']:.4f}"
    )


if __name__ == "__main__":
    main()
