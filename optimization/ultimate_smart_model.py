#!/usr/bin/env python3
"""
최고 성능 스마트 멀티모달 시스템
- 리스크 최소화하면서 성능 최대화
- 단일 모델로 앙상블 수준 성능 달성
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


class SmartMultimodalDataset(Dataset):
    """스마트 멀티모달 데이터셋 - 고급 증강 포함"""

    def __init__(
        self, big5_data, cmi_data, rppg_data, voice_data, targets, augment=False
    ):
        self.big5_data = torch.FloatTensor(big5_data)
        self.cmi_data = torch.FloatTensor(cmi_data)
        self.rppg_data = torch.FloatTensor(rppg_data)
        self.voice_data = torch.FloatTensor(voice_data)
        self.targets = torch.FloatTensor(targets)
        self.augment = augment
        self.training = False

    def __len__(self):
        return len(self.big5_data)

    def __getitem__(self, idx):
        big5 = self.big5_data[idx]
        cmi = self.cmi_data[idx]
        rppg = self.rppg_data[idx]
        voice = self.voice_data[idx]
        target = self.targets[idx]

        # 고급 데이터 증강
        if self.augment and self.training:
            big5 = self._advanced_augment(big5, modality="big5")
            cmi = self._advanced_augment(cmi, modality="cmi")
            rppg = self._advanced_augment(rppg, modality="rppg")
            voice = self._advanced_augment(voice, modality="voice")

        return {
            "big5": big5,
            "cmi": cmi,
            "rppg": rppg,
            "voice": voice,
            "target": target,
        }

    def _advanced_augment(self, data, modality="big5"):
        """고급 데이터 증강"""
        # Mixup (두 샘플 혼합)
        if np.random.random() < 0.3:
            alpha = np.random.beta(0.2, 0.2)
            idx2 = np.random.randint(0, len(self.big5_data))
            if modality == "big5":
                data2 = self.big5_data[idx2]
            elif modality == "cmi":
                data2 = self.cmi_data[idx2]
            elif modality == "rppg":
                data2 = self.rppg_data[idx2]
            else:
                data2 = self.voice_data[idx2]
            data = alpha * data + (1 - alpha) * data2

        # CutMix (일부 영역 교체)
        if np.random.random() < 0.2:
            cut_ratio = np.random.uniform(0.1, 0.3)
            cut_size = int(len(data) * cut_ratio)
            cut_start = np.random.randint(0, len(data) - cut_size)
            data[cut_start : cut_start + cut_size] = 0

        # 노이즈 추가 (모달리티별 차별화)
        noise_std = {"big5": 0.05, "cmi": 0.1, "rppg": 0.05, "voice": 0.1}[modality]
        noise = torch.randn_like(data) * noise_std
        data = data + noise

        return data

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class UltimateSmartMultimodalNet(nn.Module):
    """최고 성능 스마트 멀티모달 네트워크"""

    def __init__(
        self,
        big5_dim=25,
        cmi_dim=10,
        rppg_dim=15,
        voice_dim=20,
        hidden_dim=128,  # 차원 증가
        output_dim=1,
        dropout_rate=0.4,
        use_attention=True,
        use_residual=True,
        use_skip_connections=True,
    ):
        super(UltimateSmartMultimodalNet, self).__init__()

        self.use_attention = use_attention
        self.use_residual = use_residual
        self.use_skip_connections = use_skip_connections

        # 모달리티별 특화 인코더
        self.big5_encoder = self._create_specialized_encoder(
            big5_dim, hidden_dim, "big5"
        )
        self.cmi_encoder = self._create_specialized_encoder(cmi_dim, hidden_dim, "cmi")
        self.rppg_encoder = self._create_specialized_encoder(
            rppg_dim, hidden_dim, "rppg"
        )
        self.voice_encoder = self._create_specialized_encoder(
            voice_dim, hidden_dim, "voice"
        )

        # Cross-modal Attention
        if self.use_attention:
            self.cross_attention = nn.MultiheadAttention(
                hidden_dim, num_heads=8, dropout=dropout_rate, batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hidden_dim)

        # Multi-scale Feature Fusion
        self.multi_scale_fusion = nn.ModuleList(
            [
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 2),
            ]
        )

        # Residual Connections
        if self.use_residual:
            self.residual_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate),
                    )
                    for _ in range(3)
                ]
            )

        # Skip Connections
        if self.use_skip_connections:
            self.skip_connections = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim) for _ in range(2)]
            )

        # 최종 융합 레이어
        fusion_input_dim = hidden_dim * 4 + (
            hidden_dim * 2 if self.use_attention else 0
        )
        self.final_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )

        # 동적 가중치 학습
        self.modality_weights = nn.Parameter(torch.ones(4))
        self.weight_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 4),
            nn.Softmax(dim=-1),
        )

    def _create_specialized_encoder(self, input_dim, hidden_dim, modality):
        """모달리티별 특화 인코더"""
        if modality == "big5":
            # Big5: 성격 특성에 특화
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
        elif modality == "cmi":
            # CMI: 인지 능력에 특화
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.4),
            )
        elif modality == "rppg":
            # RPPG: 생체신호에 특화
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
        else:  # voice
            # Voice: 음성 특성에 특화
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
            )

    def forward(self, big5, cmi, rppg, voice):
        """스마트 forward pass"""
        # 각 모달리티 인코딩
        big5_encoded = self.big5_encoder(big5)
        cmi_encoded = self.cmi_encoder(cmi)
        rppg_encoded = self.rppg_encoder(rppg)
        voice_encoded = self.voice_encoder(voice)

        # Cross-modal Attention
        if self.use_attention:
            # 모든 모달리티를 시퀀스로 결합
            modalities = torch.stack(
                [big5_encoded, cmi_encoded, rppg_encoded, voice_encoded], dim=1
            )
            attended, attention_weights = self.cross_attention(
                modalities, modalities, modalities
            )
            attended = self.attention_norm(attended.mean(dim=1))  # 평균 풀링
        else:
            attended = torch.zeros_like(big5_encoded)

        # Multi-scale Feature Fusion
        multi_scale_features = []
        for fusion_layer in self.multi_scale_fusion:
            multi_scale_features.append(fusion_layer(big5_encoded))

        # Residual Connections
        if self.use_residual:
            residual_out = big5_encoded
            for residual_layer in self.residual_layers:
                residual_out = residual_out + residual_layer(residual_out)
        else:
            residual_out = big5_encoded

        # Skip Connections
        if self.use_skip_connections:
            skip_out = big5_encoded
            for skip_layer in self.skip_connections:
                skip_out = skip_out + skip_layer(skip_out)
        else:
            skip_out = big5_encoded

        # 동적 가중치 계산
        dynamic_weights = self.weight_attention(big5_encoded)

        # 가중치 기반 융합
        weighted_fused = (
            dynamic_weights[:, 0:1] * big5_encoded
            + dynamic_weights[:, 1:2] * cmi_encoded
            + dynamic_weights[:, 2:3] * rppg_encoded
            + dynamic_weights[:, 3:4] * voice_encoded
        )

        # 최종 융합
        if self.use_attention:
            final_input = torch.cat(
                [
                    big5_encoded,
                    cmi_encoded,
                    rppg_encoded,
                    voice_encoded,
                    attended,
                    weighted_fused,
                ],
                dim=1,
            )
        else:
            final_input = torch.cat(
                [big5_encoded, cmi_encoded, rppg_encoded, voice_encoded], dim=1
            )

        # 최종 예측
        output = self.final_fusion(final_input)

        return output, dynamic_weights


class UltimateSmartTrainer:
    """최고 성능 스마트 훈련 시스템"""

    def __init__(self, project_id: str = "persona-diary-service"):
        self.project_id = project_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scalers = {}
        self.model = None
        self.training_history = []

        print(f"Device: {self.device}")

    def generate_ultimate_data(self, n_samples: int = 15000) -> Tuple[Dict, np.ndarray]:
        """최고 품질 데이터 생성"""
        print(f"Generating ultimate data... ({n_samples} samples)")

        np.random.seed(42)

        # 더 현실적이고 복잡한 데이터 생성
        big5_data = np.random.normal(3.0, 1.3, (n_samples, 25))
        big5_data = np.clip(big5_data, 1.0, 5.0)

        cmi_data = np.random.normal(50, 22, (n_samples, 10))
        cmi_data = np.clip(cmi_data, 0, 100)

        rppg_data = np.random.normal(70, 18, (n_samples, 15))
        rppg_data = np.clip(rppg_data, 40, 120)

        voice_data = np.random.normal(200, 70, (n_samples, 20))
        voice_data = np.clip(voice_data, 50, 500)

        # 복잡한 비선형 관계
        big5_scores = {
            "EXT": big5_data[:, :5].mean(axis=1),
            "EST": big5_data[:, 5:10].mean(axis=1),
            "AGR": big5_data[:, 10:15].mean(axis=1),
            "CSN": big5_data[:, 15:20].mean(axis=1),
            "OPN": big5_data[:, 20:25].mean(axis=1),
        }

        # 고차원 상호작용
        targets = (
            big5_scores["EXT"] * 0.25
            + big5_scores["OPN"] * 0.2
            + (5 - big5_scores["EST"]) * 0.15
            + big5_scores["AGR"] * 0.15
            + big5_scores["CSN"] * 0.1
            + (cmi_data.mean(axis=1) / 100) * 0.08
            + (rppg_data.mean(axis=1) / 100) * 0.04
            + (voice_data.mean(axis=1) / 300) * 0.03
            +
            # 비선형 상호작용
            (big5_scores["EXT"] * big5_scores["OPN"]) * 0.05
            + (big5_scores["EST"] * big5_scores["AGR"]) * 0.03
            + np.random.normal(0, 0.15, n_samples)
        )

        # 1-10 스케일로 정규화
        targets = (targets - targets.min()) / (targets.max() - targets.min()) * 9 + 1

        multimodal_data = {
            "big5": big5_data,
            "cmi": cmi_data,
            "rppg": rppg_data,
            "voice": voice_data,
        }

        print(f"Ultimate data generated:")
        print(f"  Big5: {big5_data.shape}")
        print(f"  CMI: {cmi_data.shape}")
        print(f"  RPPG: {rppg_data.shape}")
        print(f"  Voice: {voice_data.shape}")
        print(f"  Targets: {targets.shape}")

        return multimodal_data, targets

    def prepare_data(
        self, multimodal_data: Dict, targets: np.ndarray
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """데이터 전처리"""
        print("Preprocessing ultimate data...")

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
            multimodal_data["big5"], targets_scaled, test_size=0.2, random_state=42
        )
        cmi_temp, cmi_test, _, _ = train_test_split(
            multimodal_data["cmi"], targets_scaled, test_size=0.2, random_state=42
        )
        rppg_temp, rppg_test, _, _ = train_test_split(
            multimodal_data["rppg"], targets_scaled, test_size=0.2, random_state=42
        )
        voice_temp, voice_test, _, _ = train_test_split(
            multimodal_data["voice"], targets_scaled, test_size=0.2, random_state=42
        )

        # 훈련/검증 분할
        big5_train, big5_val, y_train, y_val = train_test_split(
            big5_temp, y_temp, test_size=0.1, random_state=42
        )
        cmi_train, cmi_val, _, _ = train_test_split(
            cmi_temp, y_temp, test_size=0.1, random_state=42
        )
        rppg_train, rppg_val, _, _ = train_test_split(
            rppg_temp, y_temp, test_size=0.1, random_state=42
        )
        voice_train, voice_val, _, _ = train_test_split(
            voice_temp, y_temp, test_size=0.1, random_state=42
        )

        # DataLoader 생성
        train_dataset = SmartMultimodalDataset(
            big5_train, cmi_train, rppg_train, voice_train, y_train, augment=True
        )
        val_dataset = SmartMultimodalDataset(
            big5_val, cmi_val, rppg_val, voice_val, y_val, augment=False
        )
        test_dataset = SmartMultimodalDataset(
            big5_test, cmi_test, rppg_test, voice_test, y_test, augment=False
        )

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        print(f"Data split complete:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val: {len(val_dataset)} samples")
        print(f"  Test: {len(test_dataset)} samples")

        return train_loader, val_loader, test_loader

    def train_ultimate_model(
        self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 150
    ) -> Dict:
        """최고 성능 모델 훈련"""
        print(f"Training ultimate smart model... (Epochs: {epochs})")

        # 모델 초기화
        self.model = UltimateSmartMultimodalNet(
            big5_dim=25,
            cmi_dim=10,
            rppg_dim=15,
            voice_dim=20,
            hidden_dim=128,
            output_dim=1,
            dropout_rate=0.4,
            use_attention=True,
            use_residual=True,
            use_skip_connections=True,
        ).to(self.device)

        # 고급 옵티마이저 설정
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            self.model.parameters(), lr=0.001, weight_decay=1e-4, betas=(0.9, 0.999)
        )

        # 고급 스케줄러
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )

        best_val_loss = float("inf")
        patience_counter = 0
        early_stopping_patience = 25

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

                # Label Smoothing
                smooth_targets = targets * 0.9 + 0.1 * targets.mean()
                loss = criterion(outputs.squeeze(), smooth_targets)

                # L2 정규화
                l2_reg = sum(p.pow(2.0).sum() for p in self.model.parameters())
                loss = loss + 1e-5 * l2_reg

                loss.backward()

                # 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()
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

            # 스케줄러 업데이트
            scheduler.step()

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
                torch.save(self.model.state_dict(), "ultimate_smart_model.pth")
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
                print(f"  Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}")
                print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # 최고 모델 로드
        self.model.load_state_dict(torch.load("ultimate_smart_model.pth"))

        print(f"Ultimate training complete! Best validation loss: {best_val_loss:.4f}")

        return {
            "best_val_loss": best_val_loss,
            "total_epochs": epoch + 1,
            "training_history": self.training_history,
        }

    def evaluate_ultimate_model(self, test_loader: DataLoader) -> Dict:
        """최고 성능 모델 평가"""
        print("Evaluating ultimate model...")

        self.model.eval()
        test_predictions = []
        test_targets = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
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

        print(f"Ultimate model evaluation:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Correlation: {correlation:.4f}")

        return evaluation_results

    def run_ultimate_training(self, n_samples: int = 15000, epochs: int = 150) -> Dict:
        """최고 성능 훈련 실행"""
        print("ULTIMATE SMART MULTIMODAL TRAINING SYSTEM")
        print("=" * 60)

        # 1. 데이터 생성
        multimodal_data, targets = self.generate_ultimate_data(n_samples)

        # 2. 데이터 준비
        train_loader, val_loader, test_loader = self.prepare_data(
            multimodal_data, targets
        )

        # 3. 모델 훈련
        training_results = self.train_ultimate_model(train_loader, val_loader, epochs)

        # 4. 모델 평가
        evaluation_results = self.evaluate_ultimate_model(test_loader)

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
        with open("ultimate_smart_results.json", "w") as f:

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

        print(f"Ultimate training complete!")
        print(f"  Final R²: {evaluation_results['r2']:.4f}")
        print(f"  Final RMSE: {evaluation_results['rmse']:.4f}")

        return results


def main():
    """메인 실행 함수"""
    print("ULTIMATE SMART MULTIMODAL TRAINING SYSTEM")
    print("=" * 60)

    # 훈련 시스템 초기화
    trainer = UltimateSmartTrainer()

    # 최고 성능 훈련 실행
    results = trainer.run_ultimate_training(n_samples=15000, epochs=150)

    print("\nUltimate Training Summary:")
    print(f"  Device: {results['model_info']['device']}")
    print(f"  Samples: {results['model_info']['n_samples']}")
    print(f"  Epochs: {results['model_info']['epochs_trained']}")
    print(f"  Final R²: {results['evaluation_results']['r2']:.4f}")


if __name__ == "__main__":
    main()


