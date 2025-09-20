#!/usr/bin/env python3
"""
고급 멀티모달 훈련 시스템
- 딥러닝 기반 멀티모달 융합
- 실시간 학습 및 적응
- 고성능 모델 훈련
"""

import json
import os
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


class MultimodalDataset(Dataset):
    """멀티모달 데이터셋 클래스"""

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


class MultimodalFusionNet(nn.Module):
    """고급 멀티모달 융합 신경망 - 강화된 융합 알고리즘"""

    def __init__(
        self,
        big5_dim=25,
        cmi_dim=10,
        rppg_dim=15,
        voice_dim=20,
        hidden_dim=128,
        output_dim=1,
        dropout_rate=0.3,
        use_transformer=True,
        use_cross_attention=True,
    ):
        super(MultimodalFusionNet, self).__init__()

        self.use_transformer = use_transformer
        self.use_cross_attention = use_cross_attention

        # Debugging: Add print statement to verify use_cross_attention during initialization
        print(
            f"Initializing MultimodalFusionNet with use_cross_attention={use_cross_attention}"
        )

        # 각 모달리티별 고급 인코더 (모든 모달리티를 같은 차원으로)
        self.big5_encoder = self._create_advanced_encoder(
            big5_dim, hidden_dim // 2, dropout_rate
        )
        self.cmi_encoder = self._create_advanced_encoder(
            cmi_dim, hidden_dim // 2, dropout_rate
        )
        self.rppg_encoder = self._create_advanced_encoder(
            rppg_dim, hidden_dim // 2, dropout_rate
        )
        self.voice_encoder = self._create_advanced_encoder(
            voice_dim, hidden_dim // 2, dropout_rate
        )

        # 크로스 어텐션 메커니즘 (모달리티 간 상호작용)
        if use_cross_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim // 2,
                num_heads=8,
                dropout=dropout_rate,
                batch_first=True,
            )

            # 모달리티별 쿼리, 키, 밸류 변환 (모든 모달리티가 hidden_dim // 2로 통일)
            self.query_transform = nn.Linear(
                hidden_dim // 2, hidden_dim // 2
            )  # Big5: hidden_dim // 2
            self.key_transform = nn.Linear(
                hidden_dim // 2, hidden_dim // 2
            )  # CMI/RPPG/Voice: hidden_dim // 2
            self.value_transform = nn.Linear(
                hidden_dim // 2, hidden_dim // 2
            )  # CMI/RPPG/Voice: hidden_dim // 2

        # 트랜스포머 기반 융합 (선택적)
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim // 2,
                nhead=8,
                dim_feedforward=hidden_dim,
                dropout=dropout_rate,
                batch_first=True,
            )
            self.transformer_fusion = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 적응형 융합 레이어
        # final_fused 차원: basic_fused(4*64) + weighted_fused(64) + importance_weighted(64) + attended_features(64) + transformer_fused(64) = 512
        fusion_input_dim = 4 * (hidden_dim // 2) + 4 * (
            hidden_dim // 2
        )  # 256 + 256 = 512
        self.adaptive_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        # 동적 모달리티 가중치 학습
        self.modality_weights = nn.Parameter(torch.ones(4))
        self.weight_gate = nn.Sequential(
            nn.Linear(hidden_dim // 2, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.Softmax(dim=-1),
        )

        # 모달리티별 중요도 학습 (통일된 차원)
        self.importance_networks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim // 2, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid(),
                )
                for _ in range(4)
            ]
        )

    def _create_advanced_encoder(
        self, input_dim: int, output_dim: int, dropout_rate: float
    ) -> nn.Module:
        """고급 인코더 생성"""
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, big5, cmi, rppg, voice):
        """고급 멀티모달 융합 forward pass"""
        # 각 모달리티 인코딩
        big5_encoded = self.big5_encoder(big5)
        cmi_encoded = self.cmi_encoder(cmi)
        rppg_encoded = self.rppg_encoder(rppg)
        voice_encoded = self.voice_encoder(voice)

        # 모달리티별 중요도 계산
        importance_scores = []
        modalities = [big5_encoded, cmi_encoded, rppg_encoded, voice_encoded]

        for i, modality in enumerate(modalities):
            importance = self.importance_networks[i](modality)
            importance_scores.append(importance)

        # 동적 가중치 계산
        dynamic_weights = self.weight_gate(big5_encoded)  # Big5를 기준으로 가중치 계산

        # 크로스 어텐션 적용 (모달리티 간 상호작용)
        # Debugging: Add print statement to verify use_cross_attention
        print(f"use_cross_attention: {self.use_cross_attention}")
        if self.use_cross_attention:
            # Big5를 쿼리로 사용하여 다른 모달리티에 어텐션
            query = self.query_transform(big5_encoded).unsqueeze(1)

            # 다른 모달리티들을 키/밸류로 변환
            keys = []
            values = []
            for modality in [cmi_encoded, rppg_encoded, voice_encoded]:
                key = self.key_transform(modality).unsqueeze(1)
                value = self.value_transform(modality).unsqueeze(1)
                keys.append(key)
                values.append(value)

            key_tensor = torch.cat(keys, dim=1)
            value_tensor = torch.cat(values, dim=1)

            attended_features, cross_attention_weights = self.cross_attention(
                query, key_tensor, value_tensor
            )
            attended_features = attended_features.squeeze(1)
        else:
            attended_features = big5_encoded
            cross_attention_weights = None

        # 트랜스포머 기반 융합 (선택적)
        if self.use_transformer:
            # 모든 모달리티를 시퀀스로 결합
            modality_sequence = torch.stack(
                [big5_encoded, cmi_encoded, rppg_encoded, voice_encoded], dim=1
            )

            # 트랜스포머로 융합
            transformer_output = self.transformer_fusion(modality_sequence)

            # 평균 풀링으로 최종 표현 생성
            transformer_fused = transformer_output.mean(dim=1)
        else:
            transformer_fused = big5_encoded

        # 적응형 융합
        # 1. 기본 융합 (단순 연결)
        basic_fused = torch.cat(
            [big5_encoded, cmi_encoded, rppg_encoded, voice_encoded], dim=1
        )

        # 2. 가중치 기반 융합
        weighted_fused = (
            dynamic_weights[:, 0:1] * big5_encoded
            + dynamic_weights[:, 1:2] * cmi_encoded
            + dynamic_weights[:, 2:3] * rppg_encoded
            + dynamic_weights[:, 3:4] * voice_encoded
        )

        # 3. 중요도 기반 융합
        importance_weighted = (
            importance_scores[0] * big5_encoded
            + importance_scores[1] * cmi_encoded
            + importance_scores[2] * rppg_encoded
            + importance_scores[3] * voice_encoded
        )

        # 최종 융합 (여러 융합 방법 결합)
        final_fused = torch.cat(
            [
                basic_fused,
                weighted_fused,
                importance_weighted,
                attended_features,
                transformer_fused,
            ],
            dim=1,
        )

        # 적응형 융합 레이어로 최종 예측
        output = self.adaptive_fusion(final_fused)

        # 반환값 구성
        attention_info = {
            "cross_attention_weights": cross_attention_weights,
            "dynamic_weights": dynamic_weights,
            "importance_scores": importance_scores,
            "modality_weights": torch.softmax(self.modality_weights, dim=0),
        }

        return output, attention_info, dynamic_weights


class AdvancedMultimodalTrainer:
    """고급 멀티모달 훈련 시스템"""

    def __init__(self, project_id: str = "persona-diary-service"):
        self.project_id = project_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scalers = {}
        self.model = None
        self.training_history = []

        print(f"디바이스: {self.device}")

    def generate_synthetic_multimodal_data(
        self, n_samples: int = 10000
    ) -> Tuple[Dict, np.ndarray]:
        """합성 멀티모달 데이터 생성"""
        print(f"합성 멀티모달 데이터 생성 중... ({n_samples}개 샘플)")

        np.random.seed(42)

        # Big5 데이터 (25개 특성)
        big5_data = np.random.normal(3.5, 1.0, (n_samples, 25))
        big5_data = np.clip(big5_data, 1.0, 5.0)

        # CMI 데이터 (10개 특성)
        cmi_data = np.random.normal(50, 15, (n_samples, 10))
        cmi_data = np.clip(cmi_data, 0, 100)

        # RPPG 데이터 (15개 특성)
        rppg_data = np.random.normal(70, 10, (n_samples, 15))
        rppg_data = np.clip(rppg_data, 40, 120)

        # Voice 데이터 (20개 특성)
        voice_data = np.random.normal(200, 50, (n_samples, 20))
        voice_data = np.clip(voice_data, 50, 400)

        # 타겟 변수 생성 (만족도 점수)
        # Big5의 EXT, OPN이 높고 EST가 낮을수록 만족도 높음
        big5_scores = {
            "EXT": big5_data[:, :5].mean(axis=1),
            "EST": big5_data[:, 5:10].mean(axis=1),
            "AGR": big5_data[:, 10:15].mean(axis=1),
            "CSN": big5_data[:, 15:20].mean(axis=1),
            "OPN": big5_data[:, 20:25].mean(axis=1),
        }

        # 복합 타겟 변수
        targets = (
            big5_scores["EXT"] * 0.3
            + big5_scores["OPN"] * 0.25
            + (5 - big5_scores["EST"]) * 0.2
            + big5_scores["AGR"] * 0.15
            + big5_scores["CSN"] * 0.1
            + (cmi_data.mean(axis=1) / 100) * 0.1
            + (rppg_data.mean(axis=1) / 100) * 0.05
            + (voice_data.mean(axis=1) / 300) * 0.05
        )

        # 1-10 스케일로 정규화
        targets = (targets - targets.min()) / (targets.max() - targets.min()) * 9 + 1

        multimodal_data = {
            "big5": big5_data,
            "cmi": cmi_data,
            "rppg": rppg_data,
            "voice": voice_data,
        }

        print(f"데이터 생성 완료:")
        print(f"   Big5: {big5_data.shape}")
        print(f"   CMI: {cmi_data.shape}")
        print(f"   RPPG: {rppg_data.shape}")
        print(f"   Voice: {voice_data.shape}")
        print(f"   Targets: {targets.shape}")

        return multimodal_data, targets

    def prepare_data(
        self,
        multimodal_data: Dict,
        targets: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """데이터 전처리 및 DataLoader 생성"""
        print("데이터 전처리 중...")

        # 데이터 정규화
        for modality, data in multimodal_data.items():
            scaler = StandardScaler()
            multimodal_data[modality] = scaler.fit_transform(data)
            self.scalers[modality] = scaler

        # 타겟 정규화
        target_scaler = StandardScaler()
        targets_scaled = target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()
        self.scalers["target"] = target_scaler

        # 데이터 분할 (딕셔너리 형태로 처리)
        # 각 모달리티별로 분할
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
        train_dataset = MultimodalDataset(
            X_train["big5"], X_train["cmi"], X_train["rppg"], X_train["voice"], y_train
        )
        val_dataset = MultimodalDataset(
            X_val["big5"], X_val["cmi"], X_val["rppg"], X_val["voice"], y_val
        )
        test_dataset = MultimodalDataset(
            X_test["big5"], X_test["cmi"], X_test["rppg"], X_test["voice"], y_test
        )

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        print(f"데이터 분할 완료:")
        print(f"   훈련: {len(train_dataset)}개")
        print(f"   검증: {len(val_dataset)}개")
        print(f"   테스트: {len(test_dataset)}개")

        return train_loader, val_loader, test_loader

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        learning_rate: float = 0.001,
        use_cross_attention=True,
    ) -> Dict:
        """모델 훈련"""
        print(f"모델 훈련 시작 (Epochs: {epochs})")

        # 모델 초기화
        self.model = MultimodalFusionNet(
            big5_dim=25,
            cmi_dim=10,
            rppg_dim=15,
            voice_dim=20,
            hidden_dim=128,
            output_dim=1,
            dropout_rate=0.3,
            use_transformer=True,
            use_cross_attention=use_cross_attention,
        ).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
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
            train_predictions = []
            train_targets = []

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                big5 = batch["big5"].to(self.device)
                cmi = batch["cmi"].to(self.device)
                rppg = batch["rppg"].to(self.device)
                voice = batch["voice"].to(self.device)
                targets = batch["target"].to(self.device)

                optimizer.zero_grad()
                outputs, attention_info, dynamic_weights = self.model(
                    big5, cmi, rppg, voice
                )

                # 기본 손실
                basic_loss = criterion(outputs.squeeze(), targets)

                # 정규화 손실 (모달리티 가중치 균형)
                modality_weights = attention_info["modality_weights"]
                weight_entropy = -torch.sum(
                    modality_weights * torch.log(modality_weights + 1e-8)
                )
                weight_regularization = 0.01 * weight_entropy

                # 중요도 스코어 정규화
                importance_scores = attention_info["importance_scores"]
                importance_regularization = 0.001 * sum(
                    torch.mean(score) for score in importance_scores
                )

                # 총 손실
                total_loss = (
                    basic_loss + weight_regularization + importance_regularization
                )

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

                    outputs, attention_info, dynamic_weights = self.model(
                        big5, cmi, rppg, voice
                    )
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
                torch.save(self.model.state_dict(), "best_multimodal_model.pth")
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"조기 종료 (Epoch {epoch+1})")
                break

            # 진행 상황 출력
            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
                print(f"  Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}")
                print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # 최고 모델 로드
        self.model.load_state_dict(torch.load("best_multimodal_model.pth"))

        print(f"훈련 완료! 최고 검증 손실: {best_val_loss:.4f}")

        return {
            "best_val_loss": best_val_loss,
            "total_epochs": epoch + 1,
            "training_history": self.training_history,
        }

    def evaluate_model(self, test_loader: DataLoader) -> Dict:
        """모델 평가"""
        print("모델 평가 중...")

        self.model.eval()
        test_predictions = []
        test_targets = []
        attention_weights_list = []
        modality_weights_list = []

        with torch.no_grad():
            for batch in test_loader:
                big5 = batch["big5"].to(self.device)
                cmi = batch["cmi"].to(self.device)
                rppg = batch["rppg"].to(self.device)
                voice = batch["voice"].to(self.device)
                targets = batch["target"].to(self.device)

                outputs, attention_info, dynamic_weights = self.model(
                    big5, cmi, rppg, voice
                )

                test_predictions.extend(outputs.squeeze().cpu().numpy())
                test_targets.extend(targets.cpu().numpy())

                # 어텐션 정보 저장
                if attention_info["cross_attention_weights"] is not None:
                    attention_weights_list.append(
                        attention_info["cross_attention_weights"].cpu().numpy()
                    )
                modality_weights_list.append(dynamic_weights.cpu().numpy())

        # 메트릭 계산
        test_predictions = np.array(test_predictions)
        test_targets = np.array(test_targets)

        mse = np.mean((test_predictions - test_targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(test_predictions - test_targets))
        r2 = 1 - (
            np.sum((test_targets - test_predictions) ** 2)
            / np.sum((test_targets - np.mean(test_targets)) ** 2)
        )

        # 상관계수
        correlation = np.corrcoef(test_predictions, test_targets)[0, 1]

        # 모달리티 가중치 분석
        if modality_weights_list and len(modality_weights_list) > 0:
            try:
                # 모든 가중치를 동일한 형태로 변환
                weights_array = np.array(
                    [
                        w.flatten() if hasattr(w, "flatten") else w
                        for w in modality_weights_list
                    ]
                )
                avg_modality_weights = np.mean(weights_array, axis=0)
            except:
                avg_modality_weights = np.array(
                    [0.25, 0.25, 0.25, 0.25]
                )  # 기본 균등 가중치
        else:
            avg_modality_weights = np.array(
                [0.25, 0.25, 0.25, 0.25]
            )  # 기본 균등 가중치

        evaluation_results = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "correlation": correlation,
            "modality_weights": {
                "big5": float(avg_modality_weights[0]),
                "cmi": float(avg_modality_weights[1]),
                "rppg": float(avg_modality_weights[2]),
                "voice": float(avg_modality_weights[3]),
            },
            "predictions": test_predictions,
            "targets": test_targets,
        }

        print(f"평가 완료:")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   R²: {r2:.4f}")
        print(f"   상관계수: {correlation:.4f}")

        return evaluation_results

    def create_visualizations(
        self, evaluation_results: Dict, save_dir: str = "multimodal_results"
    ):
        """시각화 생성"""
        print("시각화 생성 중...")

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

        # 모달리티 가중치
        plt.subplot(2, 3, 5)
        modalities = list(evaluation_results["modality_weights"].keys())
        weights = list(evaluation_results["modality_weights"].values())
        plt.bar(modalities, weights)
        plt.title("Modality Weights")
        plt.ylabel("Weight")
        plt.xticks(rotation=45)

        # 잔차 분포
        plt.subplot(2, 3, 6)
        residuals = evaluation_results["predictions"] - evaluation_results["targets"]
        plt.hist(residuals, bins=30, alpha=0.7)
        plt.title("Residual Distribution")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(
            f"{save_dir}/multimodal_training_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print(f"시각화 저장: {save_dir}/multimodal_training_analysis.png")

    def run_comprehensive_training(
        self, n_samples: int = 10000, epochs: int = 100, use_cross_attention=True
    ) -> Dict:
        """종합 훈련 실행"""
        print("고급 멀티모달 훈련 시스템 시작")
        print("=" * 60)

        # 1. 데이터 생성
        multimodal_data, targets = self.generate_synthetic_multimodal_data(n_samples)

        # 2. 데이터 준비
        train_loader, val_loader, test_loader = self.prepare_data(
            multimodal_data, targets
        )

        # 3. 모델 훈련
        training_results = self.train_model(train_loader, val_loader, epochs)

        # 4. 모델 평가
        evaluation_results = self.evaluate_model(test_loader)

        # 5. 시각화 생성
        self.create_visualizations(evaluation_results)

        # 6. 결과 저장
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
        with open("multimodal_training_results.json", "w") as f:
            # NumPy 배열을 리스트로 변환
            json_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    json_results[key] = {}
                    for k, v in value.items():
                        if isinstance(v, np.ndarray):
                            json_results[key][k] = v.tolist()
                        else:
                            json_results[key][k] = v
                else:
                    json_results[key] = value
            json.dump(json_results, f, indent=2)

        print(f"멀티모달 훈련 완료!")
        print(f"   최종 R²: {evaluation_results['r2']:.4f}")
        print(f"   최종 RMSE: {evaluation_results['rmse']:.4f}")
        print(f"   모달리티 가중치: {evaluation_results['modality_weights']}")

        return results


def main():
    """메인 실행 함수"""
    print("고급 멀티모달 훈련 시스템")
    print("=" * 60)

    # 훈련 시스템 초기화
    trainer = AdvancedMultimodalTrainer()

    # 종합 훈련 실행
    results = trainer.run_comprehensive_training(
        n_samples=10000, epochs=100, use_cross_attention=False
    )

    print("\n훈련 결과 요약:")
    print(f"   디바이스: {results['model_info']['device']}")
    print(f"   샘플 수: {results['model_info']['n_samples']}")
    print(f"   훈련 에포크: {results['model_info']['epochs_trained']}")
    print(f"   최종 성능: R² = {results['evaluation_results']['r2']:.4f}")


if __name__ == "__main__":
    main()
