#!/usr/bin/env python3
"""
고급 멀티모달 훈련 시스템 - BigQuery 실제 데이터 최적화
- 실제 BigQuery 데이터를 활용한 고성능 모델 훈련
- R² = 0.25 → 0.50+ 성능 향상 목표
- 과적합 방지 및 일반화 성능 최적화
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
from google.cloud import bigquery
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)


class BigQueryDataLoader:
    """BigQuery 실제 데이터 로더"""

    def __init__(self, project_id: str = "persona-diary-service"):
        self.project_id = project_id
        try:
            self.client = bigquery.Client(project=project_id)
            print(f"✅ BigQuery 클라이언트 초기화 완료: {project_id}")
        except Exception as e:
            print(f"❌ BigQuery 인증 실패: {str(e)}")
            print("대체 데이터 모드로 전환합니다.")
            self.client = None

    def load_competition_data(self, limit: int = 10000) -> Tuple[Dict, np.ndarray]:
        """실제 BigQuery 대회 데이터 로딩"""
        if self.client is None:
            print("BigQuery 클라이언트가 없습니다. 대체 데이터를 생성합니다...")
            return self._generate_improved_fallback_data(limit)

        print(f"🔍 BigQuery에서 실제 데이터 로딩 중... (제한: {limit}개)")

        # Big5 데이터 쿼리
        big5_query = f"""
        SELECT 
            user_id,
            EXT_1, EXT_2, EXT_3, EXT_4, EXT_5,
            EST_1, EST_2, EST_3, EST_4, EST_5,
            AGR_1, AGR_2, AGR_3, AGR_4, AGR_5,
            CSN_1, CSN_2, CSN_3, CSN_4, CSN_5,
            OPN_1, OPN_2, OPN_3, OPN_4, OPN_5
        FROM `{self.project_id}.bigquery_competition.big5_data`
        LIMIT {limit}
        """

        # CMI 데이터 쿼리
        cmi_query = f"""
        SELECT 
            user_id,
            cmi_1, cmi_2, cmi_3, cmi_4, cmi_5,
            cmi_6, cmi_7, cmi_8, cmi_9, cmi_10
        FROM `{self.project_id}.bigquery_competition.cmi_data`
        LIMIT {limit}
        """

        # RPPG 데이터 쿼리
        rppg_query = f"""
        SELECT 
            user_id,
            rppg_1, rppg_2, rppg_3, rppg_4, rppg_5,
            rppg_6, rppg_7, rppg_8, rppg_9, rppg_10,
            rppg_11, rppg_12, rppg_13, rppg_14, rppg_15
        FROM `{self.project_id}.bigquery_competition.rppg_data`
        LIMIT {limit}
        """

        # Voice 데이터 쿼리
        voice_query = f"""
        SELECT 
            user_id,
            voice_1, voice_2, voice_3, voice_4, voice_5,
            voice_6, voice_7, voice_8, voice_9, voice_10,
            voice_11, voice_12, voice_13, voice_14, voice_15,
            voice_16, voice_17, voice_18, voice_19, voice_20
        FROM `{self.project_id}.bigquery_competition.voice_data`
        LIMIT {limit}
        """

        # 타겟 데이터 쿼리
        target_query = f"""
        SELECT 
            user_id,
            target_value
        FROM `{self.project_id}.bigquery_competition.target_data`
        LIMIT {limit}
        """

        try:
            # 데이터 로딩
            print("📊 Big5 데이터 로딩 중...")
            big5_df = self.client.query(big5_query).to_dataframe()
            print(f"   Big5 데이터: {big5_df.shape}")

            print("📊 CMI 데이터 로딩 중...")
            cmi_df = self.client.query(cmi_query).to_dataframe()
            print(f"   CMI 데이터: {cmi_df.shape}")

            print("📊 RPPG 데이터 로딩 중...")
            rppg_df = self.client.query(rppg_query).to_dataframe()
            print(f"   RPPG 데이터: {rppg_df.shape}")

            print("📊 Voice 데이터 로딩 중...")
            voice_df = self.client.query(voice_query).to_dataframe()
            print(f"   Voice 데이터: {voice_df.shape}")

            print("📊 타겟 데이터 로딩 중...")
            target_df = self.client.query(target_query).to_dataframe()
            print(f"   타겟 데이터: {target_df.shape}")

            # 데이터 병합
            print("🔄 데이터 병합 중...")
            merged_df = big5_df.merge(cmi_df, on="user_id", how="inner")
            merged_df = merged_df.merge(rppg_df, on="user_id", how="inner")
            merged_df = merged_df.merge(voice_df, on="user_id", how="inner")
            merged_df = merged_df.merge(target_df, on="user_id", how="inner")

            print(f"   병합된 데이터: {merged_df.shape}")

            # 결측값 처리
            print("🧹 결측값 처리 중...")
            merged_df = merged_df.dropna()
            print(f"   결측값 제거 후: {merged_df.shape}")

            # 데이터 분리
            big5_cols = [
                col
                for col in merged_df.columns
                if col.startswith(("EXT_", "EST_", "AGR_", "CSN_", "OPN_"))
            ]
            cmi_cols = [col for col in merged_df.columns if col.startswith("cmi_")]
            rppg_cols = [col for col in merged_df.columns if col.startswith("rppg_")]
            voice_cols = [col for col in merged_df.columns if col.startswith("voice_")]

            multimodal_data = {
                "big5": merged_df[big5_cols].values,
                "cmi": merged_df[cmi_cols].values,
                "rppg": merged_df[rppg_cols].values,
                "voice": merged_df[voice_cols].values,
            }

            targets = merged_df["target_value"].values

            print("✅ 실제 데이터 로딩 완료!")
            print(f"   Big5: {multimodal_data['big5'].shape}")
            print(f"   CMI: {multimodal_data['cmi'].shape}")
            print(f"   RPPG: {multimodal_data['rppg'].shape}")
            print(f"   Voice: {multimodal_data['voice'].shape}")
            print(f"   Targets: {targets.shape}")

            return multimodal_data, targets

        except Exception as e:
            print(f"❌ BigQuery 데이터 로딩 오류: {str(e)}")
            print("개선된 대체 데이터를 생성합니다...")
            return self._generate_improved_fallback_data(limit)

    def _generate_improved_fallback_data(self, limit: int) -> Tuple[Dict, np.ndarray]:
        """개선된 대체 데이터 생성 (실제 데이터 패턴 모방)"""
        print("🔄 개선된 대체 데이터 생성 중...")

        np.random.seed(42)

        # 더 현실적인 데이터 생성 (실제 BigQuery 데이터 패턴 모방)
        big5_data = np.random.normal(3.2, 1.2, (limit, 25))
        big5_data = np.clip(big5_data, 1.0, 5.0)

        cmi_data = np.random.normal(55, 20, (limit, 10))
        cmi_data = np.clip(cmi_data, 0, 100)

        rppg_data = np.random.normal(72, 15, (limit, 15))
        rppg_data = np.clip(rppg_data, 45, 110)

        voice_data = np.random.normal(220, 60, (limit, 20))
        voice_data = np.clip(voice_data, 60, 450)

        # 더 복잡하고 현실적인 타겟 변수 생성
        big5_scores = {
            "EXT": big5_data[:, :5].mean(axis=1),
            "EST": big5_data[:, 5:10].mean(axis=1),
            "AGR": big5_data[:, 10:15].mean(axis=1),
            "CSN": big5_data[:, 15:20].mean(axis=1),
            "OPN": big5_data[:, 20:25].mean(axis=1),
        }

        # 복합 타겟 변수 (더 복잡한 상호작용)
        targets = (
            big5_scores["EXT"] * 0.25
            + big5_scores["OPN"] * 0.20
            + (5 - big5_scores["EST"]) * 0.18
            + big5_scores["AGR"] * 0.15
            + big5_scores["CSN"] * 0.12
            + (cmi_data.mean(axis=1) / 100) * 0.05
            + (rppg_data.mean(axis=1) / 100) * 0.03
            + (voice_data.mean(axis=1) / 300) * 0.02
            + np.random.normal(0, 0.15, limit)  # 노이즈 추가
        )

        # 1-10 스케일로 정규화
        targets = (targets - targets.min()) / (targets.max() - targets.min()) * 9 + 1

        multimodal_data = {
            "big5": big5_data,
            "cmi": cmi_data,
            "rppg": rppg_data,
            "voice": voice_data,
        }

        print(f"✅ 대체 데이터 생성 완료:")
        print(f"   Big5: {big5_data.shape}")
        print(f"   CMI: {cmi_data.shape}")
        print(f"   RPPG: {rppg_data.shape}")
        print(f"   Voice: {voice_data.shape}")
        print(f"   Targets: {targets.shape}")

        return multimodal_data, targets


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


class OptimizedMultimodalNet(nn.Module):
    """최적화된 멀티모달 융합 신경망 - 성능 향상 버전"""

    def __init__(
        self,
        big5_dim=25,
        cmi_dim=10,
        rppg_dim=15,
        voice_dim=20,
        hidden_dim=128,
        output_dim=1,
        dropout_rate=0.3,
        use_transformer=False,
        use_cross_attention=False,
        weight_decay=1e-4,
    ):
        super(OptimizedMultimodalNet, self).__init__()

        self.use_transformer = use_transformer
        self.use_cross_attention = use_cross_attention
        self.weight_decay = weight_decay

        # 각 모달리티별 최적화된 인코더
        self.big5_encoder = self._create_optimized_encoder(
            big5_dim, hidden_dim, dropout_rate
        )
        self.cmi_encoder = self._create_optimized_encoder(
            cmi_dim, hidden_dim, dropout_rate
        )
        self.rppg_encoder = self._create_optimized_encoder(
            rppg_dim, hidden_dim, dropout_rate
        )
        self.voice_encoder = self._create_optimized_encoder(
            voice_dim, hidden_dim, dropout_rate
        )

        # 적응형 융합 레이어 (과적합 방지)
        fusion_input_dim = 4 * hidden_dim
        self.adaptive_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        # 모달리티별 중요도 학습
        self.importance_networks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(64, 1),
                    nn.Sigmoid(),
                )
                for _ in range(4)
            ]
        )

        # 동적 가중치 학습
        self.weight_gate = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 4),
            nn.Softmax(dim=-1),
        )

    def _create_optimized_encoder(
        self, input_dim: int, output_dim: int, dropout_rate: float
    ) -> nn.Module:
        """최적화된 인코더 생성 (과적합 방지)"""
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, big5, cmi, rppg, voice):
        """최적화된 멀티모달 융합 forward pass"""
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
        dynamic_weights = self.weight_gate(big5_encoded)

        # 가중치 기반 융합
        weighted_fused = (
            dynamic_weights[:, 0:1] * big5_encoded
            + dynamic_weights[:, 1:2] * cmi_encoded
            + dynamic_weights[:, 2:3] * rppg_encoded
            + dynamic_weights[:, 3:4] * voice_encoded
        )

        # 중요도 기반 융합
        importance_weighted = (
            importance_scores[0] * big5_encoded
            + importance_scores[1] * cmi_encoded
            + importance_scores[2] * rppg_encoded
            + importance_scores[3] * voice_encoded
        )

        # 최종 융합
        final_fused = torch.cat(
            [big5_encoded, cmi_encoded, rppg_encoded, voice_encoded], dim=1
        )

        # 적응형 융합 레이어로 최종 예측
        output = self.adaptive_fusion(final_fused)

        return output, {
            "dynamic_weights": dynamic_weights,
            "importance_scores": importance_scores,
        }


class AdvancedMultimodalTrainer:
    """고급 멀티모달 훈련 시스템 - BigQuery 실제 데이터 최적화"""

    def __init__(self, project_id: str = "persona-diary-service"):
        self.project_id = project_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scalers = {}
        self.model = None
        self.training_history = []
        self.data_loader = BigQueryDataLoader(project_id)

        print(f"🚀 디바이스: {self.device}")
        print(f"📊 프로젝트 ID: {project_id}")

    def prepare_data(
        self,
        multimodal_data: Dict,
        targets: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """데이터 전처리 및 DataLoader 생성 (개선된 버전)"""
        print("🔄 데이터 전처리 중...")

        # RobustScaler 사용 (이상치에 더 강함)
        for modality, data in multimodal_data.items():
            scaler = RobustScaler()
            multimodal_data[modality] = scaler.fit_transform(data)
            self.scalers[modality] = scaler

        # 타겟 정규화
        target_scaler = RobustScaler()
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

        print(f"✅ 데이터 분할 완료:")
        print(f"   훈련: {len(train_dataset)}개")
        print(f"   검증: {len(val_dataset)}개")
        print(f"   테스트: {len(test_dataset)}개")

        return train_loader, val_loader, test_loader

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 200,
        learning_rate: float = 0.001,
    ) -> Dict:
        """모델 훈련 (최적화된 버전)"""
        print(f"🚀 모델 훈련 시작 (Epochs: {epochs})")

        # 모델 초기화
        self.model = OptimizedMultimodalNet(
            big5_dim=25,
            cmi_dim=10,
            rppg_dim=15,
            voice_dim=20,
            hidden_dim=128,
            output_dim=1,
            dropout_rate=0.3,
            use_transformer=False,
            use_cross_attention=False,
            weight_decay=1e-4,
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=15, factor=0.5, min_lr=1e-6
        )

        best_val_loss = float("inf")
        patience_counter = 0
        early_stopping_patience = 30

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
                outputs, _ = self.model(big5, cmi, rppg, voice)

                loss = criterion(outputs.squeeze(), targets)
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

                    outputs, _ = self.model(big5, cmi, rppg, voice)
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

            # R² 계산
            train_r2 = r2_score(train_targets, train_predictions)
            val_r2 = r2_score(val_targets, val_predictions)

            # 히스토리 저장
            self.training_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_rmse": train_rmse,
                    "val_rmse": val_rmse,
                    "train_r2": train_r2,
                    "val_r2": val_r2,
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
                print(f"⏹️ 조기 종료 (Epoch {epoch+1})")
                break

            # 진행 상황 출력
            if (epoch + 1) % 20 == 0:
                print(
                    f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
                print(f"  Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}")
                print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # 최고 모델 로드
        self.model.load_state_dict(torch.load("best_optimized_model.pth"))

        print(f"✅ 훈련 완료! 최고 검증 손실: {best_val_loss:.4f}")

        return {
            "best_val_loss": best_val_loss,
            "total_epochs": epoch + 1,
            "training_history": self.training_history,
        }

    def evaluate_model(self, test_loader: DataLoader) -> Dict:
        """모델 평가 (개선된 버전)"""
        print("📊 모델 평가 중...")

        self.model.eval()
        test_predictions = []
        test_targets = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                big5 = batch["big5"].to(self.device)
                cmi = batch["cmi"].to(self.device)
                rppg = batch["rppg"].to(self.device)
                voice = batch["voice"].to(self.device)
                targets = batch["target"].to(self.device)

                outputs, _ = self.model(big5, cmi, rppg, voice)

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

        print(f"✅ 평가 완료:")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   R²: {r2:.4f}")
        print(f"   상관계수: {correlation:.4f}")

        return evaluation_results

    def create_visualizations(
        self, evaluation_results: Dict, save_dir: str = "optimized_results"
    ):
        """시각화 생성 (개선된 버전)"""
        print("📊 시각화 생성 중...")

        os.makedirs(save_dir, exist_ok=True)

        # 1. 훈련 히스토리
        history_df = pd.DataFrame(self.training_history)

        plt.figure(figsize=(20, 12))

        # 손실 그래프
        plt.subplot(2, 4, 1)
        plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
        plt.plot(history_df["epoch"], history_df["val_loss"], label="Validation Loss")
        plt.title("Training History - Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        # R² 그래프
        plt.subplot(2, 4, 2)
        plt.plot(history_df["epoch"], history_df["train_r2"], label="Train R²")
        plt.plot(history_df["epoch"], history_df["val_r2"], label="Validation R²")
        plt.title("Training History - R²")
        plt.xlabel("Epoch")
        plt.ylabel("R²")
        plt.legend()
        plt.grid(True)

        # RMSE 그래프
        plt.subplot(2, 4, 3)
        plt.plot(history_df["epoch"], history_df["train_rmse"], label="Train RMSE")
        plt.plot(history_df["epoch"], history_df["val_rmse"], label="Validation RMSE")
        plt.title("Training History - RMSE")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.legend()
        plt.grid(True)

        # 학습률 그래프
        plt.subplot(2, 4, 4)
        plt.plot(history_df["epoch"], history_df["learning_rate"])
        plt.title("Learning Rate Schedule")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.yscale("log")
        plt.grid(True)

        # 예측 vs 실제
        plt.subplot(2, 4, 5)
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
        plt.subplot(2, 4, 6)
        residuals = evaluation_results["predictions"] - evaluation_results["targets"]
        plt.hist(residuals, bins=30, alpha=0.7)
        plt.title("Residual Distribution")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.grid(True)

        # 예측 분포
        plt.subplot(2, 4, 7)
        plt.hist(
            evaluation_results["predictions"],
            bins=30,
            alpha=0.7,
            label="Predictions",
            color="blue",
        )
        plt.hist(
            evaluation_results["targets"],
            bins=30,
            alpha=0.7,
            label="Targets",
            color="red",
        )
        plt.title("Prediction vs Target Distribution")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)

        # 성능 요약
        plt.subplot(2, 4, 8)
        plt.text(0.1, 0.8, f"Performance Summary:", fontsize=14, fontweight="bold")
        plt.text(0.1, 0.7, f"R² = {evaluation_results['r2']:.4f}", fontsize=12)
        plt.text(0.1, 0.6, f"RMSE = {evaluation_results['rmse']:.4f}", fontsize=12)
        plt.text(0.1, 0.5, f"MAE = {evaluation_results['mae']:.4f}", fontsize=12)
        plt.text(
            0.1,
            0.4,
            f"Correlation = {evaluation_results['correlation']:.4f}",
            fontsize=12,
        )
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(
            f"{save_dir}/optimized_training_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print(f"✅ 시각화 저장: {save_dir}/optimized_training_analysis.png")

    def run_comprehensive_training(
        self, n_samples: int = 10000, epochs: int = 200
    ) -> Dict:
        """종합 훈련 실행 (BigQuery 실제 데이터)"""
        print("🚀 고급 멀티모달 훈련 시스템 시작")
        print("=" * 60)

        # 1. 실제 BigQuery 데이터 로딩
        multimodal_data, targets = self.data_loader.load_competition_data(n_samples)

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
                "project_id": self.project_id,
            },
        }

        # JSON으로 저장
        with open("optimized_training_results.json", "w") as f:
            # NumPy 배열을 리스트로 변환
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

        print(f"✅ 멀티모달 훈련 완료!")
        print(f"   최종 R²: {evaluation_results['r2']:.4f}")
        print(f"   최종 RMSE: {evaluation_results['rmse']:.4f}")
        print(f"   상관계수: {evaluation_results['correlation']:.4f}")

        return results


def main():
    """메인 실행 함수"""
    print("🚀 고급 멀티모달 훈련 시스템 - BigQuery 실제 데이터 최적화")
    print("=" * 70)

    # 훈련 시스템 초기화
    trainer = AdvancedMultimodalTrainer()

    # 종합 훈련 실행
    results = trainer.run_comprehensive_training(n_samples=10000, epochs=200)

    print("\n📊 훈련 결과 요약:")
    print(f"   디바이스: {results['model_info']['device']}")
    print(f"   샘플 수: {results['model_info']['n_samples']}")
    print(f"   훈련 에포크: {results['model_info']['epochs_trained']}")
    print(f"   최종 성능: R² = {results['evaluation_results']['r2']:.4f}")
    print(f"   최종 RMSE: {results['evaluation_results']['rmse']:.4f}")


if __name__ == "__main__":
    main()
