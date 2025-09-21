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


class ImprovedBigQueryDataLoader:
    """개선된 BigQuery 데이터 로더"""

    def __init__(self, project_id: str = "persona-diary-service"):
        self.project_id = project_id
        try:
            self.client = bigquery.Client(project=project_id)
            print(f"✅ BigQuery 클라이언트 초기화 완료: {project_id}")
        except Exception as e:
            print(f"⚠️ BigQuery 인증 실패: {str(e)}")
            print("대체 데이터 모드로 전환합니다.")
            self.client = None

    def load_competition_data(self, limit: int = 50000) -> dict:
        """대회 데이터 로드 (더 많은 샘플)"""
        if self.client is None:
            print("BigQuery 클라이언트가 없습니다. 대체 데이터를 생성합니다...")
            return self._generate_improved_fallback_data(limit)

        print(f"BigQuery에서 데이터 로딩 중... (제한: {limit}개)")

        try:
            # Big5 데이터
            big5_query = f"""
            SELECT 
                openness, conscientiousness, extraversion, agreeableness, neuroticism
            FROM `{self.project_id}.persona_diary.big5_scores`
            LIMIT {limit}
            """

            # CMI 데이터
            cmi_query = f"""
            SELECT 
                cmi_1, cmi_2, cmi_3, cmi_4, cmi_5, cmi_6, cmi_7, cmi_8, cmi_9, cmi_10,
                cmi_11, cmi_12, cmi_13, cmi_14, cmi_15, cmi_16, cmi_17, cmi_18, cmi_19, cmi_20
            FROM `{self.project_id}.persona_diary.cmi_scores`
            LIMIT {limit}
            """

            # RPPG 데이터
            rppg_query = f"""
            SELECT 
                rppg_1, rppg_2, rppg_3, rppg_4, rppg_5, rppg_6, rppg_7, rppg_8, rppg_9, rppg_10
            FROM `{self.project_id}.persona_diary.rppg_features`
            LIMIT {limit}
            """

            # Voice 데이터
            voice_query = f"""
            SELECT 
                voice_1, voice_2, voice_3, voice_4, voice_5, voice_6, voice_7, voice_8, voice_9, voice_10,
                voice_11, voice_12, voice_13, voice_14, voice_15, voice_16, voice_17, voice_18, voice_19, voice_20
            FROM `{self.project_id}.persona_diary.voice_features`
            LIMIT {limit}
            """

            # 타겟 데이터
            target_query = f"""
            SELECT 
                target_score
            FROM `{self.project_id}.persona_diary.target_scores`
            LIMIT {limit}
            """

            # 데이터 로드
            big5_df = self.client.query(big5_query).to_dataframe()
            cmi_df = self.client.query(cmi_query).to_dataframe()
            rppg_df = self.client.query(rppg_query).to_dataframe()
            voice_df = self.client.query(voice_query).to_dataframe()
            target_df = self.client.query(target_query).to_dataframe()

            print(f"✅ 데이터 로드 완료:")
            print(f"  Big5: {len(big5_df)}개")
            print(f"  CMI: {len(cmi_df)}개")
            print(f"  RPPG: {len(rppg_df)}개")
            print(f"  Voice: {len(voice_df)}개")
            print(f"  Target: {len(target_df)}개")

            return {
                "big5": big5_df.values,
                "cmi": cmi_df.values,
                "rppg": rppg_df.values,
                "voice": voice_df.values,
                "targets": target_df.values.flatten(),
            }

        except Exception as e:
            print(f"❌ BigQuery 데이터 로드 실패: {str(e)}")
            print("개선된 대체 데이터를 생성합니다...")
            return self._generate_improved_fallback_data(limit)

    def _generate_improved_fallback_data(self, limit: int) -> dict:
        """개선된 대체 데이터 생성 (더 현실적이고 다양한 데이터)"""
        print("📊 개선된 대체 데이터 생성 중...")

        # 더 현실적인 데이터 생성
        np.random.seed(42)

        # Big5 데이터 (더 다양한 분포)
        big5_data = np.random.beta(2, 2, (limit, 5))

        # CMI 데이터 (더 복잡한 패턴)
        cmi_data = np.random.beta(1.5, 1.5, (limit, 20))

        # RPPG 데이터 (생체신호 특성 반영)
        rppg_data = np.random.normal(0, 1, (limit, 10))

        # Voice 데이터 (음성 특성 반영)
        voice_data = np.random.normal(0, 1, (limit, 20))

        # 더 복잡한 타겟 생성 (비선형 관계 포함)
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
            + np.random.normal(0, 0.12, limit)  # 노이즈
        )

        # 타겟 정규화 (0-1 범위)
        targets = (targets - targets.min()) / (targets.max() - targets.min())

        print(f"✅ 개선된 대체 데이터 생성 완료: {limit}개 샘플")

        return {
            "big5": big5_data,
            "cmi": cmi_data,
            "rppg": rppg_data,
            "voice": voice_data,
            "targets": targets,
        }


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

        # 3. 다항식 특징 생성
        print("  📈 다항식 특징 생성...")
        big5_poly = self.poly_features.fit_transform(big5)

        # 4. 모든 특징 결합
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


class ImprovedEnsembleModel:
    """개선된 앙상블 모델"""

    def __init__(self, device="cpu"):
        self.device = device
        self.models = {}
        self.scalers = {}
        self.weights = {}

    def add_model(self, name: str, model, model_type: str, weight: float = 1.0):
        """모델 추가"""
        self.models[name] = {"model": model, "type": model_type, "weight": weight}

    def train_models(
        self, X_train: Dict, y_train: np.ndarray, X_val: Dict, y_val: np.ndarray
    ):
        """모든 모델 훈련"""
        print("🚀 개선된 앙상블 모델 훈련 시작!")

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

        print("✅ 개선된 앙상블 모델 훈련 완료!")

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
        patience = 10  # 더 긴 patience
        patience_counter = 0

        for epoch in range(50):  # 더 많은 에포크
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

        # 가중치 최적화 (R² 점수 기반, 음수 제외)
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


def create_improved_models(big5_dim=14, cmi_dim=29, rppg_dim=16, voice_dim=26):
    """개선된 모델 생성 (특징 엔지니어링된 차원에 맞춤)"""
    print("🎯 개선된 멀티모달 모델 생성!")

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    # 개선된 앙상블 모델 생성
    ensemble = ImprovedEnsembleModel(device)

    # 1. 신경망 모델들 추가 (특징 엔지니어링된 차원에 맞춤)
    model1 = TransferLearningMultimodalNet(
        big5_dim=big5_dim,
        cmi_dim=cmi_dim,
        rppg_dim=rppg_dim,
        voice_dim=voice_dim,
        hidden_dim=512,
        dropout_rate=0.3,
        use_pretrained=False,
    )
    ensemble.add_model("neural_net_large", model1, "neural_network", weight=1.0)

    model2 = TransferLearningMultimodalNet(
        big5_dim=big5_dim,
        cmi_dim=cmi_dim,
        rppg_dim=rppg_dim,
        voice_dim=voice_dim,
        hidden_dim=256,
        dropout_rate=0.4,
        use_pretrained=False,
    )
    ensemble.add_model("neural_net_medium", model2, "neural_network", weight=1.0)

    # 2. Scikit-learn 모델들 추가 (더 강력한 모델)
    ensemble.add_model(
        "random_forest_strong",
        RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
        "sklearn",
        weight=1.0,
    )
    ensemble.add_model(
        "gradient_boosting_strong",
        GradientBoostingRegressor(n_estimators=200, max_depth=8, random_state=42),
        "sklearn",
        weight=1.0,
    )
    ensemble.add_model("ridge_optimized", Ridge(alpha=0.1), "sklearn", weight=1.0)

    return ensemble


def test_improved_model():
    """개선된 모델 테스트"""
    print("🧪 개선된 모델 테스트 시작!")

    # 1. BigQuery 데이터 로드 (더 많은 샘플)
    data_loader = ImprovedBigQueryDataLoader()
    data = data_loader.load_competition_data(limit=20000)  # 2만개 샘플

    # 2. 특징 엔지니어링
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

    # 4. 개선된 앙상블 모델 생성 및 훈련 (특징 엔지니어링된 차원 전달)
    ensemble = create_improved_models(
        big5_dim=X_train["big5"].shape[1],
        cmi_dim=X_train["cmi"].shape[1],
        rppg_dim=X_train["rppg"].shape[1],
        voice_dim=X_train["voice"].shape[1],
    )
    ensemble.train_models(X_train, y_train, X_val, y_val)

    # 5. 테스트 예측
    test_predictions = ensemble.predict(X_test)

    # 6. 성능 평가
    test_r2 = r2_score(y_test, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    test_mae = mean_absolute_error(y_test, test_predictions)

    print("\n📊 개선된 모델 테스트 결과:")
    print(f"  테스트 R²: {test_r2:.4f}")
    print(f"  테스트 RMSE: {test_rmse:.4f}")
    print(f"  테스트 MAE: {test_mae:.4f}")

    # 7. 결과 저장
    results = {
        "improved_model_test": {
            "test_r2": test_r2,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "model_count": len(ensemble.models),
            "weights": ensemble.weights,
            "data_source": "BigQuery" if data_loader.client else "Enhanced_Simulated",
            "sample_count": len(enhanced_data["targets"]),
        }
    }

    with open("improved_model_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("✅ 결과 저장 완료!")
    print("📁 improved_model_test_results.json")

    return results


if __name__ == "__main__":
    results = test_improved_model()
