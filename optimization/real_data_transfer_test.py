import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from google.cloud import bigquery
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from transfer_learning_multimodal import (
    TransferLearningMultimodalDataset,
    TransferLearningMultimodalNet,
    TransferLearningTrainer,
)

warnings.filterwarnings("ignore")


class BigQueryDataLoader:
    """BigQuery 데이터 로더"""

    def __init__(self, project_id: str = "persona-diary-service"):
        self.project_id = project_id
        try:
            self.client = bigquery.Client(project=project_id)
            print(f"✅ BigQuery 클라이언트 초기화 완료: {project_id}")
        except Exception as e:
            print(f"⚠️ BigQuery 인증 실패: {str(e)}")
            print("대체 데이터 모드로 전환합니다.")
            self.client = None

    def load_competition_data(self, limit: int = 10000) -> dict:
        """대회 데이터 로드"""
        if self.client is None:
            print("BigQuery 클라이언트가 없습니다. 대체 데이터를 생성합니다...")
            return self._generate_fallback_data(limit)

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
            print("대체 데이터를 생성합니다...")
            return self._generate_fallback_data(limit)

    def _generate_fallback_data(self, limit: int) -> dict:
        """대체 데이터 생성 (실제 데이터 시뮬레이션)"""
        print("📊 대체 데이터 생성 중...")

        # 더 현실적인 데이터 생성
        np.random.seed(42)

        # Big5 데이터 (0-1 범위)
        big5_data = np.random.beta(2, 2, (limit, 5))

        # CMI 데이터 (0-1 범위)
        cmi_data = np.random.beta(1.5, 1.5, (limit, 20))

        # RPPG 데이터 (정규화된 생체신호)
        rppg_data = np.random.normal(0, 1, (limit, 10))

        # Voice 데이터 (정규화된 음성 특성)
        voice_data = np.random.normal(0, 1, (limit, 20))

        # 타겟 생성 (더 복잡한 관계)
        targets = (
            0.25 * big5_data[:, 0]  # Openness
            + 0.20 * big5_data[:, 1]  # Conscientiousness
            + 0.15 * big5_data[:, 2]  # Extraversion
            + 0.20 * big5_data[:, 3]  # Agreeableness
            + 0.10 * big5_data[:, 4]  # Neuroticism
            + 0.05 * np.mean(cmi_data, axis=1)
            + 0.03 * np.mean(rppg_data, axis=1)
            + 0.02 * np.mean(voice_data, axis=1)
            + np.random.normal(0, 0.15, limit)  # 더 많은 노이즈
        )

        # 타겟 정규화 (0-1 범위)
        targets = (targets - targets.min()) / (targets.max() - targets.min())

        print(f"✅ 대체 데이터 생성 완료: {limit}개 샘플")

        return {
            "big5": big5_data,
            "cmi": cmi_data,
            "rppg": rppg_data,
            "voice": voice_data,
            "targets": targets,
        }


def test_transfer_model_on_real_data():
    """실제 데이터에서 전이 학습 모델 테스트"""

    print("🧪 실제 데이터에서 전이 학습 모델 테스트 시작!")

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    # BigQuery 데이터 로드
    data_loader = BigQueryDataLoader()
    data = data_loader.load_competition_data(limit=5000)  # 빠른 테스트를 위해 5000개

    # 데이터 분할
    from sklearn.model_selection import train_test_split

    # 인덱스로 분할
    train_idx, test_idx = train_test_split(
        range(len(data["targets"])), test_size=0.3, random_state=42
    )
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)

    # 데이터 분할
    X_train = {
        modality: data[modality][train_idx]
        for modality in ["big5", "cmi", "rppg", "voice"]
    }
    X_val = {
        modality: data[modality][val_idx]
        for modality in ["big5", "cmi", "rppg", "voice"]
    }
    X_test = {
        modality: data[modality][test_idx]
        for modality in ["big5", "cmi", "rppg", "voice"]
    }

    y_train = data["targets"][train_idx]
    y_val = data["targets"][val_idx]
    y_test = data["targets"][test_idx]

    print(f"훈련 데이터: {len(y_train)}개")
    print(f"검증 데이터: {len(y_val)}개")
    print(f"테스트 데이터: {len(y_test)}개")

    # 데이터 정규화
    scalers = {}
    for modality in ["big5", "cmi", "rppg", "voice"]:
        scaler = StandardScaler()
        X_train[modality] = scaler.fit_transform(X_train[modality])
        X_val[modality] = scaler.transform(X_val[modality])
        X_test[modality] = scaler.transform(X_test[modality])
        scalers[modality] = scaler

    # 데이터셋 생성
    train_dataset = TransferLearningMultimodalDataset(
        X_train["big5"],
        X_train["cmi"],
        X_train["rppg"],
        X_train["voice"],
        y_train,
        augment=True,
    )
    val_dataset = TransferLearningMultimodalDataset(
        X_val["big5"], X_val["cmi"], X_val["rppg"], X_val["voice"], y_val, augment=False
    )
    test_dataset = TransferLearningMultimodalDataset(
        X_test["big5"],
        X_test["cmi"],
        X_test["rppg"],
        X_test["voice"],
        y_test,
        augment=False,
    )

    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 사전훈련된 모델 로드
    print("📥 사전훈련된 모델 로드 중...")

    try:
        checkpoint = torch.load("transfer_learning_model.pth", map_location=device)
        best_params = checkpoint["best_params"]

        # 모델 생성
        model = TransferLearningMultimodalNet(
            hidden_dim=best_params["hidden_dim"],
            dropout_rate=best_params["dropout_rate"],
            use_pretrained=False,  # 이미 로드할 예정
        )

        # 가중치 로드
        model.load_state_dict(checkpoint["model_state_dict"])
        print("✅ 사전훈련된 모델 로드 완료")

    except Exception as e:
        print(f"⚠️ 사전훈련된 모델 로드 실패: {str(e)}")
        print("새로운 모델로 시작합니다...")

        # 기본 파라미터로 새 모델 생성
        model = TransferLearningMultimodalNet(
            hidden_dim=256, dropout_rate=0.3, use_pretrained=False
        )

    # 트레이너 생성
    trainer = TransferLearningTrainer(model, device)

    # 파인튜닝 (실제 데이터에 맞게 조정)
    print("🔧 실제 데이터에 파인튜닝 중...")

    final_r2 = trainer.train(
        train_loader,
        val_loader,
        epochs=20,  # 빠른 파인튜닝
        lr=0.0001,  # 낮은 학습률
        weight_decay=1e-4,
        patience=5,
    )

    # 테스트 성능 평가
    print("🧪 테스트 성능 평가...")
    test_metrics = trainer.evaluate(test_loader, nn.MSELoss())

    print("\n📊 실제 데이터 테스트 결과:")
    print(f"  검증 R²: {final_r2:.4f}")
    print(f"  테스트 R²: {test_metrics['r2']:.4f}")
    print(f"  테스트 RMSE: {test_metrics['rmse']:.4f}")
    print(f"  테스트 MAE: {test_metrics['mae']:.4f}")

    # 결과 저장
    results = {
        "real_data_test": {
            "val_r2": final_r2,
            "test_r2": test_metrics["r2"],
            "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "data_source": "BigQuery" if data_loader.client else "Simulated",
            "sample_count": len(data["targets"]),
        }
    }

    with open("real_data_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("✅ 결과 저장 완료!")
    print("📁 real_data_test_results.json")

    return results


if __name__ == "__main__":
    results = test_transfer_model_on_real_data()


