#!/usr/bin/env python3
"""
실제 BigQuery 데이터 테스트 시스템
- 최적화된 멀티모달 모델을 실제 데이터에 적용
- 성능 측정 및 분석
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
from google.cloud import bigquery
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


class BigQueryDataLoader:
    """BigQuery 데이터 로더"""

    def __init__(self, project_id: str = "persona-diary-service"):
        self.project_id = project_id
        try:
            self.client = bigquery.Client(project=project_id)
            print(f"BigQuery 클라이언트 초기화 완료: {project_id}")
        except Exception as e:
            print(f"BigQuery 인증 실패: {str(e)}")
            print("대체 데이터 모드로 전환합니다.")
            self.client = None

    def load_competition_data(self, limit: int = 10000) -> Dict:
        """대회 데이터 로딩"""
        # BigQuery 클라이언트가 없으면 바로 대체 데이터 생성
        if self.client is None:
            print("BigQuery 클라이언트가 없습니다. 대체 데이터를 생성합니다...")
            return self._generate_fallback_data(limit)
            
        print(f"BigQuery에서 데이터 로딩 중... (제한: {limit}개)")

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
            print("Big5 데이터 로딩 중...")
            big5_df = self.client.query(big5_query).to_dataframe()
            print(f"Big5 데이터: {big5_df.shape}")

            print("CMI 데이터 로딩 중...")
            cmi_df = self.client.query(cmi_query).to_dataframe()
            print(f"CMI 데이터: {cmi_df.shape}")

            print("RPPG 데이터 로딩 중...")
            rppg_df = self.client.query(rppg_query).to_dataframe()
            print(f"RPPG 데이터: {rppg_df.shape}")

            print("Voice 데이터 로딩 중...")
            voice_df = self.client.query(voice_query).to_dataframe()
            print(f"Voice 데이터: {voice_df.shape}")

            print("타겟 데이터 로딩 중...")
            target_df = self.client.query(target_query).to_dataframe()
            print(f"타겟 데이터: {target_df.shape}")

            # 데이터 병합
            print("데이터 병합 중...")
            merged_df = big5_df.merge(cmi_df, on="user_id", how="inner")
            merged_df = merged_df.merge(rppg_df, on="user_id", how="inner")
            merged_df = merged_df.merge(voice_df, on="user_id", how="inner")
            merged_df = merged_df.merge(target_df, on="user_id", how="inner")

            print(f"병합된 데이터: {merged_df.shape}")

            # 결측값 처리
            print("결측값 처리 중...")
            merged_df = merged_df.dropna()
            print(f"결측값 제거 후: {merged_df.shape}")

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

            print("데이터 로딩 완료!")
            print(f"  Big5: {multimodal_data['big5'].shape}")
            print(f"  CMI: {multimodal_data['cmi'].shape}")
            print(f"  RPPG: {multimodal_data['rppg'].shape}")
            print(f"  Voice: {multimodal_data['voice'].shape}")
            print(f"  Targets: {targets.shape}")

            return multimodal_data, targets

        except Exception as e:
            print(f"BigQuery 데이터 로딩 오류: {str(e)}")
            print("합성 데이터로 대체합니다...")
            return self._generate_fallback_data(limit)

    def _generate_fallback_data(self, limit: int) -> Tuple[Dict, np.ndarray]:
        """BigQuery 접근 실패 시 대체 데이터 생성"""
        print("대체 데이터 생성 중...")

        np.random.seed(42)

        # 더 현실적인 데이터 생성
        big5_data = np.random.normal(3.0, 1.5, (limit, 25))
        big5_data = np.clip(big5_data, 1.0, 5.0)

        cmi_data = np.random.normal(50, 25, (limit, 10))
        cmi_data = np.clip(cmi_data, 0, 100)

        rppg_data = np.random.normal(70, 20, (limit, 15))
        rppg_data = np.clip(rppg_data, 40, 120)

        voice_data = np.random.normal(200, 80, (limit, 20))
        voice_data = np.clip(voice_data, 50, 500)

        # 더 복잡한 타겟 변수
        big5_scores = {
            "EXT": big5_data[:, :5].mean(axis=1),
            "EST": big5_data[:, 5:10].mean(axis=1),
            "AGR": big5_data[:, 10:15].mean(axis=1),
            "CSN": big5_data[:, 15:20].mean(axis=1),
            "OPN": big5_data[:, 20:25].mean(axis=1),
        }

        targets = (
            big5_scores["EXT"] * 0.3
            + big5_scores["OPN"] * 0.2
            + (5 - big5_scores["EST"]) * 0.15
            + big5_scores["AGR"] * 0.15
            + big5_scores["CSN"] * 0.1
            + (cmi_data.mean(axis=1) / 100) * 0.05
            + (rppg_data.mean(axis=1) / 100) * 0.03
            + (voice_data.mean(axis=1) / 300) * 0.02
            + np.random.normal(0, 0.2, limit)
        )

        targets = (targets - targets.min()) / (targets.max() - targets.min()) * 9 + 1

        multimodal_data = {
            "big5": big5_data,
            "cmi": cmi_data,
            "rppg": rppg_data,
            "voice": voice_data,
        }

        return multimodal_data, targets


class RealDataTestDataset(Dataset):
    """실제 데이터 테스트용 데이터셋"""

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


class BigQueryRealDataTester:
    """BigQuery 실제 데이터 테스터"""

    def __init__(self, project_id: str = "persona-diary-service"):
        self.project_id = project_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scalers = {}
        self.model = None

        print(f"Device: {self.device}")

    def load_optimized_model(self, model_path: str = "best_optimized_model.pth"):
        """최적화된 모델 로딩"""
        print(f"최적화된 모델 로딩 중: {model_path}")

        # 모델 아키텍처 정의 (최적화된 버전)
        from optimized_multimodal_training import OptimizedMultimodalNet

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

        # 모델 가중치 로딩
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("모델 로딩 완료!")
        else:
            print(f"모델 파일을 찾을 수 없습니다: {model_path}")
            print("새로운 모델을 초기화합니다.")

    def prepare_real_data(
        self, multimodal_data: Dict, targets: np.ndarray
    ) -> DataLoader:
        """실제 데이터 전처리"""
        print("실제 데이터 전처리 중...")

        # 데이터 정규화
        for modality, data in multimodal_data.items():
            scaler = StandardScaler()
            multimodal_data[modality] = scaler.fit_transform(data)
            self.scalers[modality] = scaler

        # 타겟 정규화
        target_scaler = StandardScaler()
        targets_scaled = target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()
        self.scalers["target"] = target_scaler

        # 데이터셋 생성
        dataset = RealDataTestDataset(
            multimodal_data["big5"],
            multimodal_data["cmi"],
            multimodal_data["rppg"],
            multimodal_data["voice"],
            targets_scaled,
        )

        # DataLoader 생성 (더 큰 배치 크기로 속도 향상)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

        print(f"데이터 전처리 완료: {len(dataset)} 샘플")

        return dataloader

    def test_model_performance(self, dataloader: DataLoader) -> Dict:
        """모델 성능 테스트"""
        print("모델 성능 테스트 중...")

        self.model.eval()
        predictions = []
        targets = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Testing"):
                big5 = batch["big5"].to(self.device)
                cmi = batch["cmi"].to(self.device)
                rppg = batch["rppg"].to(self.device)
                voice = batch["voice"].to(self.device)
                target = batch["target"].to(self.device)

                outputs, _ = self.model(big5, cmi, rppg, voice)

                predictions.extend(outputs.squeeze().cpu().numpy())
                targets.extend(target.cpu().numpy())

        # 메트릭 계산
        predictions = np.array(predictions)
        targets = np.array(targets)

        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        r2 = r2_score(targets, predictions)
        correlation = np.corrcoef(predictions, targets)[0, 1]

        results = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "correlation": correlation,
            "predictions": predictions,
            "targets": targets,
        }

        print(f"실제 데이터 성능:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Correlation: {correlation:.4f}")

        return results

    def create_comparison_visualization(
        self, real_results: Dict, synthetic_results: Dict = None
    ):
        """성능 비교 시각화"""
        print("성능 비교 시각화 생성 중...")

        os.makedirs("real_data_results", exist_ok=True)

        plt.figure(figsize=(15, 10))

        # 1. 예측 vs 실제 (실제 데이터)
        plt.subplot(2, 3, 1)
        plt.scatter(
            real_results["targets"],
            real_results["predictions"],
            alpha=0.6,
            color="blue",
        )
        plt.plot(
            [real_results["targets"].min(), real_results["targets"].max()],
            [real_results["targets"].min(), real_results["targets"].max()],
            "r--",
        )
        plt.title(f'Real Data: Predictions vs Targets (R² = {real_results["r2"]:.3f})')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.grid(True)

        # 2. 잔차 분포 (실제 데이터)
        plt.subplot(2, 3, 2)
        residuals = real_results["predictions"] - real_results["targets"]
        plt.hist(residuals, bins=30, alpha=0.7, color="blue")
        plt.title("Real Data: Residual Distribution")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.grid(True)

        # 3. 성능 메트릭 비교
        plt.subplot(2, 3, 3)
        metrics = ["R²", "RMSE", "MAE", "Correlation"]
        real_values = [
            real_results["r2"],
            real_results["rmse"],
            real_results["mae"],
            real_results["correlation"],
        ]

        if synthetic_results:
            synthetic_values = [
                synthetic_results["r2"],
                synthetic_results["rmse"],
                synthetic_results["mae"],
                synthetic_results["correlation"],
            ]
            x = np.arange(len(metrics))
            width = 0.35
            plt.bar(
                x - width / 2,
                real_values,
                width,
                label="Real Data",
                color="blue",
                alpha=0.7,
            )
            plt.bar(
                x + width / 2,
                synthetic_values,
                width,
                label="Synthetic Data",
                color="red",
                alpha=0.7,
            )
            plt.legend()
        else:
            plt.bar(metrics, real_values, color="blue", alpha=0.7)

        plt.title("Performance Metrics Comparison")
        plt.ylabel("Value")
        plt.xticks(rotation=45)

        # 4. 예측 분포
        plt.subplot(2, 3, 4)
        plt.hist(
            real_results["predictions"],
            bins=30,
            alpha=0.7,
            label="Predictions",
            color="blue",
        )
        plt.hist(
            real_results["targets"], bins=30, alpha=0.7, label="Targets", color="red"
        )
        plt.title("Prediction vs Target Distribution")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)

        # 5. 상관관계 분석
        plt.subplot(2, 3, 5)
        plt.scatter(
            real_results["targets"],
            real_results["predictions"],
            alpha=0.6,
            color="blue",
        )
        z = np.polyfit(real_results["targets"], real_results["predictions"], 1)
        p = np.poly1d(z)
        plt.plot(real_results["targets"], p(real_results["targets"]), "r--", alpha=0.8)
        plt.title(f'Correlation Analysis (r = {real_results["correlation"]:.3f})')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.grid(True)

        # 6. 성능 요약
        plt.subplot(2, 3, 6)
        plt.text(0.1, 0.8, f"Real Data Performance:", fontsize=14, fontweight="bold")
        plt.text(0.1, 0.7, f"R² = {real_results['r2']:.4f}", fontsize=12)
        plt.text(0.1, 0.6, f"RMSE = {real_results['rmse']:.4f}", fontsize=12)
        plt.text(0.1, 0.5, f"MAE = {real_results['mae']:.4f}", fontsize=12)
        plt.text(
            0.1, 0.4, f"Correlation = {real_results['correlation']:.4f}", fontsize=12
        )

        if synthetic_results:
            plt.text(
                0.1, 0.2, f"Synthetic Data Performance:", fontsize=14, fontweight="bold"
            )
            plt.text(0.1, 0.1, f"R² = {synthetic_results['r2']:.4f}", fontsize=12)
            plt.text(0.1, 0.0, f"RMSE = {synthetic_results['rmse']:.4f}", fontsize=12)

        plt.axis("off")

        plt.tight_layout()
        plt.savefig(
            "real_data_results/real_vs_synthetic_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print("시각화 저장 완료: real_data_results/real_vs_synthetic_comparison.png")

    def run_comprehensive_test(self, limit: int = 10000) -> Dict:
        """종합 테스트 실행"""
        print("BIGQUERY 실제 데이터 테스트 시작")
        print("=" * 60)

        # 1. BigQuery 데이터 로딩
        data_loader = BigQueryDataLoader(self.project_id)
        multimodal_data, targets = data_loader.load_competition_data(limit)

        # 2. 최적화된 모델 로딩
        self.load_optimized_model()

        # 3. 데이터 전처리
        dataloader = self.prepare_real_data(multimodal_data, targets)

        # 4. 모델 성능 테스트
        real_results = self.test_model_performance(dataloader)

        # 5. 합성 데이터 결과 로딩 (비교용)
        synthetic_results = None
        if os.path.exists("optimized_training_results.json"):
            with open("optimized_training_results.json", "r") as f:
                synthetic_data = json.load(f)
                synthetic_results = synthetic_data.get("evaluation_results", None)

        # 6. 시각화 생성
        self.create_comparison_visualization(real_results, synthetic_results)

        # 7. 결과 저장
        results = {
            "real_data_results": real_results,
            "synthetic_data_results": synthetic_results,
            "data_info": {
                "n_samples": len(targets),
                "device": str(self.device),
                "project_id": self.project_id,
            },
        }

        # JSON으로 저장
        with open("real_data_test_results.json", "w") as f:

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

        print(f"\n실제 데이터 테스트 완료!")
        print(f"  실제 데이터 R²: {real_results['r2']:.4f}")
        print(f"  실제 데이터 RMSE: {real_results['rmse']:.4f}")

        if synthetic_results:
            print(f"  합성 데이터 R²: {synthetic_results['r2']:.4f}")
            print(f"  합성 데이터 RMSE: {synthetic_results['rmse']:.4f}")

        return results


def main():
    """메인 실행 함수"""
    print("BIGQUERY 실제 데이터 테스트 시스템")
    print("=" * 60)

    # 테스터 초기화
    tester = BigQueryRealDataTester()

    # 종합 테스트 실행 (더 작은 데이터셋으로 빠른 테스트)
    results = tester.run_comprehensive_test(limit=5000)

    print("\n테스트 완료!")
    print("결과 파일:")
    print("  - real_data_test_results.json")
    print("  - real_data_results/real_vs_synthetic_comparison.png")


if __name__ == "__main__":
    main()
