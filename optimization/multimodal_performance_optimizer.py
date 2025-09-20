#!/usr/bin/env python3
"""
멀티모달 성능 최적화 시스템
- 모델 압축 및 최적화
- 추론 속도 향상
- 메모리 사용량 최적화
- 배치 처리 최적화
"""

import json
import multiprocessing as mp
import os
import threading
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import torch
import torch.jit
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


class OptimizedMultimodalDataset(Dataset):
    """최적화된 멀티모달 데이터셋"""

    def __init__(
        self,
        big5_data,
        cmi_data,
        rppg_data,
        voice_data,
        targets,
        use_caching=True,
        cache_size=1000,
    ):
        self.big5_data = torch.FloatTensor(big5_data)
        self.cmi_data = torch.FloatTensor(cmi_data)
        self.rppg_data = torch.FloatTensor(rppg_data)
        self.voice_data = torch.FloatTensor(voice_data)
        self.targets = torch.FloatTensor(targets)

        self.use_caching = use_caching
        self.cache_size = cache_size
        self.cache = {}

        # 메모리 최적화
        if use_caching:
            self._preload_cache()

    def _preload_cache(self):
        """자주 사용되는 데이터를 캐시에 미리 로드"""
        cache_indices = np.random.choice(
            len(self), min(self.cache_size, len(self)), replace=False
        )

        for idx in cache_indices:
            self.cache[idx] = {
                "big5": self.big5_data[idx],
                "cmi": self.cmi_data[idx],
                "rppg": self.rppg_data[idx],
                "voice": self.voice_data[idx],
                "target": self.targets[idx],
            }

    def __len__(self):
        return len(self.big5_data)

    def __getitem__(self, idx):
        if self.use_caching and idx in self.cache:
            return self.cache[idx]

        return {
            "big5": self.big5_data[idx],
            "cmi": self.cmi_data[idx],
            "rppg": self.rppg_data[idx],
            "voice": self.voice_data[idx],
            "target": self.targets[idx],
        }


class QuantizedMultimodalNet(nn.Module):
    """양자화된 멀티모달 네트워크"""

    def __init__(self, big5_dim=25, cmi_dim=10, rppg_dim=15, voice_dim=20):
        super(QuantizedMultimodalNet, self).__init__()

        # 경량화된 인코더들
        self.big5_encoder = nn.Sequential(
            nn.Linear(big5_dim, 32), nn.ReLU(inplace=True), nn.Linear(32, 16)
        )

        self.cmi_encoder = nn.Sequential(
            nn.Linear(cmi_dim, 16), nn.ReLU(inplace=True), nn.Linear(16, 8)
        )

        self.rppg_encoder = nn.Sequential(
            nn.Linear(rppg_dim, 16), nn.ReLU(inplace=True), nn.Linear(16, 8)
        )

        self.voice_encoder = nn.Sequential(
            nn.Linear(voice_dim, 16), nn.ReLU(inplace=True), nn.Linear(16, 8)
        )

        # 융합 레이어
        fusion_dim = 16 + 8 + 8 + 8
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

        # 모달리티 가중치
        self.modality_weights = nn.Parameter(torch.ones(4))

    def forward(self, big5, cmi, rppg, voice):
        # 각 모달리티 인코딩
        big5_encoded = self.big5_encoder(big5)
        cmi_encoded = self.cmi_encoder(cmi)
        rppg_encoded = self.rppg_encoder(rppg)
        voice_encoded = self.voice_encoder(voice)

        # 융합
        fused = torch.cat(
            [big5_encoded, cmi_encoded, rppg_encoded, voice_encoded], dim=1
        )
        output = self.fusion(fused)

        # 가중치 계산
        weights = torch.softmax(self.modality_weights, dim=0)

        return output, weights


class MultimodalPerformanceOptimizer:
    """멀티모달 성능 최적화 시스템"""

    def __init__(self, project_id: str = "persona-diary-service"):
        self.project_id = project_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scalers = {}
        self.optimized_model = None
        self.performance_metrics = {}

        print(f"🔧 성능 최적화 시스템 초기화")
        print(f"   디바이스: {self.device}")
        print(f"   CPU 코어 수: {mp.cpu_count()}")
        print(f"   메모리: {psutil.virtual_memory().total / (1024**3):.1f}GB")

    def generate_optimization_data(
        self, n_samples: int = 50000
    ) -> Tuple[Dict, np.ndarray]:
        """최적화용 대용량 데이터 생성"""
        print(f"📊 최적화용 데이터 생성 중... ({n_samples}개 샘플)")

        np.random.seed(42)

        # Big5 데이터
        big5_data = np.random.normal(3.5, 1.0, (n_samples, 25))
        big5_data = np.clip(big5_data, 1.0, 5.0)

        # CMI 데이터
        cmi_data = np.random.normal(50, 15, (n_samples, 10))
        cmi_data = np.clip(cmi_data, 0, 100)

        # RPPG 데이터
        rppg_data = np.random.normal(70, 10, (n_samples, 15))
        rppg_data = np.clip(rppg_data, 40, 120)

        # Voice 데이터
        voice_data = np.random.normal(200, 50, (n_samples, 20))
        voice_data = np.clip(voice_data, 50, 400)

        # 타겟 생성
        big5_scores = {
            "EXT": big5_data[:, :5].mean(axis=1),
            "EST": big5_data[:, 5:10].mean(axis=1),
            "AGR": big5_data[:, 10:15].mean(axis=1),
            "CSN": big5_data[:, 15:20].mean(axis=1),
            "OPN": big5_data[:, 20:25].mean(axis=1),
        }

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

        targets = (targets - targets.min()) / (targets.max() - targets.min()) * 9 + 1

        multimodal_data = {
            "big5": big5_data,
            "cmi": cmi_data,
            "rppg": rppg_data,
            "voice": voice_data,
        }

        print(f"✅ 데이터 생성 완료: {n_samples}개 샘플")
        return multimodal_data, targets

    def optimize_data_loading(
        self,
        multimodal_data: Dict,
        targets: np.ndarray,
        batch_size: int = 256,
        num_workers: int = 4,
    ) -> DataLoader:
        """데이터 로딩 최적화"""
        print(f"⚡ 데이터 로딩 최적화 중...")

        # 데이터 정규화
        for modality, data in multimodal_data.items():
            scaler = StandardScaler()
            multimodal_data[modality] = scaler.fit_transform(data)
            self.scalers[modality] = scaler

        # 타겟 정규화
        target_scaler = StandardScaler()
        targets_scaled = target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()
        self.scalers["target"] = target_scaler

        # 최적화된 데이터셋 생성
        dataset = OptimizedMultimodalDataset(
            multimodal_data["big5"],
            multimodal_data["cmi"],
            multimodal_data["rppg"],
            multimodal_data["voice"],
            targets_scaled,
            use_caching=True,
            cache_size=1000,
        )

        # 최적화된 DataLoader 생성
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == "cuda" else False,
            persistent_workers=True if num_workers > 0 else False,
        )

        print(f"✅ 데이터 로딩 최적화 완료:")
        print(f"   배치 크기: {batch_size}")
        print(f"   워커 수: {num_workers}")
        print(f"   캐시 사용: True")

        return dataloader

    def train_optimized_model(self, dataloader: DataLoader, epochs: int = 50) -> Dict:
        """최적화된 모델 훈련"""
        print(f"🚀 최적화된 모델 훈련 시작 (Epochs: {epochs})")

        # 모델 초기화
        self.optimized_model = QuantizedMultimodalNet().to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.optimized_model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=5, factor=0.5
        )

        training_history = []
        best_loss = float("inf")

        for epoch in range(epochs):
            start_time = time.time()

            # 훈련
            self.optimized_model.train()
            train_loss = 0.0
            train_predictions = []
            train_targets = []

            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                big5 = batch["big5"].to(self.device, non_blocking=True)
                cmi = batch["cmi"].to(self.device, non_blocking=True)
                rppg = batch["rppg"].to(self.device, non_blocking=True)
                voice = batch["voice"].to(self.device, non_blocking=True)
                targets = batch["target"].to(self.device, non_blocking=True)

                optimizer.zero_grad()
                outputs, weights = self.optimized_model(big5, cmi, rppg, voice)
                loss = criterion(outputs.squeeze(), targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_predictions.extend(outputs.squeeze().detach().cpu().numpy())
                train_targets.extend(targets.detach().cpu().numpy())

            # 평균 손실 계산
            train_loss /= len(dataloader)
            train_rmse = np.sqrt(
                np.mean((np.array(train_predictions) - np.array(train_targets)) ** 2)
            )

            # 학습률 스케줄링
            scheduler.step(train_loss)

            # 히스토리 저장
            epoch_time = time.time() - start_time
            training_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_rmse": train_rmse,
                    "epoch_time": epoch_time,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

            # 최고 모델 저장
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(
                    self.optimized_model.state_dict(), "best_optimized_model.pth"
                )

            # 진행 상황 출력
            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}: Loss={train_loss:.4f}, RMSE={train_rmse:.4f}, Time={epoch_time:.2f}s"
                )

        # 최고 모델 로드
        self.optimized_model.load_state_dict(torch.load("best_optimized_model.pth"))

        print(f"✅ 훈련 완료! 최고 손실: {best_loss:.4f}")

        return {
            "best_loss": best_loss,
            "total_epochs": epochs,
            "training_history": training_history,
        }

    def optimize_inference_speed(self, test_dataloader: DataLoader) -> Dict:
        """추론 속도 최적화"""
        print("⚡ 추론 속도 최적화 중...")

        self.optimized_model.eval()

        # 1. 기본 추론 속도 측정
        start_time = time.time()
        predictions = []
        targets = []

        with torch.no_grad():
            for batch in test_dataloader:
                big5 = batch["big5"].to(self.device, non_blocking=True)
                cmi = batch["cmi"].to(self.device, non_blocking=True)
                rppg = batch["rppg"].to(self.device, non_blocking=True)
                voice = batch["voice"].to(self.device, non_blocking=True)
                target = batch["target"].to(self.device, non_blocking=True)

                outputs, weights = self.optimized_model(big5, cmi, rppg, voice)
                predictions.extend(outputs.squeeze().cpu().numpy())
                targets.extend(target.cpu().numpy())

        basic_time = time.time() - start_time
        basic_throughput = len(predictions) / basic_time

        # 2. TorchScript 최적화
        print("   TorchScript 최적화 중...")
        self.optimized_model.eval()

        # 예시 입력 생성
        example_big5 = torch.randn(1, 25).to(self.device)
        example_cmi = torch.randn(1, 10).to(self.device)
        example_rppg = torch.randn(1, 15).to(self.device)
        example_voice = torch.randn(1, 20).to(self.device)

        # TorchScript 모델 생성
        traced_model = torch.jit.trace(
            self.optimized_model,
            (example_big5, example_cmi, example_rppg, example_voice),
        )

        # TorchScript 모델로 추론 속도 측정
        start_time = time.time()
        predictions_ts = []

        with torch.no_grad():
            for batch in test_dataloader:
                big5 = batch["big5"].to(self.device, non_blocking=True)
                cmi = batch["cmi"].to(self.device, non_blocking=True)
                rppg = batch["rppg"].to(self.device, non_blocking=True)
                voice = batch["voice"].to(self.device, non_blocking=True)

                outputs, weights = traced_model(big5, cmi, rppg, voice)
                predictions_ts.extend(outputs.squeeze().cpu().numpy())

        torchscript_time = time.time() - start_time
        torchscript_throughput = len(predictions_ts) / torchscript_time

        # 3. 배치 크기 최적화
        print("   배치 크기 최적화 중...")
        batch_sizes = [32, 64, 128, 256, 512]
        batch_performance = {}

        for batch_size in batch_sizes:
            test_loader = DataLoader(
                test_dataloader.dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,  # 배치 크기 테스트 시 워커 비활성화
                pin_memory=False,
            )

            start_time = time.time()
            with torch.no_grad():
                for batch in test_loader:
                    big5 = batch["big5"].to(self.device)
                    cmi = batch["cmi"].to(self.device)
                    rppg = batch["rppg"].to(self.device)
                    voice = batch["voice"].to(self.device)

                    outputs, weights = traced_model(big5, cmi, rppg, voice)

            batch_time = time.time() - start_time
            batch_throughput = len(test_loader.dataset) / batch_time

            batch_performance[batch_size] = {
                "time": batch_time,
                "throughput": batch_throughput,
            }

        # 최적 배치 크기 찾기
        optimal_batch_size = max(
            batch_performance.keys(), key=lambda x: batch_performance[x]["throughput"]
        )

        # 4. 메모리 사용량 측정
        memory_usage = psutil.Process().memory_info().rss / (1024**2)  # MB

        optimization_results = {
            "basic_inference": {
                "time": basic_time,
                "throughput": basic_throughput,
                "samples_per_second": basic_throughput,
            },
            "torchscript_inference": {
                "time": torchscript_time,
                "throughput": torchscript_throughput,
                "samples_per_second": torchscript_throughput,
                "speedup": torchscript_throughput / basic_throughput,
            },
            "batch_optimization": {
                "optimal_batch_size": optimal_batch_size,
                "batch_performance": batch_performance,
            },
            "memory_usage_mb": memory_usage,
            "device": str(self.device),
        }

        print(f"✅ 추론 속도 최적화 완료:")
        print(f"   기본 추론: {basic_throughput:.1f} samples/sec")
        print(f"   TorchScript: {torchscript_throughput:.1f} samples/sec")
        print(f"   속도 향상: {torchscript_throughput/basic_throughput:.2f}x")
        print(f"   최적 배치 크기: {optimal_batch_size}")
        print(f"   메모리 사용량: {memory_usage:.1f}MB")

        return optimization_results

    def create_performance_report(
        self,
        training_results: Dict,
        optimization_results: Dict,
        save_path: str = "performance_optimization_report.png",
    ):
        """성능 최적화 보고서 생성"""
        print("📊 성능 최적화 보고서 생성 중...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. 훈련 히스토리
        history_df = pd.DataFrame(training_results["training_history"])

        axes[0, 0].plot(history_df["epoch"], history_df["train_loss"])
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(True)

        axes[0, 1].plot(history_df["epoch"], history_df["train_rmse"])
        axes[0, 1].set_title("Training RMSE")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("RMSE")
        axes[0, 1].grid(True)

        axes[0, 2].plot(history_df["epoch"], history_df["epoch_time"])
        axes[0, 2].set_title("Epoch Time")
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("Time (s)")
        axes[0, 2].grid(True)

        # 2. 추론 성능 비교
        methods = ["Basic", "TorchScript"]
        throughputs = [
            optimization_results["basic_inference"]["throughput"],
            optimization_results["torchscript_inference"]["throughput"],
        ]

        axes[1, 0].bar(methods, throughputs)
        axes[1, 0].set_title("Inference Throughput Comparison")
        axes[1, 0].set_ylabel("Samples/Second")
        for i, v in enumerate(throughputs):
            axes[1, 0].text(
                i, v + max(throughputs) * 0.01, f"{v:.1f}", ha="center", va="bottom"
            )

        # 3. 배치 크기 최적화
        batch_sizes = list(
            optimization_results["batch_optimization"]["batch_performance"].keys()
        )
        batch_throughputs = [
            optimization_results["batch_optimization"]["batch_performance"][bs][
                "throughput"
            ]
            for bs in batch_sizes
        ]

        axes[1, 1].plot(batch_sizes, batch_throughputs, "o-")
        axes[1, 1].set_title("Batch Size Optimization")
        axes[1, 1].set_xlabel("Batch Size")
        axes[1, 1].set_ylabel("Throughput (samples/sec)")
        axes[1, 1].grid(True)

        # 최적 배치 크기 표시
        optimal_bs = optimization_results["batch_optimization"]["optimal_batch_size"]
        optimal_throughput = optimization_results["batch_optimization"][
            "batch_performance"
        ][optimal_bs]["throughput"]
        axes[1, 1].axvline(x=optimal_bs, color="red", linestyle="--", alpha=0.7)
        axes[1, 1].text(
            optimal_bs,
            optimal_throughput,
            f"Optimal: {optimal_bs}",
            ha="center",
            va="bottom",
            color="red",
        )

        # 4. 성능 요약
        axes[1, 2].axis("off")
        summary_text = f"""
        Performance Optimization Summary
        
        Device: {optimization_results['device']}
        Memory Usage: {optimization_results['memory_usage_mb']:.1f} MB
        
        Inference Performance:
        • Basic: {optimization_results['basic_inference']['throughput']:.1f} samples/sec
        • TorchScript: {optimization_results['torchscript_inference']['throughput']:.1f} samples/sec
        • Speedup: {optimization_results['torchscript_inference']['speedup']:.2f}x
        
        Optimal Configuration:
        • Batch Size: {optimal_bs}
        • Best Throughput: {optimal_throughput:.1f} samples/sec
        
        Training Performance:
        • Total Epochs: {training_results['total_epochs']}
        • Best Loss: {training_results['best_loss']:.4f}
        • Avg Epoch Time: {history_df['epoch_time'].mean():.2f}s
        """

        axes[1, 2].text(
            0.1,
            0.9,
            summary_text,
            transform=axes[1, 2].transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✅ 성능 최적화 보고서 저장: {save_path}")

    def run_comprehensive_optimization(
        self, n_samples: int = 50000, epochs: int = 50
    ) -> Dict:
        """종합 성능 최적화 실행"""
        print("🚀 종합 성능 최적화 시작")
        print("=" * 60)

        # 1. 데이터 생성
        multimodal_data, targets = self.generate_optimization_data(n_samples)

        # 2. 데이터 로딩 최적화
        dataloader = self.optimize_data_loading(multimodal_data, targets)

        # 3. 모델 훈련
        training_results = self.train_optimized_model(dataloader, epochs)

        # 4. 추론 속도 최적화
        optimization_results = self.optimize_inference_speed(dataloader)

        # 5. 성능 보고서 생성
        self.create_performance_report(training_results, optimization_results)

        # 6. 결과 저장
        results = {
            "training_results": training_results,
            "optimization_results": optimization_results,
            "system_info": {
                "device": str(self.device),
                "n_samples": n_samples,
                "epochs": epochs,
                "cpu_cores": mp.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / (1024**3),
            },
        }

        # JSON으로 저장
        with open("multimodal_optimization_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n🎉 종합 성능 최적화 완료!")
        print(
            f"   최종 추론 속도: {optimization_results['torchscript_inference']['throughput']:.1f} samples/sec"
        )
        print(
            f"   속도 향상: {optimization_results['torchscript_inference']['speedup']:.2f}x"
        )
        print(
            f"   최적 배치 크기: {optimization_results['batch_optimization']['optimal_batch_size']}"
        )

        return results


def main():
    """메인 실행 함수"""
    print("🧠 멀티모달 성능 최적화 시스템")
    print("=" * 60)

    # 최적화 시스템 초기화
    optimizer = MultimodalPerformanceOptimizer()

    # 종합 최적화 실행
    results = optimizer.run_comprehensive_optimization(n_samples=50000, epochs=50)

    print("\n🎯 최적화 결과 요약:")
    print(f"   디바이스: {results['system_info']['device']}")
    print(f"   샘플 수: {results['system_info']['n_samples']}")
    print(
        f"   최종 성능: {results['optimization_results']['torchscript_inference']['throughput']:.1f} samples/sec"
    )


if __name__ == "__main__":
    main()
