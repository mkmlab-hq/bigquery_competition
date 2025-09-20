#!/usr/bin/env python3
"""
실시간 멀티모달 학습 시스템
- 온라인 학습 및 적응
- 실시간 성능 모니터링
- 동적 모델 업데이트
"""

import json
import os
import queue
import threading
import time
import warnings
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


class OnlineMultimodalLearner:
    """고급 온라인 멀티모달 학습 시스템 - 실시간 적응 및 최적화"""

    def __init__(
        self,
        big5_dim=25,
        cmi_dim=10,
        rppg_dim=15,
        voice_dim=20,
        learning_rate=0.001,
        buffer_size=1000,
        update_frequency=10,
        use_adaptive_lr=True,
        use_uncertainty_estimation=True,
        use_concept_drift_detection=True,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.update_frequency = update_frequency
        self.use_adaptive_lr = use_adaptive_lr
        self.use_uncertainty_estimation = use_uncertainty_estimation
        self.use_concept_drift_detection = use_concept_drift_detection

        # 적응형 학습률 관리
        self.adaptive_lr_scheduler = None
        self.lr_decay_factor = 0.95
        self.lr_patience = 5

        # 불확실성 추정
        self.uncertainty_threshold = 0.1
        self.uncertainty_history = deque(maxlen=100)

        # 컨셉 드리프트 탐지
        self.drift_detector = None
        self.drift_threshold = 0.05
        self.drift_history = deque(maxlen=200)

        # 성능 모니터링
        self.performance_history = deque(maxlen=1000)
        self.alert_threshold = 0.1

        # 데이터 버퍼
        self.data_buffer = deque(maxlen=buffer_size)
        self.target_buffer = deque(maxlen=buffer_size)

        # 모델 초기화
        self.model = self._create_model(big5_dim, cmi_dim, rppg_dim, voice_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # 스케일러
        self.scalers = {
            "big5": StandardScaler(),
            "cmi": StandardScaler(),
            "rppg": StandardScaler(),
            "voice": StandardScaler(),
            "target": StandardScaler(),
        }

        # 성능 모니터링
        self.performance_history = []
        self.update_count = 0
        self.is_training = False

        # 스레드 안전을 위한 락
        self.lock = threading.Lock()

        print("Real-Time Multimodal Learning System")
        print("=" * 60)
        print(f"   디바이스: {self.device}")
        print(f"   버퍼 크기: {buffer_size}")
        print(f"   업데이트 주기: {update_frequency}")

    def _create_model(self, big5_dim, cmi_dim, rppg_dim, voice_dim):
        """경량화된 모델 생성"""

        class LightweightMultimodalNet(nn.Module):
            def __init__(self):
                super().__init__()

                # 각 모달리티별 인코더 (경량화)
                self.big5_encoder = nn.Linear(big5_dim, 64)
                self.cmi_encoder = nn.Linear(cmi_dim, 32)
                self.rppg_encoder = nn.Linear(rppg_dim, 32)
                self.voice_encoder = nn.Linear(voice_dim, 32)

                # 융합 레이어
                fusion_dim = 64 + 32 + 32 + 32
                self.fusion = nn.Sequential(
                    nn.Linear(fusion_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 1),
                )

                # 모달리티 가중치
                self.modality_weights = nn.Parameter(torch.ones(4))

            def forward(self, big5, cmi, rppg, voice):
                # 각 모달리티 인코딩
                big5_encoded = torch.relu(self.big5_encoder(big5))
                cmi_encoded = torch.relu(self.cmi_encoder(cmi))
                rppg_encoded = torch.relu(self.rppg_encoder(rppg))
                voice_encoded = torch.relu(self.voice_encoder(voice))

                # 융합
                fused = torch.cat(
                    [big5_encoded, cmi_encoded, rppg_encoded, voice_encoded], dim=1
                )
                output = self.fusion(fused)

                # 가중치 계산
                weights = torch.softmax(self.modality_weights, dim=0)

                return output, weights

        return LightweightMultimodalNet().to(self.device)

    def add_data_point(self, big5_data, cmi_data, rppg_data, voice_data, target):
        """새로운 데이터 포인트 추가"""
        with self.lock:
            # 데이터 정규화
            big5_scaled = (
                self.scalers["big5"]
                .partial_fit(big5_data.reshape(1, -1))
                .transform(big5_data.reshape(1, -1))
            )
            cmi_scaled = (
                self.scalers["cmi"]
                .partial_fit(cmi_data.reshape(1, -1))
                .transform(cmi_data.reshape(1, -1))
            )
            rppg_scaled = (
                self.scalers["rppg"]
                .partial_fit(rppg_data.reshape(1, -1))
                .transform(rppg_data.reshape(1, -1))
            )
            voice_scaled = (
                self.scalers["voice"]
                .partial_fit(voice_data.reshape(1, -1))
                .transform(voice_data.reshape(1, -1))
            )
            target_scaled = (
                self.scalers["target"]
                .partial_fit(np.array([[target]]))
                .transform(np.array([[target]]))[0, 0]
            )

            # 버퍼에 추가
            self.data_buffer.append(
                {
                    "big5": big5_scaled[0],
                    "cmi": cmi_scaled[0],
                    "rppg": rppg_scaled[0],
                    "voice": voice_scaled[0],
                }
            )
            self.target_buffer.append(target_scaled)

            # 업데이트 주기 확인
            if len(self.data_buffer) >= self.update_frequency:
                self._update_model()

    def _update_model(self):
        """고급 모델 업데이트 - 적응형 학습 및 드리프트 탐지"""
        if len(self.data_buffer) < self.update_frequency:
            return

        self.is_training = True

        # 최근 데이터로 배치 생성
        recent_data = list(self.data_buffer)[-self.update_frequency :]
        recent_targets = list(self.target_buffer)[-self.update_frequency :]

        # 텐서로 변환
        big5_batch = torch.FloatTensor([d["big5"] for d in recent_data]).to(self.device)
        cmi_batch = torch.FloatTensor([d["cmi"] for d in recent_data]).to(self.device)
        rppg_batch = torch.FloatTensor([d["rppg"] for d in recent_data]).to(self.device)
        voice_batch = torch.FloatTensor([d["voice"] for d in recent_data]).to(
            self.device
        )
        targets_batch = torch.FloatTensor(recent_targets).to(self.device)

        # 컨셉 드리프트 탐지
        if self.use_concept_drift_detection:
            drift_detected = self._detect_concept_drift(big5_batch, targets_batch)
            if drift_detected:
                self._handle_concept_drift()

        # 불확실성 추정
        uncertainty = 0.0
        if self.use_uncertainty_estimation:
            uncertainty = self._estimate_uncertainty(
                big5_batch, cmi_batch, rppg_batch, voice_batch
            )
            self.uncertainty_history.append(uncertainty)

            # 불확실성이 높으면 학습률 조정
            if uncertainty > self.uncertainty_threshold:
                self.learning_rate *= 1.1  # 학습률 증가
            else:
                self.learning_rate *= 0.99  # 학습률 감소

        # 적응형 학습률 적용
        if self.use_adaptive_lr:
            self._update_adaptive_learning_rate()

        # 훈련
        self.model.train()
        self.optimizer.zero_grad()

        outputs, weights = self.model(big5_batch, cmi_batch, rppg_batch, voice_batch)
        loss = self.criterion(outputs.squeeze(), targets_batch)

        # 정규화 손실 추가
        l2_reg = 0.001 * sum(p.pow(2.0).sum() for p in self.model.parameters())
        total_loss = loss + l2_reg

        total_loss.backward()

        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        # 성능 기록 및 모니터링
        with torch.no_grad():
            predictions = outputs.squeeze().cpu().numpy()
            targets_np = targets_batch.cpu().numpy()

            mse = mean_squared_error(targets_np, predictions)
            mae = mean_absolute_error(targets_np, predictions)

            self.performance_history.append(
                {
                    "update_count": self.update_count,
                    "mse": mse,
                    "mae": mae,
                    "loss": loss.item(),
                    "modality_weights": weights.cpu().numpy().tolist(),
                    "uncertainty": uncertainty,
                    "learning_rate": self.learning_rate,
                }
            )

            # 성능 알림 체크
            self._check_performance_alerts(mae)

        self.update_count += 1
        self.is_training = False

    def _detect_concept_drift(self, big5_batch, targets_batch):
        """컨셉 드리프트 탐지"""
        if len(self.drift_history) < 50:
            self.drift_history.append(
                (big5_batch.cpu().numpy(), targets_batch.cpu().numpy())
            )
            return False

        # 최근 데이터와 이전 데이터 비교
        recent_data = np.array([x[0] for x in list(self.drift_history)[-10:]])
        older_data = np.array([x[0] for x in list(self.drift_history)[-50:-10]])

        # 분포 변화 측정 (간단한 통계적 테스트)
        recent_mean = np.mean(recent_data, axis=0)
        older_mean = np.mean(older_data, axis=0)

        drift_score = np.linalg.norm(recent_mean - older_mean)

        # 새로운 데이터 추가
        self.drift_history.append(
            (big5_batch.cpu().numpy(), targets_batch.cpu().numpy())
        )

        return drift_score > self.drift_threshold

    def _handle_concept_drift(self):
        """컨셉 드리프트 처리"""
        print("컨셉 드리프트 탐지됨 - 모델 재초기화")

        # 학습률 증가
        self.learning_rate *= 1.5

        # 모델 가중치 재초기화 (부분적)
        for layer in self.model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def _estimate_uncertainty(self, big5_batch, cmi_batch, rppg_batch, voice_batch):
        """불확실성 추정 (Monte Carlo Dropout)"""
        self.model.train()  # Dropout 활성화

        predictions = []
        for _ in range(10):  # 10번 샘플링
            with torch.no_grad():
                pred, _ = self.model(big5_batch, cmi_batch, rppg_batch, voice_batch)
                predictions.append(pred.cpu().numpy())

        predictions = np.array(predictions)
        uncertainty = np.std(predictions, axis=0).mean()

        return uncertainty

    def _update_adaptive_learning_rate(self):
        """적응형 학습률 업데이트"""
        if len(self.performance_history) < self.lr_patience:
            return

        recent_performance = [
            p["mae"] for p in list(self.performance_history)[-self.lr_patience :]
        ]
        older_performance = [
            p["mae"]
            for p in list(self.performance_history)[
                -self.lr_patience * 2 : -self.lr_patience
            ]
        ]

        if len(older_performance) > 0:
            recent_avg = np.mean(recent_performance)
            older_avg = np.mean(older_performance)

            # 성능이 개선되지 않으면 학습률 감소
            if recent_avg >= older_avg:
                self.learning_rate *= self.lr_decay_factor
                self.learning_rate = max(self.learning_rate, 1e-6)  # 최소값 보장

    def _check_performance_alerts(self, current_mae):
        """성능 알림 체크"""
        if len(self.performance_history) < 10:
            return

        recent_mae = [p["mae"] for p in list(self.performance_history)[-10:]]
        overall_mae = [p["mae"] for p in list(self.performance_history)]

        recent_avg = np.mean(recent_mae)
        overall_avg = np.mean(overall_mae)

        # 성능이 크게 저하된 경우
        if recent_avg > overall_avg * (1 + self.alert_threshold):
            print(f"성능 저하 감지: {recent_avg:.4f} > {overall_avg:.4f}")

            # 자동 복구 시도
            self._attempt_recovery()

    def _attempt_recovery(self):
        """성능 복구 시도"""
        print("성능 복구 시도 중...")

        # 학습률 조정
        self.learning_rate *= 0.5

        # 모델 가중치 부분 재초기화
        for name, param in self.model.named_parameters():
            if "weight" in name and param.requires_grad:
                # 가중치에 작은 노이즈 추가
                noise = torch.randn_like(param) * 0.01
                param.data += noise

    def predict(self, big5_data, cmi_data, rppg_data, voice_data):
        """예측 수행"""
        with self.lock:
            self.model.eval()

            with torch.no_grad():
                # 데이터 정규화
                big5_scaled = self.scalers["big5"].transform(big5_data.reshape(1, -1))
                cmi_scaled = self.scalers["cmi"].transform(cmi_data.reshape(1, -1))
                rppg_scaled = self.scalers["rppg"].transform(rppg_data.reshape(1, -1))
                voice_scaled = self.scalers["voice"].transform(
                    voice_data.reshape(1, -1)
                )

                # 텐서로 변환
                big5_tensor = torch.FloatTensor(big5_scaled).to(self.device)
                cmi_tensor = torch.FloatTensor(cmi_scaled).to(self.device)
                rppg_tensor = torch.FloatTensor(rppg_scaled).to(self.device)
                voice_tensor = torch.FloatTensor(voice_scaled).to(self.device)

                # 예측
                output, weights = self.model(
                    big5_tensor, cmi_tensor, rppg_tensor, voice_tensor
                )
                prediction = output.cpu().numpy()[0, 0]

                # 역정규화
                prediction_original = self.scalers["target"].inverse_transform(
                    [[prediction]]
                )[0, 0]

                return prediction_original, weights.cpu().numpy()

    def get_performance_metrics(self):
        """성능 메트릭 반환"""
        if not self.performance_history:
            return None

        recent_performance = self.performance_history[-10:]  # 최근 10회

        return {
            "avg_mse": np.mean([p["mse"] for p in recent_performance]),
            "avg_mae": np.mean([p["mae"] for p in recent_performance]),
            "avg_loss": np.mean([p["loss"] for p in recent_performance]),
            "total_updates": self.update_count,
            "buffer_size": len(self.data_buffer),
            "is_training": self.is_training,
        }

    def save_model(self, filepath: str):
        """모델 저장"""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scalers": self.scalers,
                "performance_history": self.performance_history,
                "update_count": self.update_count,
            },
            filepath,
        )
        print(f"모델 저장: {filepath}")

    def load_model(self, filepath: str):
        """모델 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scalers = checkpoint["scalers"]
        self.performance_history = checkpoint["performance_history"]
        self.update_count = checkpoint["update_count"]
        print(f"모델 로드: {filepath}")


class RealTimeMultimodalSystem:
    """실시간 멀티모달 시스템"""

    def __init__(self, project_id: str = "persona-diary-service"):
        self.project_id = project_id
        self.learner = OnlineMultimodalLearner()
        self.data_queue = queue.Queue()
        self.running = False
        self.worker_thread = None

        print("실시간 멀티모달 시스템 초기화")

    def start_learning_loop(self):
        """학습 루프 시작"""
        self.running = True
        self.worker_thread = threading.Thread(target=self._learning_worker)
        self.worker_thread.start()
        print("실시간 학습 루프 시작")

    def stop_learning_loop(self):
        """학습 루프 중지"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()
        print("실시간 학습 루프 중지")

    def _learning_worker(self):
        """학습 워커 스레드"""
        while self.running:
            try:
                # 큐에서 데이터 가져오기 (타임아웃 1초)
                data = self.data_queue.get(timeout=1.0)

                # 데이터 처리
                big5_data = data["big5"]
                cmi_data = data["cmi"]
                rppg_data = data["rppg"]
                voice_data = data["voice"]
                target = data["target"]

                # 학습에 추가
                self.learner.add_data_point(
                    big5_data, cmi_data, rppg_data, voice_data, target
                )

            except queue.Empty:
                continue
            except Exception as e:
                print(f"학습 워커 오류: {e}")

    def add_training_data(self, big5_data, cmi_data, rppg_data, voice_data, target):
        """훈련 데이터 추가"""
        self.data_queue.put(
            {
                "big5": big5_data,
                "cmi": cmi_data,
                "rppg": rppg_data,
                "voice": voice_data,
                "target": target,
            }
        )

    def predict(self, big5_data, cmi_data, rppg_data, voice_data):
        """실시간 예측"""
        return self.learner.predict(big5_data, cmi_data, rppg_data, voice_data)

    def get_system_status(self):
        """시스템 상태 반환"""
        metrics = self.learner.get_performance_metrics()
        return {
            "is_running": self.running,
            "queue_size": self.data_queue.qsize(),
            "performance": metrics,
            "learner_status": {
                "buffer_size": len(self.learner.data_buffer),
                "update_count": self.learner.update_count,
                "is_training": self.learner.is_training,
            },
        }

    def create_performance_dashboard(self, save_path: str = "realtime_performance.png"):
        """성능 대시보드 생성"""
        if not self.learner.performance_history:
            print("성능 데이터가 없습니다.")
            return

        history_df = pd.DataFrame(self.learner.performance_history)

        plt.figure(figsize=(15, 10))

        # MSE 추이
        plt.subplot(2, 3, 1)
        plt.plot(history_df["update_count"], history_df["mse"])
        plt.title("MSE Over Time")
        plt.xlabel("Update Count")
        plt.ylabel("MSE")
        plt.grid(True)

        # MAE 추이
        plt.subplot(2, 3, 2)
        plt.plot(history_df["update_count"], history_df["mae"])
        plt.title("MAE Over Time")
        plt.xlabel("Update Count")
        plt.ylabel("MAE")
        plt.grid(True)

        # Loss 추이
        plt.subplot(2, 3, 3)
        plt.plot(history_df["update_count"], history_df["loss"])
        plt.title("Loss Over Time")
        plt.xlabel("Update Count")
        plt.ylabel("Loss")
        plt.grid(True)

        # 모달리티 가중치 추이
        plt.subplot(2, 3, 4)
        weights_df = pd.DataFrame(
            history_df["modality_weights"].tolist(),
            columns=["Big5", "CMI", "RPPG", "Voice"],
        )
        for col in weights_df.columns:
            plt.plot(history_df["update_count"], weights_df[col], label=col)
        plt.title("Modality Weights Over Time")
        plt.xlabel("Update Count")
        plt.ylabel("Weight")
        plt.legend()
        plt.grid(True)

        # 최근 성능 히트맵
        plt.subplot(2, 3, 5)
        recent_data = history_df.tail(20)
        performance_matrix = recent_data[["mse", "mae", "loss"]].values
        sns.heatmap(
            performance_matrix.T,
            xticklabels=recent_data["update_count"],
            yticklabels=["MSE", "MAE", "Loss"],
            cmap="YlOrRd",
            annot=True,
            fmt=".4f",
        )
        plt.title("Recent Performance Heatmap")

        # 성능 분포
        plt.subplot(2, 3, 6)
        plt.hist(history_df["mse"], bins=20, alpha=0.7, label="MSE")
        plt.hist(history_df["mae"], bins=20, alpha=0.7, label="MAE")
        plt.title("Performance Distribution")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"성능 대시보드 저장: {save_path}")


def simulate_real_time_learning():
    """실시간 학습 시뮬레이션"""
    print("실시간 학습 시뮬레이션 시작")
    print("=" * 60)

    # 시스템 초기화
    system = RealTimeMultimodalSystem()
    system.start_learning_loop()

    # 시뮬레이션 데이터 생성
    np.random.seed(42)
    n_samples = 1000

    print(f"{n_samples}개 샘플로 시뮬레이션 시작...")

    for i in tqdm(range(n_samples)):
        # 시뮬레이션 데이터 생성
        big5_data = np.random.normal(3.5, 1.0, 25)
        cmi_data = np.random.normal(50, 15, 10)
        rppg_data = np.random.normal(70, 10, 15)
        voice_data = np.random.normal(200, 50, 20)

        # 타겟 생성 (복합 점수)
        target = (
            big5_data[:5].mean() * 0.3  # EXT
            + big5_data[20:25].mean() * 0.25  # OPN
            + (5 - big5_data[5:10].mean()) * 0.2  # EST (역)
            + big5_data[10:15].mean() * 0.15  # AGR
            + big5_data[15:20].mean() * 0.1  # CSN
            + cmi_data.mean() / 100 * 0.1
            + rppg_data.mean() / 100 * 0.05
            + voice_data.mean() / 300 * 0.05
        )

        # 데이터 추가
        system.add_training_data(big5_data, cmi_data, rppg_data, voice_data, target)

        # 주기적으로 상태 출력
        if (i + 1) % 100 == 0:
            status = system.get_system_status()
            print(f"\n진행 상황 ({i+1}/{n_samples}):")
            print(f"   큐 크기: {status['queue_size']}")
            print(f"   버퍼 크기: {status['learner_status']['buffer_size']}")
            print(f"   업데이트 횟수: {status['learner_status']['update_count']}")
            if status["performance"]:
                print(f"   평균 MSE: {status['performance']['avg_mse']:.4f}")
                print(f"   평균 MAE: {status['performance']['avg_mae']:.4f}")

    # 최종 성능 대시보드 생성
    system.create_performance_dashboard()

    # 시스템 중지
    system.stop_learning_loop()

    # 최종 예측 테스트
    print("\n최종 예측 테스트:")
    test_big5 = np.random.normal(3.5, 1.0, 25)
    test_cmi = np.random.normal(50, 15, 10)
    test_rppg = np.random.normal(70, 10, 15)
    test_voice = np.random.normal(200, 50, 20)

    prediction, weights = system.predict(test_big5, test_cmi, test_rppg, test_voice)
    print(f"   예측값: {prediction:.4f}")
    print(f"   모달리티 가중치: {weights}")

    print("\n실시간 학습 시뮬레이션 완료!")


def main():
    """메인 실행 함수"""
    print("실시간 멀티모달 학습 시스템")
    print("=" * 60)

    # 실시간 학습 시뮬레이션 실행
    simulate_real_time_learning()


if __name__ == "__main__":
    main()
