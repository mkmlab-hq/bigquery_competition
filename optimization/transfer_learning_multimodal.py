import json
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# from optuna.integration import PyTorchLightningPruningCallback  # 사용하지 않음
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


class TransferLearningMultimodalDataset(Dataset):
    """전이 학습용 멀티모달 데이터셋"""

    def __init__(
        self,
        big5_data,
        cmi_data,
        rppg_data,
        voice_data,
        targets,
        augment=False,
        mixup_alpha=0.2,
        cutmix_alpha=1.0,
        noise_std=0.1,
    ):
        self.big5_data = big5_data
        self.cmi_data = cmi_data
        self.rppg_data = rppg_data
        self.voice_data = voice_data
        self.targets = targets
        self.augment = augment
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.noise_std = noise_std
        self.training = False

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        big5 = torch.FloatTensor(self.big5_data[idx])
        cmi = torch.FloatTensor(self.cmi_data[idx])
        rppg = torch.FloatTensor(self.rppg_data[idx])
        voice = torch.FloatTensor(self.voice_data[idx])
        target = torch.FloatTensor([self.targets[idx]])

        # 데이터 증강 적용
        if self.augment and self.training:
            big5, cmi, rppg, voice, target = self._apply_augmentation(
                big5, cmi, rppg, voice, target
            )

        return {
            "big5": big5,
            "cmi": cmi,
            "rppg": rppg,
            "voice": voice,
            "target": target,
        }

    def _apply_augmentation(self, big5, cmi, rppg, voice, target):
        """고급 데이터 증강 적용"""

        # 1. 가우시안 노이즈 추가
        if np.random.random() < 0.3:
            noise_std = self.noise_std * np.random.uniform(0.5, 1.5)
            big5 += torch.randn_like(big5) * noise_std
            cmi += torch.randn_like(cmi) * noise_std
            rppg += torch.randn_like(rppg) * noise_std
            voice += torch.randn_like(voice) * noise_std

        # 2. Mixup 적용
        if np.random.random() < 0.2:
            big5, cmi, rppg, voice, target = self._apply_mixup(
                big5, cmi, rppg, voice, target
            )

        # 3. CutMix 적용
        if np.random.random() < 0.2:
            big5, cmi, rppg, voice, target = self._apply_cutmix(
                big5, cmi, rppg, voice, target
            )

        return big5, cmi, rppg, voice, target

    def _apply_mixup(self, big5, cmi, rppg, voice, target):
        """Mixup 데이터 증강"""
        alpha = self.mixup_alpha
        lam = np.random.beta(alpha, alpha)

        # 랜덤 샘플 선택
        idx = np.random.randint(0, len(self.targets))
        big5_mix = torch.FloatTensor(self.big5_data[idx])
        cmi_mix = torch.FloatTensor(self.cmi_data[idx])
        rppg_mix = torch.FloatTensor(self.rppg_data[idx])
        voice_mix = torch.FloatTensor(self.voice_data[idx])
        target_mix = torch.FloatTensor([self.targets[idx]])

        # Mixup 적용
        big5 = lam * big5 + (1 - lam) * big5_mix
        cmi = lam * cmi + (1 - lam) * cmi_mix
        rppg = lam * rppg + (1 - lam) * rppg_mix
        voice = lam * voice + (1 - lam) * voice_mix
        target = lam * target + (1 - lam) * target_mix

        return big5, cmi, rppg, voice, target

    def _apply_cutmix(self, big5, cmi, rppg, voice, target):
        """CutMix 데이터 증강"""
        alpha = self.cutmix_alpha
        lam = np.random.beta(alpha, alpha)

        # 랜덤 샘플 선택
        idx = np.random.randint(0, len(self.targets))
        big5_mix = torch.FloatTensor(self.big5_data[idx])
        cmi_mix = torch.FloatTensor(self.cmi_data[idx])
        rppg_mix = torch.FloatTensor(self.rppg_data[idx])
        voice_mix = torch.FloatTensor(self.voice_data[idx])
        target_mix = torch.FloatTensor([self.targets[idx]])

        # CutMix 적용 (특징 차원에서)
        cut_len = int(lam * len(big5))
        cut_start = np.random.randint(0, len(big5) - cut_len + 1)

        big5[cut_start : cut_start + cut_len] = big5_mix[
            cut_start : cut_start + cut_len
        ]
        cmi[cut_start : cut_start + cut_len] = cmi_mix[cut_start : cut_start + cut_len]
        rppg[cut_start : cut_start + cut_len] = rppg_mix[
            cut_start : cut_start + cut_len
        ]
        voice[cut_start : cut_start + cut_len] = voice_mix[
            cut_start : cut_start + cut_len
        ]

        # 타겟은 가중 평균
        target = lam * target + (1 - lam) * target_mix

        return big5, cmi, rppg, voice, target


class TransferLearningMultimodalNet(nn.Module):
    """전이 학습용 멀티모달 네트워크"""

    def __init__(
        self,
        big5_dim=5,
        cmi_dim=20,
        rppg_dim=10,
        voice_dim=20,
        hidden_dim=128,
        dropout_rate=0.3,
        use_pretrained=True,
    ):
        super(TransferLearningMultimodalNet, self).__init__()

        self.hidden_dim = hidden_dim
        self.use_pretrained = use_pretrained

        # 모달리티별 인코더 (사전훈련된 가중치 활용)
        self.big5_encoder = nn.Sequential(
            nn.Linear(big5_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.cmi_encoder = nn.Sequential(
            nn.Linear(cmi_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.rppg_encoder = nn.Sequential(
            nn.Linear(rppg_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.voice_encoder = nn.Sequential(
            nn.Linear(voice_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # 크로스 모달 어텐션 (간단한 버전)
        self.attention_weights = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softmax(dim=1),
        )

        # 융합 레이어
        fusion_input_dim = 4 * (hidden_dim // 2) + (
            hidden_dim // 2
        )  # 4개 모달리티 + 어텐션
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # 출력 레이어
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid()  # 0-1 범위로 정규화
        )

        # 사전훈련된 가중치 로드 (가능한 경우)
        if use_pretrained:
            self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        """사전훈련된 가중치 로드"""
        try:
            # 기존 모델의 가중치 로드 시도
            pretrained_path = "ultimate_smart_model.pth"
            if os.path.exists(pretrained_path):
                checkpoint = torch.load(pretrained_path, map_location="cpu")
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                    # 호환되는 가중치만 로드
                    self._load_compatible_weights(state_dict)
                    print("✅ 사전훈련된 가중치 로드 완료")
                else:
                    print(
                        "⚠️ 사전훈련된 가중치를 찾을 수 없습니다. 랜덤 초기화를 사용합니다."
                    )
            else:
                print(
                    "⚠️ 사전훈련된 모델 파일을 찾을 수 없습니다. 랜덤 초기화를 사용합니다."
                )
        except Exception as e:
            print(f"⚠️ 사전훈련된 가중치 로드 실패: {str(e)}. 랜덤 초기화를 사용합니다.")

    def _load_compatible_weights(self, state_dict):
        """호환되는 가중치만 로드"""
        model_dict = self.state_dict()
        compatible_dict = {}

        for k, v in state_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                compatible_dict[k] = v

        model_dict.update(compatible_dict)
        self.load_state_dict(model_dict)
        print(f"✅ {len(compatible_dict)}개 가중치 로드 완료")

    def forward(self, big5, cmi, rppg, voice):
        # 모달리티별 인코딩
        big5_encoded = self.big5_encoder(big5)
        cmi_encoded = self.cmi_encoder(cmi)
        rppg_encoded = self.rppg_encoder(rppg)
        voice_encoded = self.voice_encoder(voice)

        # 간단한 어텐션 메커니즘
        # 모든 모달리티를 스택
        modalities = torch.stack(
            [big5_encoded, cmi_encoded, rppg_encoded, voice_encoded], dim=1
        )
        # 어텐션 가중치 계산
        attention_weights = self.attention_weights(modalities)  # [batch_size, 4, 1]
        # 가중 평균으로 어텐션 적용
        attended_features = torch.sum(
            modalities * attention_weights, dim=1
        )  # [batch_size, hidden_dim//2]

        # 융합
        fused_features = torch.cat(
            [big5_encoded, cmi_encoded, rppg_encoded, voice_encoded, attended_features],
            dim=1,
        )

        # 최종 예측
        fused = self.fusion_layer(fused_features)
        output = self.output_layer(fused)

        return output


class TransferLearningTrainer:
    """전이 학습 트레이너"""

    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.model.to(device)

    def train_epoch(self, dataloader, optimizer, criterion, scheduler=None):
        """한 에포크 훈련"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            optimizer.zero_grad()

            # 데이터를 디바이스로 이동
            big5 = batch["big5"].to(self.device)
            cmi = batch["cmi"].to(self.device)
            rppg = batch["rppg"].to(self.device)
            voice = batch["voice"].to(self.device)
            targets = batch["target"].to(self.device)

            # 예측
            predictions = self.model(big5, cmi, rppg, voice)

            # 손실 계산
            loss = criterion(predictions, targets)

            # 역전파
            loss.backward()

            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if scheduler:
            scheduler.step()

        return total_loss / num_batches

    def evaluate(self, dataloader, criterion):
        """모델 평가"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                big5 = batch["big5"].to(self.device)
                cmi = batch["cmi"].to(self.device)
                rppg = batch["rppg"].to(self.device)
                voice = batch["voice"].to(self.device)
                targets = batch["target"].to(self.device)

                predictions = self.model(big5, cmi, rppg, voice)
                loss = criterion(predictions, targets)

                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())

        # 메트릭 계산
        r2 = r2_score(all_targets, all_predictions)
        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        mae = mean_absolute_error(all_targets, all_predictions)

        return {
            "loss": total_loss / len(dataloader),
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "predictions": all_predictions,
            "targets": all_targets,
        }

    def train(
        self,
        train_loader,
        val_loader,
        epochs=50,
        lr=0.001,
        weight_decay=1e-4,
        patience=10,
    ):
        """전체 훈련 과정"""

        # 옵티마이저 및 스케줄러
        optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        criterion = nn.MSELoss()

        # 조기 종료
        best_val_r2 = -float("inf")
        patience_counter = 0
        best_model_state = None

        print(f"🚀 전이 학습 시작 (에포크: {epochs}, 학습률: {lr})")

        for epoch in range(epochs):
            # 훈련
            train_loss = self.train_epoch(train_loader, optimizer, criterion, scheduler)

            # 검증
            val_metrics = self.evaluate(val_loader, criterion)

            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val R²: {val_metrics['r2']:.4f}")
            print(f"  Val RMSE: {val_metrics['rmse']:.4f}")
            print(f"  Val MAE: {val_metrics['mae']:.4f}")

            # 최고 성능 모델 저장
            if val_metrics["r2"] > best_val_r2:
                best_val_r2 = val_metrics["r2"]
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                print(f"  ✅ 새로운 최고 성능! R²: {best_val_r2:.4f}")
            else:
                patience_counter += 1
                print(f"  ⏳ 조기 종료 카운터: {patience_counter}/{patience}")

            # 조기 종료
            if patience_counter >= patience:
                print(f"🛑 조기 종료! {patience}번 연속 개선 없음")
                break

        # 최고 성능 모델 복원
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            print(f"✅ 최고 성능 모델 복원 완료 (R²: {best_val_r2:.4f})")

        return best_val_r2


def objective(trial, train_loader, val_loader, device):
    """Optuna 최적화 목적 함수"""

    # 하이퍼파라미터 제안
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    # 모델 생성
    model = TransferLearningMultimodalNet(
        hidden_dim=hidden_dim, dropout_rate=dropout_rate
    )

    # 트레이너 생성
    trainer = TransferLearningTrainer(model, device)

    # 훈련
    best_r2 = trainer.train(
        train_loader,
        val_loader,
        epochs=30,  # 빠른 튜닝을 위해 에포크 수 감소
        lr=lr,
        weight_decay=weight_decay,
        patience=5,
    )

    return best_r2


def run_transfer_learning_experiment():
    """전이 학습 실험 실행"""

    print("🎯 전이 학습 멀티모달 모델 실험 시작!")

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    # 데이터 생성 (실제 BigQuery 데이터 시뮬레이션)
    print("📊 데이터 생성 중...")

    # 합성 데이터 생성 (사전훈련용)
    n_samples = 10000
    big5_data = np.random.randn(n_samples, 5)
    cmi_data = np.random.randn(n_samples, 20)
    rppg_data = np.random.randn(n_samples, 10)
    voice_data = np.random.randn(n_samples, 20)

    # 타겟 생성 (더 현실적인 관계)
    targets = (
        0.3 * big5_data[:, 0]  # Openness
        + 0.2 * big5_data[:, 1]  # Conscientiousness
        + 0.1 * big5_data[:, 2]  # Extraversion
        + 0.2 * big5_data[:, 3]  # Agreeableness
        + 0.1 * big5_data[:, 4]  # Neuroticism
        + 0.1 * np.mean(cmi_data, axis=1)
        + 0.05 * np.mean(rppg_data, axis=1)
        + 0.05 * np.mean(voice_data, axis=1)
        + np.random.normal(0, 0.1, n_samples)  # 노이즈
    )

    # 타겟 정규화 (0-1 범위)
    targets = (targets - targets.min()) / (targets.max() - targets.min())

    # 데이터 분할
    # 먼저 인덱스로 분할
    train_idx, test_idx = train_test_split(
        range(len(targets)), test_size=0.2, random_state=42
    )
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)

    # 데이터 분할
    X_train = {
        "big5": big5_data[train_idx],
        "cmi": cmi_data[train_idx],
        "rppg": rppg_data[train_idx],
        "voice": voice_data[train_idx],
    }
    X_val = {
        "big5": big5_data[val_idx],
        "cmi": cmi_data[val_idx],
        "rppg": rppg_data[val_idx],
        "voice": voice_data[val_idx],
    }
    X_test = {
        "big5": big5_data[test_idx],
        "cmi": cmi_data[test_idx],
        "rppg": rppg_data[test_idx],
        "voice": voice_data[test_idx],
    }

    y_train = targets[train_idx]
    y_val = targets[val_idx]
    y_test = targets[test_idx]

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

    # Optuna 하이퍼파라미터 튜닝
    print("🔍 하이퍼파라미터 튜닝 시작...")

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader, device),
        n_trials=20,  # 빠른 튜닝을 위해 시도 횟수 감소
        show_progress_bar=True,
    )

    print(f"✅ 최적 하이퍼파라미터: {study.best_params}")
    print(f"✅ 최고 R²: {study.best_value:.4f}")

    # 최적 하이퍼파라미터로 최종 모델 훈련
    print("🎯 최적 하이퍼파라미터로 최종 모델 훈련...")

    best_model = TransferLearningMultimodalNet(
        hidden_dim=study.best_params["hidden_dim"],
        dropout_rate=study.best_params["dropout_rate"],
    )

    best_trainer = TransferLearningTrainer(best_model, device)
    final_r2 = best_trainer.train(
        train_loader,
        val_loader,
        epochs=50,
        lr=study.best_params["lr"],
        weight_decay=study.best_params["weight_decay"],
        patience=10,
    )

    # 테스트 성능 평가
    print("🧪 테스트 성능 평가...")
    test_metrics = best_trainer.evaluate(test_loader, nn.MSELoss())

    print("\n📊 최종 결과:")
    print(f"  검증 R²: {final_r2:.4f}")
    print(f"  테스트 R²: {test_metrics['r2']:.4f}")
    print(f"  테스트 RMSE: {test_metrics['rmse']:.4f}")
    print(f"  테스트 MAE: {test_metrics['mae']:.4f}")

    # 결과 저장
    results = {
        "best_params": study.best_params,
        "best_val_r2": study.best_value,
        "final_val_r2": final_r2,
        "test_r2": test_metrics["r2"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "model_architecture": {
            "hidden_dim": study.best_params["hidden_dim"],
            "dropout_rate": study.best_params["dropout_rate"],
        },
    }

    with open("transfer_learning_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # 모델 저장
    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "best_params": study.best_params,
            "test_metrics": test_metrics,
        },
        "transfer_learning_model.pth",
    )

    print("✅ 결과 저장 완료!")
    print("📁 transfer_learning_results.json")
    print("📁 transfer_learning_model.pth")

    return results


if __name__ == "__main__":
    results = run_transfer_learning_experiment()
