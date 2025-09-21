#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contrastive Learning 멀티모달 예시
- 모달리티 간 유사성 학습
- 공통 표현 공간 구축
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveMultiModal(nn.Module):
    """대조 학습 기반 멀티모달"""

    def __init__(self, hidden_dim=128, projection_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim

        # 각 모달리티별 인코더
        self.big5_encoder = nn.Sequential(
            nn.Linear(25, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, projection_dim)
        )
        self.cmi_encoder = nn.Sequential(
            nn.Linear(10, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, projection_dim)
        )
        self.rppg_encoder = nn.Sequential(
            nn.Linear(15, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, projection_dim)
        )
        self.voice_encoder = nn.Sequential(
            nn.Linear(20, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, projection_dim)
        )

        # 공통 표현 공간으로의 투영
        self.projection = nn.Linear(projection_dim * 4, hidden_dim)

        # 최종 예측
        self.predictor = nn.Linear(hidden_dim, 1)

    def forward(self, big5, cmi, rppg, voice):
        # 1. 각 모달리티를 공통 공간으로 투영
        big5_proj = F.normalize(self.big5_encoder(big5), dim=1)
        cmi_proj = F.normalize(self.cmi_encoder(cmi), dim=1)
        rppg_proj = F.normalize(self.rppg_encoder(rppg), dim=1)
        voice_proj = F.normalize(self.voice_encoder(voice), dim=1)

        # 2. 모달리티 간 유사성 계산
        similarities = []
        modalities = [big5_proj, cmi_proj, rppg_proj, voice_proj]

        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities):
                if i != j:
                    sim = F.cosine_similarity(mod1, mod2, dim=1)
                    similarities.append(sim)

        # 3. 모든 모달리티 결합
        combined = torch.cat([big5_proj, cmi_proj, rppg_proj, voice_proj], dim=1)

        # 4. 공통 표현 공간으로 투영
        common_representation = self.projection(combined)

        # 5. 예측
        output = self.predictor(common_representation)

        return output, similarities

    def contrastive_loss(self, big5, cmi, rppg, voice, temperature=0.1):
        """대조 학습 손실"""
        # 각 모달리티 투영
        big5_proj = F.normalize(self.big5_encoder(big5), dim=1)
        cmi_proj = F.normalize(self.cmi_encoder(cmi), dim=1)
        rppg_proj = F.normalize(self.rppg_encoder(rppg), dim=1)
        voice_proj = F.normalize(self.voice_encoder(voice), dim=1)

        # 모달리티 간 유사성 매트릭스
        modalities = [big5_proj, cmi_proj, rppg_proj, voice_proj]
        similarity_matrix = torch.zeros(4, 4, device=big5.device)

        for i in range(4):
            for j in range(4):
                if i != j:
                    sim = F.cosine_similarity(
                        modalities[i], modalities[j], dim=1
                    ).mean()
                    similarity_matrix[i, j] = sim

        # 대조 손실 (유사한 모달리티는 가깝게, 다른 모달리티는 멀게)
        loss = 0
        for i in range(4):
            for j in range(4):
                if i != j:
                    # 같은 샘플의 다른 모달리티는 양성 쌍
                    pos_sim = F.cosine_similarity(modalities[i], modalities[j], dim=1)
                    # 다른 샘플의 같은 모달리티는 음성 쌍
                    neg_sim = F.cosine_similarity(
                        modalities[i], modalities[j].roll(1, 0), dim=1
                    )

                    # InfoNCE 손실
                    logits = (
                        torch.cat([pos_sim.unsqueeze(0), neg_sim.unsqueeze(0)])
                        / temperature
                    )
                    labels = torch.zeros(1, dtype=torch.long, device=big5.device)
                    loss += F.cross_entropy(logits, labels)

        return loss / 12  # 4*3 = 12개의 쌍


# 사용 예시
if __name__ == "__main__":
    model = ContrastiveMultiModal()

    # 샘플 데이터
    batch_size = 32
    big5 = torch.randn(batch_size, 25)
    cmi = torch.randn(batch_size, 10)
    rppg = torch.randn(batch_size, 15)
    voice = torch.randn(batch_size, 20)

    # Forward pass
    output, similarities = model(big5, cmi, rppg, voice)
    print(f"Output shape: {output.shape}")
    print(f"Similarities: {len(similarities)} pairs")

    # 대조 손실 계산
    contrastive_loss = model.contrastive_loss(big5, cmi, rppg, voice)
    print(f"Contrastive Loss: {contrastive_loss.item():.4f}")
    print("✅ Contrastive Learning 완료!")



