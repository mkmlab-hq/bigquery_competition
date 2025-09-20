#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Modal Attention 예시
- 모든 모달리티가 서로에게 어텐션
- 양방향 정보 교환
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """모든 모달리티 간 상호작용 학습"""

    def __init__(self, hidden_dim=128, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # 각 모달리티별 인코더
        self.big5_encoder = nn.Linear(25, hidden_dim)
        self.cmi_encoder = nn.Linear(10, hidden_dim)
        self.rppg_encoder = nn.Linear(15, hidden_dim)
        self.voice_encoder = nn.Linear(20, hidden_dim)

        # Cross-Modal Attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

        # 최종 예측
        self.predictor = nn.Linear(hidden_dim, 1)

    def forward(self, big5, cmi, rppg, voice):
        # 1. 각 모달리티 인코딩
        big5_encoded = self.big5_encoder(big5)
        cmi_encoded = self.cmi_encoder(cmi)
        rppg_encoded = self.rppg_encoder(rppg)
        voice_encoded = self.voice_encoder(voice)

        # 2. 모든 모달리티를 시퀀스로 결합
        modalities = torch.stack(
            [big5_encoded, cmi_encoded, rppg_encoded, voice_encoded], dim=1
        )  # [batch, 4, hidden_dim]

        # 3. Cross-Modal Attention (모든 모달리티가 서로에게 어텐션)
        attended, attention_weights = self.cross_attention(
            modalities, modalities, modalities
        )

        # 4. 평균 풀링으로 최종 표현
        final_representation = attended.mean(dim=1)

        # 5. 예측
        output = self.predictor(final_representation)

        return output, attention_weights


# 사용 예시
if __name__ == "__main__":
    model = CrossModalAttention()

    # 샘플 데이터
    batch_size = 32
    big5 = torch.randn(batch_size, 25)
    cmi = torch.randn(batch_size, 10)
    rppg = torch.randn(batch_size, 15)
    voice = torch.randn(batch_size, 20)

    # Forward pass
    output, attention = model(big5, cmi, rppg, voice)
    print(f"Output shape: {output.shape}")
    print(f"Attention shape: {attention.shape}")
    print("✅ Cross-Modal Attention 완료!")

