#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Modal Transformer 예시
- 모든 모달리티를 하나의 시퀀스로 처리
- 완전 통합된 학습
"""

import math

import torch
import torch.nn as nn


class MultiModalTransformer(nn.Module):
    """완전 통합된 멀티모달 트랜스포머"""

    def __init__(self, hidden_dim=128, num_heads=8, num_layers=6):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 모달리티별 임베딩 + 위치 임베딩
        self.big5_embedding = nn.Linear(25, hidden_dim)
        self.cmi_embedding = nn.Linear(10, hidden_dim)
        self.rppg_embedding = nn.Linear(15, hidden_dim)
        self.voice_embedding = nn.Linear(20, hidden_dim)

        # 모달리티 타입 임베딩 (Big5=0, CMI=1, RPPG=2, Voice=3)
        self.modality_embedding = nn.Embedding(4, hidden_dim)

        # 위치 임베딩
        self.position_embedding = nn.Embedding(4, hidden_dim)

        # 트랜스포머 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 최종 예측
        self.predictor = nn.Linear(hidden_dim, 1)

    def forward(self, big5, cmi, rppg, voice):
        batch_size = big5.size(0)

        # 1. 각 모달리티 임베딩
        big5_emb = self.big5_embedding(big5)
        cmi_emb = self.cmi_embedding(cmi)
        rppg_emb = self.rppg_embedding(rppg)
        voice_emb = self.voice_embedding(voice)

        # 2. 모달리티 타입 임베딩 추가
        big5_emb += self.modality_embedding(torch.zeros(batch_size, dtype=torch.long))
        cmi_emb += self.modality_embedding(torch.ones(batch_size, dtype=torch.long))
        rppg_emb += self.modality_embedding(
            torch.full((batch_size,), 2, dtype=torch.long)
        )
        voice_emb += self.modality_embedding(
            torch.full((batch_size,), 3, dtype=torch.long)
        )

        # 3. 위치 임베딩 추가
        positions = torch.arange(4).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)

        # 4. 모든 모달리티를 시퀀스로 결합
        sequence = torch.stack([big5_emb, cmi_emb, rppg_emb, voice_emb], dim=1)
        sequence += pos_emb

        # 5. 트랜스포머 처리
        transformer_output = self.transformer(sequence)

        # 6. [CLS] 토큰 역할 (첫 번째 위치) 또는 평균 풀링
        final_representation = transformer_output.mean(dim=1)

        # 7. 예측
        output = self.predictor(final_representation)

        return output


# 사용 예시
if __name__ == "__main__":
    model = MultiModalTransformer()

    # 샘플 데이터
    batch_size = 32
    big5 = torch.randn(batch_size, 25)
    cmi = torch.randn(batch_size, 10)
    rppg = torch.randn(batch_size, 15)
    voice = torch.randn(batch_size, 20)

    # Forward pass
    output = model(big5, cmi, rppg, voice)
    print(f"Output shape: {output.shape}")
    print("✅ Multi-Modal Transformer 완료!")



