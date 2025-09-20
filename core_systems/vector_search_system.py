#!/usr/bin/env python3
"""
Vector Search System for Big5 Personality Data
실제 BigQuery 데이터를 활용한 유사도 검색 시스템
"""

import json
import os
from typing import Dict, List, Tuple

import google.cloud.bigquery as bigquery
import numpy as np
import pandas as pd
from google.cloud import aiplatform
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


class Big5VectorSearch:
    def __init__(self, project_id: str = "persona-diary-service"):
        """Big5 Vector Search 시스템 초기화"""
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)
        self.scaler = StandardScaler()

        # Big5 성격 특성 컬럼들
        self.big5_columns = {
            "EXT": [f"EXT{i}" for i in range(1, 11)],  # 외향성
            "EST": [f"EST{i}" for i in range(1, 11)],  # 신경증
            "AGR": [f"AGR{i}" for i in range(1, 11)],  # 친화성
            "CSN": [f"CSN{i}" for i in range(1, 11)],  # 성실성
            "OPN": [f"OPN{i}" for i in range(1, 11)],  # 개방성
        }

        # 모든 Big5 컬럼
        self.all_big5_columns = []
        for trait_columns in self.big5_columns.values():
            self.all_big5_columns.extend(trait_columns)

    def load_data(self, limit: int = 10000) -> pd.DataFrame:
        """BigQuery에서 Big5 데이터 로드"""
        query = f"""
        SELECT {', '.join(self.all_big5_columns + ['country'])}
        FROM `{self.project_id}.big5_dataset.big5_preprocessed`
        LIMIT {limit}
        """

        print(f"BigQuery에서 데이터 로딩 중... (최대 {limit}건)")
        df = self.client.query(query).to_dataframe()
        print(f"로드된 데이터: {len(df)}건")
        return df

    def preprocess_data(self, df: pd.DataFrame) -> np.ndarray:
        """데이터 전처리 및 정규화"""
        # Big5 점수만 추출
        big5_data = df[self.all_big5_columns].values

        # 정규화
        normalized_data = self.scaler.fit_transform(big5_data)
        return normalized_data

    def find_similar_users(
        self, target_user: np.ndarray, all_users: np.ndarray, top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """유사한 사용자 찾기 (코사인 유사도 기반)"""
        # 코사인 유사도 계산
        similarities = cosine_similarity([target_user], all_users)[0]

        # 상위 K개 유사 사용자 반환
        top_indices = np.argsort(similarities)[::-1][:top_k]
        similar_users = [(idx, similarities[idx]) for idx in top_indices]

        return similar_users

    def get_personality_profile(self, user_data: np.ndarray) -> Dict[str, float]:
        """사용자의 성격 프로필 계산"""
        profile = {}
        start_idx = 0

        for trait, columns in self.big5_columns.items():
            end_idx = start_idx + len(columns)
            trait_scores = user_data[start_idx:end_idx]
            profile[trait] = float(np.mean(trait_scores))
            start_idx = end_idx

        return profile

    def search_similar_users(
        self,
        target_user_data: Dict[str, float],
        all_data: pd.DataFrame,
        top_k: int = 10,
    ) -> List[Dict]:
        """유사 사용자 검색 메인 함수"""
        print("Vector Search 시작...")

        # 타겟 사용자 데이터를 벡터로 변환
        target_vector = []
        for trait, columns in self.big5_columns.items():
            trait_score = target_user_data.get(trait, 3.0)  # 기본값 3.0
            # 10개 항목에 동일한 점수 적용 (실제로는 각 항목별 점수가 있어야 함)
            target_vector.extend([trait_score] * 10)

        target_vector = np.array(target_vector)

        # 전체 데이터 전처리
        all_vectors = self.preprocess_data(all_data)

        # 유사 사용자 찾기
        similar_users = self.find_similar_users(target_vector, all_vectors, top_k)

        # 결과 포맷팅
        results = []
        for idx, similarity in similar_users:
            user_data = all_data.iloc[idx]
            profile = self.get_personality_profile(all_vectors[idx])

            results.append(
                {
                    "user_index": idx,
                    "similarity_score": float(similarity),
                    "country": user_data["country"],
                    "personality_profile": profile,
                }
            )

        return results


def main():
    """메인 실행 함수"""
    print("🚀 Big5 Vector Search System 시작")

    # Vector Search 시스템 초기화
    search_system = Big5VectorSearch()

    # 데이터 로드 (처음 1000건으로 테스트)
    print("\n📊 데이터 로딩 중...")
    data = search_system.load_data(limit=1000)

    # 샘플 타겟 사용자 (평균적인 성격)
    target_user = {
        "EXT": 3.0,  # 외향성
        "EST": 3.0,  # 신경증
        "AGR": 3.0,  # 친화성
        "CSN": 3.0,  # 성실성
        "OPN": 3.0,  # 개방성
    }

    print(f"\n🎯 타겟 사용자: {target_user}")

    # 유사 사용자 검색
    print("\n🔍 유사 사용자 검색 중...")
    similar_users = search_system.search_similar_users(
        target_user_data=target_user, all_data=data, top_k=5
    )

    # 결과 출력
    print("\n✅ 검색 결과:")
    print("=" * 80)
    for i, user in enumerate(similar_users, 1):
        print(f"\n{i}. 사용자 #{user['user_index']}")
        print(f"   유사도: {user['similarity_score']:.4f}")
        print(f"   국가: {user['country']}")
        print(f"   성격 프로필: {user['personality_profile']}")

    print(f"\n🎉 Vector Search 완료! {len(similar_users)}명의 유사 사용자 발견")


if __name__ == "__main__":
    main()
