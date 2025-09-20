#!/usr/bin/env python3
"""
데이터 품질 개선 및 검증 시스템
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from vector_search_system import Big5VectorSearch


class DataQualityImprover:
    def __init__(self, project_id: str = "persona-diary-service"):
        self.project_id = project_id
        self.vector_search = Big5VectorSearch(project_id)

    def stratified_sampling(
        self, data: pd.DataFrame, sample_size: int = 5000
    ) -> pd.DataFrame:
        """국가별 계층적 샘플링"""
        print(f"계층적 샘플링 시작... (목표: {sample_size}건)")

        # 국가별 분포 확인
        country_counts = data["country"].value_counts()
        print(f"국가별 분포: {dict(country_counts.head())}")

        # 각 국가에서 비례적으로 샘플링
        sampled_data = []
        total_countries = len(country_counts)

        for country in country_counts.index:
            country_data = data[data["country"] == country]
            country_sample_size = max(
                1, int(sample_size * len(country_data) / len(data))
            )

            if len(country_data) >= country_sample_size:
                sampled = country_data.sample(n=country_sample_size, random_state=42)
            else:
                sampled = country_data

            sampled_data.append(sampled)
            print(f"  {country}: {len(sampled)}건 샘플링")

        result = pd.concat(sampled_data, ignore_index=True)
        print(f"계층적 샘플링 완료: {len(result)}건")
        return result

    def normalize_big5_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """Big5 점수 정규화 (1-6 범위를 1-5로 통일)"""
        print("Big5 점수 정규화 중...")

        # EST, AGR 컬럼들을 2-6 범위에서 1-5 범위로 변환
        for col in data.columns:
            if col.startswith(("EST", "AGR")):
                # 2-6 범위를 1-5로 선형 변환
                data[col] = ((data[col] - 2) / 4) * 4 + 1

        print("Big5 점수 정규화 완료")
        return data

    def validate_data_quality(self, data: pd.DataFrame) -> dict:
        """데이터 품질 검증"""
        print("데이터 품질 검증 중...")

        quality_report = {
            "total_records": len(data),
            "missing_values": data.isnull().sum().sum(),
            "duplicate_records": data.duplicated().sum(),
            "score_ranges": {},
            "country_distribution": data["country"].value_counts().to_dict(),
        }

        # Big5 점수 범위 확인
        big5_cols = [
            col
            for col in data.columns
            if any(trait in col for trait in ["EXT", "EST", "AGR", "CSN", "OPN"])
        ]
        for trait in ["EXT", "EST", "AGR", "CSN", "OPN"]:
            trait_cols = [col for col in big5_cols if col.startswith(trait)]
            if trait_cols:
                trait_data = data[trait_cols].values.flatten()
                quality_report["score_ranges"][trait] = {
                    "min": float(np.min(trait_data)),
                    "max": float(np.max(trait_data)),
                    "mean": float(np.mean(trait_data)),
                }

        return quality_report

    def improve_data_quality(self, sample_size: int = 5000) -> tuple:
        """데이터 품질 개선 메인 함수"""
        print("=== 데이터 품질 개선 시작 ===")

        # 1. 원본 데이터 로드 (더 큰 샘플)
        print("1. 원본 데이터 로드 중...")
        raw_data = self.vector_search.load_data(limit=sample_size * 2)

        # 2. 계층적 샘플링
        print("2. 계층적 샘플링 수행...")
        sampled_data = self.stratified_sampling(raw_data, sample_size)

        # 3. Big5 점수 정규화
        print("3. Big5 점수 정규화...")
        normalized_data = self.normalize_big5_scores(sampled_data.copy())

        # 4. 품질 검증
        print("4. 품질 검증 수행...")
        quality_report = self.validate_data_quality(normalized_data)

        return normalized_data, quality_report


def main():
    """메인 실행 함수"""
    print("🔧 데이터 품질 개선 시스템")

    improver = DataQualityImprover()

    # 데이터 품질 개선 실행
    improved_data, quality_report = improver.improve_data_quality(sample_size=3000)

    # 결과 출력
    print("\n" + "=" * 60)
    print("📊 데이터 품질 개선 결과")
    print("=" * 60)

    print(f"\n📈 기본 통계:")
    print(f"   총 레코드 수: {quality_report['total_records']:,}")
    print(f"   결측값 수: {quality_report['missing_values']}")
    print(f"   중복 레코드 수: {quality_report['duplicate_records']}")

    print(f"\n📊 Big5 점수 범위 (정규화 후):")
    for trait, stats in quality_report["score_ranges"].items():
        print(
            f"   {trait}: {stats['min']:.1f} ~ {stats['max']:.1f} (평균: {stats['mean']:.2f})"
        )

    print(f"\n🌍 국가별 분포 (상위 5개):")
    for country, count in list(quality_report["country_distribution"].items())[:5]:
        percentage = (count / quality_report["total_records"]) * 100
        print(f"   {country}: {count:,}건 ({percentage:.1f}%)")

    print(f"\n✅ 데이터 품질 개선 완료!")


if __name__ == "__main__":
    main()
