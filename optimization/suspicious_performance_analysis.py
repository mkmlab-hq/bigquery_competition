#!/usr/bin/env python3
"""
의심스러운 성능 분석 시스템 - R² 0.999+ 수치의 신뢰성 검증
- 타겟 변수 생성 과정에서의 데이터 누출 가능성 검증
- 실제 예측 가능성 vs 데이터 누출 구분
"""

import json
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from google.cloud import bigquery
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)


class SuspiciousPerformanceAnalyzer:
    """의심스러운 성능 분석 시스템"""

    def __init__(self, project_id: str = "persona-diary-service"):
        self.project_id = project_id
        self.scalers = {}
        self.models = {}

        # 인증 파일 경로 설정
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
            "F:/workspace/bigquery_competition/optimization/gcs-key.json"
        )

        try:
            self.client = bigquery.Client(project=project_id)
            print(f"✅ BigQuery 클라이언트 초기화 완료: {project_id}")
        except Exception as e:
            print(f"❌ BigQuery 인증 실패: {str(e)}")
            raise e

    def load_real_bigquery_data(
        self, limit: int = 10000
    ) -> Tuple[Dict, np.ndarray, Dict]:
        """실제 BigQuery 데이터 로딩 (타겟 변수 생성 과정 포함)"""
        print("🔄 실제 BigQuery 데이터 로딩 중...")
        try:
            # Big5 데이터 로딩
            big5_query = f"""
            SELECT * FROM `persona-diary-service.big5_dataset.big5_preprocessed` LIMIT {limit}
            """
            big5_df = self.client.query(big5_query).to_dataframe()

            # CMI 데이터 로딩
            cmi_query = f"""
            SELECT * FROM `persona-diary-service.cmi_dataset.cmi_preprocessed` LIMIT {limit}
            """
            cmi_df = self.client.query(cmi_query).to_dataframe()

            # RPPG 데이터 로딩
            rppg_query = f"""
            SELECT * FROM `persona-diary-service.rppg_dataset.rppg_preprocessed` LIMIT {limit}
            """
            rppg_df = self.client.query(rppg_query).to_dataframe()

            # Voice 데이터 로딩
            voice_query = f"""
            SELECT * FROM `persona-diary-service.voice_dataset.voice_preprocessed` LIMIT {limit}
            """
            voice_df = self.client.query(voice_query).to_dataframe()

            # 수치 데이터만 선택
            cmi_numeric = cmi_df.select_dtypes(include=[np.number])
            rppg_numeric = rppg_df.select_dtypes(include=[np.number])
            voice_numeric = voice_df.select_dtypes(include=[np.number])
            big5_numeric = big5_df.select_dtypes(include=[np.number])

            # 데이터 결합
            multimodal_data = {
                "big5": big5_numeric.values,
                "cmi": cmi_numeric.values,
                "rppg": rppg_numeric.values,
                "voice": voice_numeric.values,
            }

            # 타겟 변수 생성 과정 분석
            print("🔍 타겟 변수 생성 과정 분석 중...")

            # Big5 점수 계산
            big5_scores = {
                "EXT": big5_df[["EXT1", "EXT2", "EXT3", "EXT4", "EXT5"]].mean(axis=1),
                "EST": big5_df[["EST1", "EST2", "EST3", "EST4", "EST5"]].mean(axis=1),
                "AGR": big5_df[["AGR1", "AGR2", "AGR3", "AGR4", "AGR5"]].mean(axis=1),
                "CSN": big5_df[["CSN1", "CSN2", "CSN3", "CSN4", "CSN5"]].mean(axis=1),
                "OPN": big5_df[["OPN1", "OPN2", "OPN3", "OPN4", "OPN5"]].mean(axis=1),
            }

            # 타겟 변수 생성 (의심스러운 부분!)
            targets = (
                big5_scores["EXT"] * 0.25
                + big5_scores["OPN"] * 0.20
                + (6 - big5_scores["EST"]) * 0.15
                + big5_scores["AGR"] * 0.15
                + big5_scores["CSN"] * 0.10
                + (cmi_numeric.mean(axis=1) / 6) * 0.10
                + (rppg_numeric.mean(axis=1) / 6) * 0.05
            )

            # 1-10 스케일로 정규화
            targets = (targets - targets.min()) / (
                targets.max() - targets.min()
            ) * 9 + 1

            # 타겟 변수 생성 정보
            target_info = {
                "big5_scores": {k: v.tolist() for k, v in big5_scores.items()},
                "cmi_mean": cmi_numeric.mean(axis=1).tolist(),
                "rppg_mean": rppg_numeric.mean(axis=1).tolist(),
                "targets_raw": targets.tolist(),
                "target_stats": {
                    "mean": float(targets.mean()),
                    "std": float(targets.std()),
                    "min": float(targets.min()),
                    "max": float(targets.max()),
                },
            }

            print(f"✅ 실제 BigQuery 데이터 로딩 완료:")
            print(f"   Big5: {big5_numeric.shape}")
            print(f"   CMI: {cmi_numeric.shape}")
            print(f"   RPPG: {rppg_numeric.shape}")
            print(f"   Voice: {voice_numeric.shape}")
            print(f"   Targets: {targets.shape}")

            return multimodal_data, targets, target_info

        except Exception as e:
            print(f"❌ BigQuery 데이터 로딩 실패: {str(e)}")
            raise e

    def analyze_target_leakage(
        self, multimodal_data: Dict, targets: np.ndarray, target_info: Dict
    ) -> Dict:
        """타겟 변수 누출 분석"""
        print("🔍 타겟 변수 누출 분석 중...")

        # 1. 타겟 변수와 입력 특성 간의 상관관계 분석
        X = np.concatenate(
            [
                multimodal_data["big5"],
                multimodal_data["cmi"],
                multimodal_data["rppg"],
                multimodal_data["voice"],
            ],
            axis=1,
        )

        # 2. 각 모달리티별 상관관계 분석 (간단한 방법)
        leakage_analysis = {}

        # Big5와 타겟 변수의 상관관계
        big5_corrs = []
        for i in range(multimodal_data["big5"].shape[1]):
            corr = np.corrcoef(multimodal_data["big5"][:, i], targets)[0, 1]
            if not np.isnan(corr):
                big5_corrs.append(abs(corr))
        big5_max_corr = max(big5_corrs) if big5_corrs else 0

        # CMI와 타겟 변수의 상관관계
        cmi_corrs = []
        for i in range(multimodal_data["cmi"].shape[1]):
            corr = np.corrcoef(multimodal_data["cmi"][:, i], targets)[0, 1]
            if not np.isnan(corr):
                cmi_corrs.append(abs(corr))
        cmi_max_corr = max(cmi_corrs) if cmi_corrs else 0

        # RPPG와 타겟 변수의 상관관계
        rppg_corrs = []
        for i in range(multimodal_data["rppg"].shape[1]):
            corr = np.corrcoef(multimodal_data["rppg"][:, i], targets)[0, 1]
            if not np.isnan(corr):
                rppg_corrs.append(abs(corr))
        rppg_max_corr = max(rppg_corrs) if rppg_corrs else 0

        # Voice와 타겟 변수의 상관관계
        voice_corrs = []
        for i in range(multimodal_data["voice"].shape[1]):
            corr = np.corrcoef(multimodal_data["voice"][:, i], targets)[0, 1]
            if not np.isnan(corr):
                voice_corrs.append(abs(corr))
        voice_max_corr = max(voice_corrs) if voice_corrs else 0

        leakage_analysis = {
            "big5_max_correlation": float(big5_max_corr),
            "cmi_max_correlation": float(cmi_max_corr),
            "rppg_max_correlation": float(rppg_max_corr),
            "voice_max_correlation": float(voice_max_corr),
            "overall_max_correlation": float(
                np.max([big5_max_corr, cmi_max_corr, rppg_max_corr, voice_max_corr])
            ),
            "leakage_risk": (
                "HIGH"
                if np.max([big5_max_corr, cmi_max_corr, rppg_max_corr, voice_max_corr])
                > 0.9
                else (
                    "MEDIUM"
                    if np.max(
                        [big5_max_corr, cmi_max_corr, rppg_max_corr, voice_max_corr]
                    )
                    > 0.7
                    else "LOW"
                )
            ),
        }

        print(f"   Big5 최대 상관관계: {big5_max_corr:.4f}")
        print(f"   CMI 최대 상관관계: {cmi_max_corr:.4f}")
        print(f"   RPPG 최대 상관관계: {rppg_max_corr:.4f}")
        print(f"   Voice 최대 상관관계: {voice_max_corr:.4f}")
        print(
            f"   전체 최대 상관관계: {leakage_analysis['overall_max_correlation']:.4f}"
        )
        print(f"   누출 위험도: {leakage_analysis['leakage_risk']}")

        return leakage_analysis

    def test_realistic_prediction(
        self, multimodal_data: Dict, targets: np.ndarray
    ) -> Dict:
        """현실적인 예측 성능 테스트"""
        print("🔍 현실적인 예측 성능 테스트 중...")

        # 1. 데이터 준비
        X = np.concatenate(
            [
                multimodal_data["big5"],
                multimodal_data["cmi"],
                multimodal_data["rppg"],
                multimodal_data["voice"],
            ],
            axis=1,
        )

        # 2. 훈련/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, targets, test_size=0.3, random_state=42
        )

        # 3. 간단한 모델로 테스트
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression

        # Linear Regression (가장 기본적인 모델)
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        lr_r2 = r2_score(y_test, lr_pred)

        # Random Forest (과적합 방지)
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
        )
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_r2 = r2_score(y_test, rf_pred)

        realistic_test = {
            "linear_regression_r2": float(lr_r2),
            "random_forest_r2": float(rf_r2),
            "is_realistic": lr_r2 < 0.8 and rf_r2 < 0.9,  # 현실적인 범위
            "suspicious": lr_r2 > 0.95 or rf_r2 > 0.95,  # 의심스러운 범위
        }

        print(f"   Linear Regression R²: {lr_r2:.4f}")
        print(f"   Random Forest R²: {rf_r2:.4f}")
        print(f"   현실적 성능: {realistic_test['is_realistic']}")
        print(f"   의심스러운 성능: {realistic_test['suspicious']}")

        return realistic_test

    def analyze_target_construction(self, target_info: Dict) -> Dict:
        """타겟 변수 구성 분석"""
        print("🔍 타겟 변수 구성 분석 중...")

        # 타겟 변수가 어떻게 구성되었는지 분석
        construction_analysis = {
            "big5_contribution": 0.25 + 0.20 + 0.15 + 0.15 + 0.10,  # 0.85 (85%)
            "cmi_contribution": 0.10,  # 10%
            "rppg_contribution": 0.05,  # 5%
            "big5_dominant": True,  # Big5가 85% 차지
            "construction_method": "weighted_average",
            "potential_leakage": "Big5 데이터가 타겟 변수의 85%를 차지하여 데이터 누출 가능성 높음",
        }

        print(f"   Big5 기여도: {construction_analysis['big5_contribution']:.1%}")
        print(f"   CMI 기여도: {construction_analysis['cmi_contribution']:.1%}")
        print(f"   RPPG 기여도: {construction_analysis['rppg_contribution']:.1%}")
        print(f"   Big5 지배적: {construction_analysis['big5_dominant']}")
        print(f"   잠재적 누출: {construction_analysis['potential_leakage']}")

        return construction_analysis

    def run_suspicious_analysis(self, limit: int = 10000) -> Dict:
        """의심스러운 성능 분석 실행"""
        print("🚀 의심스러운 성능 분석 시스템 시작")
        print("=" * 60)

        # 1. 실제 BigQuery 데이터 로딩
        multimodal_data, targets, target_info = self.load_real_bigquery_data(limit)

        # 2. 타겟 변수 누출 분석
        leakage_analysis = self.analyze_target_leakage(
            multimodal_data, targets, target_info
        )

        # 3. 현실적인 예측 성능 테스트
        realistic_test = self.test_realistic_prediction(multimodal_data, targets)

        # 4. 타겟 변수 구성 분석
        construction_analysis = self.analyze_target_construction(target_info)

        # 5. 결과 종합
        results = {
            "leakage_analysis": leakage_analysis,
            "realistic_test": realistic_test,
            "construction_analysis": construction_analysis,
            "target_info": target_info,
            "conclusion": {
                "is_suspicious": leakage_analysis["leakage_risk"] == "HIGH"
                or realistic_test["suspicious"],
                "main_issue": "타겟 변수가 Big5 데이터로부터 직접 계산되어 데이터 누출 발생",
                "recommendation": "독립적인 타겟 변수 사용 필요",
            },
        }

        # 6. 결과 저장
        with open("suspicious_performance_analysis.json", "w") as f:

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
                elif isinstance(obj, pd.Series):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_json_serializable(item) for item in obj]
                else:
                    return obj

            json_results = convert_to_json_serializable(results)
            json.dump(json_results, f, indent=2)

        print(f"✅ 의심스러운 성능 분석 완료!")
        print(f"   의심스러운 성능: {results['conclusion']['is_suspicious']}")
        print(f"   주요 문제: {results['conclusion']['main_issue']}")
        print(f"   권장사항: {results['conclusion']['recommendation']}")

        return results


def main():
    """메인 실행 함수"""
    print("🚀 의심스러운 성능 분석 시스템 - R² 0.999+ 검증")
    print("=" * 60)

    analyzer = SuspiciousPerformanceAnalyzer()
    results = analyzer.run_suspicious_analysis(limit=10000)

    print("\n📊 의심스러운 성능 분석 결과:")
    print(f"   의심스러운 성능: {results['conclusion']['is_suspicious']}")
    print(f"   주요 문제: {results['conclusion']['main_issue']}")
    print(f"   권장사항: {results['conclusion']['recommendation']}")


if __name__ == "__main__":
    main()
