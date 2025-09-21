#!/usr/bin/env python3
"""
과적합 분석 시스템 - R² 0.9797 모델의 과적합 위험성 검증
- 실제 BigQuery 데이터로 훈련된 모델의 일반화 성능 평가
- 과적합 방지 전략 검증
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


class OverfittingAnalyzer:
    """과적합 분석 시스템"""

    def __init__(self, project_id: str = "persona-diary-service"):
        self.project_id = project_id
        self.scalers = {}
        self.models = {}
        self.ensemble_weights = None
        self.best_models = []
        self.cv_scores = {}

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

    def load_real_bigquery_data(self, limit: int = 10000) -> Tuple[Dict, np.ndarray]:
        """실제 BigQuery 데이터 로딩"""
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

            # Big5 데이터도 수치 데이터만 선택
            big5_numeric = big5_df.select_dtypes(include=[np.number])

            # 데이터 결합 (수치 데이터만 사용)
            multimodal_data = {
                "big5": big5_numeric.values,
                "cmi": cmi_numeric.values,
                "rppg": rppg_numeric.values,
                "voice": voice_numeric.values,
            }

            # 타겟 변수 생성 (Big5 점수 기반)
            big5_scores = {
                "EXT": big5_df[["EXT1", "EXT2", "EXT3", "EXT4", "EXT5"]].mean(axis=1),
                "EST": big5_df[["EST1", "EST2", "EST3", "EST4", "EST5"]].mean(axis=1),
                "AGR": big5_df[["AGR1", "AGR2", "AGR3", "AGR4", "AGR5"]].mean(axis=1),
                "CSN": big5_df[["CSN1", "CSN2", "CSN3", "CSN4", "CSN5"]].mean(axis=1),
                "OPN": big5_df[["OPN1", "OPN2", "OPN3", "OPN4", "OPN5"]].mean(axis=1),
            }

            # 타겟 변수 생성
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

            print(f"✅ 실제 BigQuery 데이터 로딩 완료:")
            print(f"   Big5: {big5_df.shape}")
            print(f"   CMI: {cmi_numeric.shape}")
            print(f"   RPPG: {rppg_numeric.shape}")
            print(f"   Voice: {voice_numeric.shape}")
            print(f"   Targets: {targets.shape}")

            return multimodal_data, targets

        except Exception as e:
            print(f"❌ BigQuery 데이터 로딩 실패: {str(e)}")
            raise e

    def prepare_data(
        self, multimodal_data: Dict, targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """데이터 준비"""
        print("🔄 데이터 준비 중...")

        # 모든 모달리티를 하나의 행렬로 결합
        X = np.concatenate(
            [
                multimodal_data["big5"],
                multimodal_data["cmi"],
                multimodal_data["rppg"],
                multimodal_data["voice"],
            ],
            axis=1,
        )

        # RobustScaler로 정규화
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers["robust_ensemble"] = scaler

        print(f"✅ 데이터 준비 완료: {X_scaled.shape}")
        return X_scaled, targets

    def create_models(self):
        """모델들 생성"""
        print("🔄 모델들 생성 중...")

        self.models = {
            "random_forest": RandomForestRegressor(
                n_estimators=150,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=10,
                random_state=42,
            ),
            "xgboost": XGBRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.7,
                colsample_bytree=0.7,
                random_state=42,
                n_jobs=-1,
            ),
            "lightgbm": LGBMRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.7,
                colsample_bytree=0.7,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
            "ridge": Ridge(alpha=5.0),
            "elastic_net": ElasticNet(alpha=0.05, l1_ratio=0.7),
            "svr": SVR(kernel="rbf", C=0.5, gamma="scale"),
        }

        print(f"✅ {len(self.models)}개 모델 생성 완료")

    def analyze_overfitting(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """과적합 분석"""
        print("🔍 과적합 분석 시작...")

        # 1. 훈련/검증/테스트 분할
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        print(f"   훈련 데이터: {X_train.shape}")
        print(f"   검증 데이터: {X_val.shape}")
        print(f"   테스트 데이터: {X_test.shape}")

        # 2. 교차 검증으로 과적합 분석
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        overfitting_analysis = {}

        for name, model in self.models.items():
            print(f"   분석 중: {name}")

            try:
                # 교차 검증 점수
                cv_scores = cross_val_score(
                    model, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1
                )

                # 훈련 데이터로 훈련
                model.fit(X_train, y_train)

                # 훈련 데이터 예측
                train_pred = model.predict(X_train)
                train_r2 = r2_score(y_train, train_pred)

                # 검증 데이터 예측
                val_pred = model.predict(X_val)
                val_r2 = r2_score(y_val, val_pred)

                # 테스트 데이터 예측
                test_pred = model.predict(X_test)
                test_r2 = r2_score(y_test, test_pred)

                # 과적합 지표 계산
                overfitting_gap = train_r2 - val_r2
                generalization_gap = val_r2 - test_r2

                overfitting_analysis[name] = {
                    "cv_mean_r2": cv_scores.mean(),
                    "cv_std_r2": cv_scores.std(),
                    "train_r2": train_r2,
                    "val_r2": val_r2,
                    "test_r2": test_r2,
                    "overfitting_gap": overfitting_gap,
                    "generalization_gap": generalization_gap,
                    "is_overfitting": overfitting_gap > 0.1,  # 10% 이상 차이면 과적합
                    "is_generalizing": abs(generalization_gap)
                    < 0.05,  # 5% 이내면 일반화 양호
                }

                print(f"     훈련 R²: {train_r2:.4f}")
                print(f"     검증 R²: {val_r2:.4f}")
                print(f"     테스트 R²: {test_r2:.4f}")
                print(f"     과적합 간격: {overfitting_gap:.4f}")
                print(f"     일반화 간격: {generalization_gap:.4f}")

            except Exception as e:
                print(f"     ❌ 분석 실패: {str(e)}")
                overfitting_analysis[name] = None

        return overfitting_analysis

    def analyze_data_quality(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """데이터 품질 분석"""
        print("🔍 데이터 품질 분석 중...")

        # 1. 데이터 분포 분석
        data_quality = {
            "n_samples": len(y),
            "n_features": X.shape[1],
            "feature_ratio": X.shape[1] / len(y),
            "target_stats": {
                "mean": float(y.mean()),
                "std": float(y.std()),
                "min": float(y.min()),
                "max": float(y.max()),
                "range": float(y.max() - y.min()),
            },
            "feature_stats": {
                "mean_std": float(X.std().mean()),
                "max_std": float(X.std().max()),
                "min_std": float(X.std().min()),
            },
        }

        # 2. 과적합 위험도 평가
        if data_quality["feature_ratio"] > 0.1:
            data_quality["overfitting_risk"] = "HIGH"
        elif data_quality["feature_ratio"] > 0.05:
            data_quality["overfitting_risk"] = "MEDIUM"
        else:
            data_quality["overfitting_risk"] = "LOW"

        print(f"   샘플 수: {data_quality['n_samples']}")
        print(f"   특성 수: {data_quality['n_features']}")
        print(f"   특성/샘플 비율: {data_quality['feature_ratio']:.4f}")
        print(f"   과적합 위험도: {data_quality['overfitting_risk']}")

        return data_quality

    def run_overfitting_analysis(self, limit: int = 10000) -> Dict:
        """과적합 분석 실행"""
        print("🚀 과적합 분석 시스템 시작")
        print("=" * 60)

        # 1. 실제 BigQuery 데이터 로딩
        multimodal_data, targets = self.load_real_bigquery_data(limit)

        # 2. 데이터 준비
        X, y = self.prepare_data(multimodal_data, targets)

        # 3. 모델들 생성
        self.create_models()

        # 4. 과적합 분석
        overfitting_analysis = self.analyze_overfitting(X, y)

        # 5. 데이터 품질 분석
        data_quality = self.analyze_data_quality(X, y)

        # 6. 결과 종합
        results = {
            "overfitting_analysis": overfitting_analysis,
            "data_quality": data_quality,
            "summary": {
                "total_models": len(
                    [m for m in overfitting_analysis.values() if m is not None]
                ),
                "overfitting_models": len(
                    [
                        m
                        for m in overfitting_analysis.values()
                        if m and m.get("is_overfitting", False)
                    ]
                ),
                "generalizing_models": len(
                    [
                        m
                        for m in overfitting_analysis.values()
                        if m and m.get("is_generalizing", False)
                    ]
                ),
                "avg_overfitting_gap": np.mean(
                    [m["overfitting_gap"] for m in overfitting_analysis.values() if m]
                ),
                "avg_generalization_gap": np.mean(
                    [
                        m["generalization_gap"]
                        for m in overfitting_analysis.values()
                        if m
                    ]
                ),
            },
        }

        # 7. 결과 저장
        with open("overfitting_analysis_results.json", "w") as f:

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

        print(f"✅ 과적합 분석 완료!")
        print(f"   총 모델 수: {results['summary']['total_models']}")
        print(f"   과적합 모델 수: {results['summary']['overfitting_models']}")
        print(f"   일반화 양호 모델 수: {results['summary']['generalizing_models']}")
        print(f"   평균 과적합 간격: {results['summary']['avg_overfitting_gap']:.4f}")
        print(
            f"   평균 일반화 간격: {results['summary']['avg_generalization_gap']:.4f}"
        )

        return results


def main():
    """메인 실행 함수"""
    print("🚀 과적합 분석 시스템 - R² 0.9797 모델 검증")
    print("=" * 60)

    analyzer = OverfittingAnalyzer()
    results = analyzer.run_overfitting_analysis(limit=10000)

    print("\n📊 과적합 분석 결과:")
    print(f"   과적합 모델 수: {results['summary']['overfitting_models']}")
    print(f"   일반화 양호 모델 수: {results['summary']['generalizing_models']}")
    print(f"   평균 과적합 간격: {results['summary']['avg_overfitting_gap']:.4f}")
    print(f"   평균 일반화 간격: {results['summary']['avg_generalization_gap']:.4f}")


if __name__ == "__main__":
    main()
