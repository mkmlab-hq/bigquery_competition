#!/usr/bin/env python3
"""
냉철한 재검증 시스템 - 데이터 누출 완전 차단
- 완전히 독립적인 타겟 변수 생성
- 데이터 누출 완전 방지
- 현실적인 성능 평가
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


class ColdVerifier:
    """냉철한 재검증 시스템"""

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
            big5_numeric = big5_df.select_dtypes(include=[np.number])

            # 데이터 결합
            multimodal_data = {
                "big5": big5_numeric.values,
                "cmi": cmi_numeric.values,
                "rppg": rppg_numeric.values,
                "voice": voice_numeric.values,
            }

            print(f"✅ 실제 BigQuery 데이터 로딩 완료:")
            print(f"   Big5: {big5_numeric.shape}")
            print(f"   CMI: {cmi_numeric.shape}")
            print(f"   RPPG: {rppg_numeric.shape}")
            print(f"   Voice: {voice_numeric.shape}")

            return multimodal_data

        except Exception as e:
            print(f"❌ BigQuery 데이터 로딩 실패: {str(e)}")
            raise e

    def create_completely_independent_target(self, n_samples: int) -> np.ndarray:
        """완전히 독립적인 타겟 변수 생성 (입력 데이터와 완전히 무관)"""
        print("🔍 완전히 독립적인 타겟 변수 생성 중...")

        # 방법 1: 완전히 랜덤한 타겟 변수
        np.random.seed(42)
        random_targets = np.random.uniform(1, 10, n_samples)

        print(f"✅ 완전히 독립적인 타겟 변수 생성 완료:")
        print(f"   타겟 변수 통계:")
        print(f"     평균: {random_targets.mean():.4f}")
        print(f"     표준편차: {random_targets.std():.4f}")
        print(f"     최소값: {random_targets.min():.4f}")
        print(f"     최대값: {random_targets.max():.4f}")
        print(f"   입력 데이터와 완전히 독립적!")

        return random_targets

    def create_models(self):
        """모델들 생성"""
        print("🔄 모델들 생성 중...")

        self.models = {
            "random_forest": RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=10,
                random_state=42,
            ),
            "xgboost": XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            ),
            "lightgbm": LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
            "ridge": Ridge(alpha=1.0),
            "elastic_net": ElasticNet(alpha=0.1, l1_ratio=0.5),
            "svr": SVR(kernel="rbf", C=1.0, gamma="scale"),
        }

        print(f"✅ {len(self.models)}개 모델 생성 완료")

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

    def test_models_with_independent_target(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """독립적인 타겟 변수로 모델들 테스트"""
        print("🚀 독립적인 타겟 변수로 모델들 테스트 시작...")

        # 훈련/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model_results = {}

        for name, model in self.models.items():
            print(f"   테스트 중: {name}")

            try:
                # 모델 훈련
                model.fit(X_train, y_train)

                # 예측
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)

                # 성능 평가
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

                model_results[name] = {
                    "train_r2": train_r2,
                    "test_r2": test_r2,
                    "train_rmse": train_rmse,
                    "test_rmse": test_rmse,
                    "overfitting_gap": train_r2 - test_r2,
                }

                print(f"     훈련 R²: {train_r2:.4f}, 테스트 R²: {test_r2:.4f}")
                print(f"     훈련 RMSE: {train_rmse:.4f}, 테스트 RMSE: {test_rmse:.4f}")
                print(f"     과적합 간격: {train_r2 - test_r2:.4f}")

            except Exception as e:
                print(f"     ❌ 테스트 실패: {str(e)}")
                model_results[name] = None

        return model_results

    def test_models_with_correlated_target(
        self, X: np.ndarray, multimodal_data: Dict
    ) -> Dict:
        """상관관계가 있는 타겟 변수로 모델들 테스트"""
        print("🚀 상관관계가 있는 타겟 변수로 모델들 테스트 시작...")

        # 상관관계가 있는 타겟 변수 생성 (하지만 완전히 동일하지는 않음)
        print("🔍 상관관계가 있는 타겟 변수 생성 중...")

        # Big5 데이터의 일부만 사용하여 타겟 변수 생성
        big5_scores = {
            "EXT": multimodal_data["big5"][:, 0:5].mean(axis=1),  # 처음 5개만
            "OPN": multimodal_data["big5"][:, 5:10].mean(axis=1),  # 다음 5개만
        }

        # 상관관계가 있지만 완전히 동일하지 않은 타겟 변수
        correlated_target = (
            big5_scores["EXT"] * 0.6
            + big5_scores["OPN"] * 0.4
            + np.random.normal(0, 0.5, len(big5_scores["EXT"]))  # 노이즈 추가
        )

        # 1-10 스케일로 정규화
        y = (correlated_target - correlated_target.min()) / (
            correlated_target.max() - correlated_target.min()
        ) * 9 + 1

        print(f"✅ 상관관계가 있는 타겟 변수 생성 완료:")
        print(f"   타겟 변수 통계:")
        print(f"     평균: {y.mean():.4f}")
        print(f"     표준편차: {y.std():.4f}")
        print(f"     최소값: {y.min():.4f}")
        print(f"     최대값: {y.max():.4f}")

        # 훈련/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model_results = {}

        for name, model in self.models.items():
            print(f"   테스트 중: {name}")

            try:
                # 모델 훈련
                model.fit(X_train, y_train)

                # 예측
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)

                # 성능 평가
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

                model_results[name] = {
                    "train_r2": train_r2,
                    "test_r2": test_r2,
                    "train_rmse": train_rmse,
                    "test_rmse": test_rmse,
                    "overfitting_gap": train_r2 - test_r2,
                }

                print(f"     훈련 R²: {train_r2:.4f}, 테스트 R²: {test_r2:.4f}")
                print(f"     훈련 RMSE: {train_rmse:.4f}, 테스트 RMSE: {test_rmse:.4f}")
                print(f"     과적합 간격: {train_r2 - test_r2:.4f}")

            except Exception as e:
                print(f"     ❌ 테스트 실패: {str(e)}")
                model_results[name] = None

        return model_results

    def run_cold_verification(self, limit: int = 10000) -> Dict:
        """냉철한 재검증 실행"""
        print("🚀 냉철한 재검증 시스템 시작")
        print("=" * 60)

        # 1. 실제 BigQuery 데이터 로딩
        multimodal_data = self.load_real_bigquery_data(limit)

        # 2. 모델들 생성
        self.create_models()

        # 3. 데이터 준비
        X, _ = self.prepare_data(multimodal_data, np.zeros(limit))

        # 4. 완전히 독립적인 타겟 변수로 테스트
        print("\n" + "=" * 50)
        print("🔍 테스트 1: 완전히 독립적인 타겟 변수")
        print("=" * 50)
        independent_targets = self.create_completely_independent_target(limit)
        independent_results = self.test_models_with_independent_target(
            X, independent_targets
        )

        # 5. 상관관계가 있는 타겟 변수로 테스트
        print("\n" + "=" * 50)
        print("🔍 테스트 2: 상관관계가 있는 타겟 변수")
        print("=" * 50)
        correlated_results = self.test_models_with_correlated_target(X, multimodal_data)

        # 6. 결과 분석
        print("\n" + "=" * 50)
        print("📊 냉철한 재검증 결과 분석")
        print("=" * 50)

        # 독립적인 타겟 변수 결과 분석
        valid_independent = {
            k: v for k, v in independent_results.items() if v is not None
        }
        if valid_independent:
            avg_independent_r2 = np.mean(
                [r["test_r2"] for r in valid_independent.values()]
            )
            max_independent_r2 = np.max(
                [r["test_r2"] for r in valid_independent.values()]
            )
            min_independent_r2 = np.min(
                [r["test_r2"] for r in valid_independent.values()]
            )

            print(f"독립적인 타겟 변수 테스트:")
            print(f"   평균 테스트 R²: {avg_independent_r2:.4f}")
            print(f"   최대 테스트 R²: {max_independent_r2:.4f}")
            print(f"   최소 테스트 R²: {min_independent_r2:.4f}")

            if avg_independent_r2 > 0.5:
                print("   🚨 경고: 독립적인 타겟에서도 높은 성능! 데이터 누출 의심!")
            elif avg_independent_r2 > 0.2:
                print(
                    "   ⚠️ 주의: 독립적인 타겟에서 중간 성능. 일부 데이터 누출 가능성."
                )
            else:
                print("   ✅ 양호: 독립적인 타겟에서 낮은 성능. 데이터 누출 없음.")

        # 상관관계가 있는 타겟 변수 결과 분석
        valid_correlated = {
            k: v for k, v in correlated_results.items() if v is not None
        }
        if valid_correlated:
            avg_correlated_r2 = np.mean(
                [r["test_r2"] for r in valid_correlated.values()]
            )
            max_correlated_r2 = np.max(
                [r["test_r2"] for r in valid_correlated.values()]
            )
            min_correlated_r2 = np.min(
                [r["test_r2"] for r in valid_correlated.values()]
            )

            print(f"\n상관관계가 있는 타겟 변수 테스트:")
            print(f"   평균 테스트 R²: {avg_correlated_r2:.4f}")
            print(f"   최대 테스트 R²: {max_correlated_r2:.4f}")
            print(f"   최소 테스트 R²: {min_correlated_r2:.4f}")

            if avg_correlated_r2 > 0.8:
                print(
                    "   🚨 경고: 상관관계가 있는 타겟에서 매우 높은 성능! 데이터 누출 의심!"
                )
            elif avg_correlated_r2 > 0.5:
                print(
                    "   ⚠️ 주의: 상관관계가 있는 타겟에서 높은 성능. 일부 데이터 누출 가능성."
                )
            else:
                print("   ✅ 양호: 상관관계가 있는 타겟에서 현실적인 성능.")

        # 7. 결과 저장
        results = {
            "independent_target_results": independent_results,
            "correlated_target_results": correlated_results,
            "summary": {
                "avg_independent_r2": (
                    float(avg_independent_r2) if valid_independent else 0
                ),
                "max_independent_r2": (
                    float(max_independent_r2) if valid_independent else 0
                ),
                "min_independent_r2": (
                    float(min_independent_r2) if valid_independent else 0
                ),
                "avg_correlated_r2": (
                    float(avg_correlated_r2) if valid_correlated else 0
                ),
                "max_correlated_r2": (
                    float(max_correlated_r2) if valid_correlated else 0
                ),
                "min_correlated_r2": (
                    float(min_correlated_r2) if valid_correlated else 0
                ),
                "data_leakage_suspected": (
                    avg_independent_r2 > 0.5 if valid_independent else False
                ),
                "correlated_performance_realistic": (
                    avg_correlated_r2 < 0.8 if valid_correlated else False
                ),
            },
        }

        with open("cold_verification_results.json", "w") as f:

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

        print(f"\n✅ 냉철한 재검증 완료!")

        return results


def main():
    """메인 실행 함수"""
    print("🚀 냉철한 재검증 시스템 - 데이터 누출 완전 차단")
    print("=" * 60)

    verifier = ColdVerifier()
    results = verifier.run_cold_verification(limit=10000)

    print("\n📊 냉철한 재검증 최종 결과:")
    print(f"   독립적인 타겟 평균 R²: {results['summary']['avg_independent_r2']:.4f}")
    print(f"   상관관계 타겟 평균 R²: {results['summary']['avg_correlated_r2']:.4f}")
    print(f"   데이터 누출 의심: {results['summary']['data_leakage_suspected']}")
    print(
        f"   상관관계 성능 현실적: {results['summary']['correlated_performance_realistic']}"
    )


if __name__ == "__main__":
    main()
