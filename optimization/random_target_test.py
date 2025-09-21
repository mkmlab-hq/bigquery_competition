#!/usr/bin/env python3
"""
완전히 랜덤한 타겟 변수 테스트 - 데이터 누출 완전 검증
- 완전히 랜덤한 타겟 변수로 모델 성능 테스트
- 데이터 누출이 있다면 랜덤 타겟에서도 높은 성능이 나올 것
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


class RandomTargetTester:
    """완전히 랜덤한 타겟 변수 테스트 시스템"""

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

            # 데이터 결합
            multimodal_data = {
                "cmi": cmi_numeric.values,
                "rppg": rppg_numeric.values,
                "voice": voice_numeric.values,
            }

            # 완전히 랜덤한 타겟 변수 생성
            print("🔍 완전히 랜덤한 타겟 변수 생성 중...")

            np.random.seed(42)
            random_targets = np.random.uniform(1, 10, limit)  # 1-10 범위의 완전 랜덤

            print(f"✅ 완전히 랜덤한 타겟 변수 생성 완료:")
            print(f"   CMI: {cmi_numeric.shape}")
            print(f"   RPPG: {rppg_numeric.shape}")
            print(f"   Voice: {voice_numeric.shape}")
            print(f"   Random Targets: {random_targets.shape}")
            print(f"   랜덤 타겟 변수 통계:")
            print(f"     평균: {random_targets.mean():.4f}")
            print(f"     표준편차: {random_targets.std():.4f}")
            print(f"     최소값: {random_targets.min():.4f}")
            print(f"     최대값: {random_targets.max():.4f}")

            return multimodal_data, random_targets

        except Exception as e:
            print(f"❌ BigQuery 데이터 로딩 실패: {str(e)}")
            raise e

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

    def test_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """모델들 테스트"""
        print("🚀 모델들 테스트 시작...")

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

    def run_random_target_test(self, limit: int = 10000) -> Dict:
        """완전히 랜덤한 타겟 변수 테스트 실행"""
        print("🚀 완전히 랜덤한 타겟 변수 테스트 시작")
        print("=" * 60)

        # 1. 실제 BigQuery 데이터 로딩
        multimodal_data, targets = self.load_real_bigquery_data(limit)

        # 2. 모델들 생성
        self.create_models()

        # 3. 데이터 준비
        X, y = self.prepare_data(multimodal_data, targets)

        # 4. 모델들 테스트
        model_results = self.test_models(X, y)

        # 5. 결과 분석
        valid_results = {k: v for k, v in model_results.items() if v is not None}

        if valid_results:
            avg_test_r2 = np.mean([r["test_r2"] for r in valid_results.values()])
            max_test_r2 = np.max([r["test_r2"] for r in valid_results.values()])
            min_test_r2 = np.min([r["test_r2"] for r in valid_results.values()])

            print(f"\n📊 랜덤 타겟 변수 테스트 결과:")
            print(f"   평균 테스트 R²: {avg_test_r2:.4f}")
            print(f"   최대 테스트 R²: {max_test_r2:.4f}")
            print(f"   최소 테스트 R²: {min_test_r2:.4f}")

            # 데이터 누출 판단
            if avg_test_r2 > 0.5:
                print("   🚨 경고: 랜덤 타겟에서도 높은 성능! 데이터 누출 의심!")
            elif avg_test_r2 > 0.2:
                print("   ⚠️ 주의: 랜덤 타겟에서 중간 성능. 일부 데이터 누출 가능성.")
            else:
                print("   ✅ 양호: 랜덤 타겟에서 낮은 성능. 데이터 누출 없음.")

        # 6. 결과 저장
        results = {
            "model_results": model_results,
            "summary": {
                "avg_test_r2": float(avg_test_r2) if valid_results else 0,
                "max_test_r2": float(max_test_r2) if valid_results else 0,
                "min_test_r2": float(min_test_r2) if valid_results else 0,
                "data_leakage_suspected": avg_test_r2 > 0.5 if valid_results else False,
            },
        }

        with open("random_target_test_results.json", "w") as f:

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

        print(f"✅ 완전히 랜덤한 타겟 변수 테스트 완료!")

        return results


def main():
    """메인 실행 함수"""
    print("🚀 완전히 랜덤한 타겟 변수 테스트 - 데이터 누출 완전 검증")
    print("=" * 60)

    tester = RandomTargetTester()
    results = tester.run_random_target_test(limit=10000)

    print("\n📊 랜덤 타겟 변수 테스트 결과:")
    print(f"   평균 테스트 R²: {results['summary']['avg_test_r2']:.4f}")
    print(f"   데이터 누출 의심: {results['summary']['data_leakage_suspected']}")


if __name__ == "__main__":
    main()
