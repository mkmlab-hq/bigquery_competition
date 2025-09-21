#!/usr/bin/env python3
"""
실제 BigQuery 데이터 전용 훈련 시스템
- 시뮬레이션 데이터 완전 제거
- 실제 BigQuery 데이터만 사용
- 과적합 방지 및 상위권 진입
- R² 0.70+ 목표
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


class RealDataOnlyTrainer:
    """실제 BigQuery 데이터 전용 훈련 시스템"""

    def __init__(self, project_id: str = "persona-diary-service"):
        self.project_id = project_id
        self.scalers = {}
        self.models = {}
        self.ensemble_weights = None
        self.best_models = []
        self.cv_scores = {}

        # 인증 파일 경로 설정 (현재 디렉토리)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
            "F:/workspace/bigquery_competition/optimization/gcs-key.json"
        )

        try:
            self.client = bigquery.Client(project=project_id)
            print(f"✅ BigQuery 클라이언트 초기화 완료: {project_id}")
        except Exception as e:
            print(f"❌ BigQuery 인증 실패: {str(e)}")
            print("인증 파일 경로를 확인해주세요.")
            raise e

    def load_real_bigquery_data(self, limit: int = 10000) -> Tuple[Dict, np.ndarray]:
        """실제 BigQuery 데이터만 로딩 (시뮬레이션 데이터 없음)"""
        print("🔄 실제 BigQuery 데이터 로딩 중...")

        try:
            # Big5 데이터 쿼리
            big5_query = f"""
            SELECT 
                EXT_1, EXT_2, EXT_3, EXT_4, EXT_5,
                EST_1, EST_2, EST_3, EST_4, EST_5,
                AGR_1, AGR_2, AGR_3, AGR_4, AGR_5,
                CSN_1, CSN_2, CSN_3, CSN_4, CSN_5,
                OPN_1, OPN_2, OPN_3, OPN_4, OPN_5
            FROM `{self.project_id}.persona_diary.big5_scores`
            LIMIT {limit}
            """

            # CMI 데이터 쿼리
            cmi_query = f"""
            SELECT 
                CMI_1, CMI_2, CMI_3, CMI_4, CMI_5,
                CMI_6, CMI_7, CMI_8, CMI_9, CMI_10
            FROM `{self.project_id}.persona_diary.cmi_scores`
            LIMIT {limit}
            """

            # RPPG 데이터 쿼리
            rppg_query = f"""
            SELECT 
                RPPG_1, RPPG_2, RPPG_3, RPPG_4, RPPG_5,
                RPPG_6, RPPG_7, RPPG_8, RPPG_9, RPPG_10,
                RPPG_11, RPPG_12, RPPG_13, RPPG_14, RPPG_15
            FROM `{self.project_id}.persona_diary.rppg_scores`
            LIMIT {limit}
            """

            # Voice 데이터 쿼리
            voice_query = f"""
            SELECT 
                VOICE_1, VOICE_2, VOICE_3, VOICE_4, VOICE_5,
                VOICE_6, VOICE_7, VOICE_8, VOICE_9, VOICE_10,
                VOICE_11, VOICE_12, VOICE_13, VOICE_14, VOICE_15,
                VOICE_16, VOICE_17, VOICE_18, VOICE_19, VOICE_20
            FROM `{self.project_id}.persona_diary.voice_scores`
            LIMIT {limit}
            """

            # 타겟 변수 쿼리
            target_query = f"""
            SELECT 
                PERSONALITY_SCORE
            FROM `{self.project_id}.persona_diary.personality_scores`
            LIMIT {limit}
            """

            # 데이터 로딩
            big5_df = self.client.query(big5_query).to_dataframe()
            cmi_df = self.client.query(cmi_query).to_dataframe()
            rppg_df = self.client.query(rppg_query).to_dataframe()
            voice_df = self.client.query(voice_query).to_dataframe()
            target_df = self.client.query(target_query).to_dataframe()

            # 데이터 결합
            multimodal_data = {
                "big5": big5_df.values,
                "cmi": cmi_df.values,
                "rppg": rppg_df.values,
                "voice": voice_df.values,
            }

            targets = target_df.values.flatten()

            print(f"✅ 실제 BigQuery 데이터 로딩 완료:")
            print(f"   Big5: {big5_df.shape}")
            print(f"   CMI: {cmi_df.shape}")
            print(f"   RPPG: {rppg_df.shape}")
            print(f"   Voice: {voice_df.shape}")
            print(f"   Targets: {targets.shape}")

            return multimodal_data, targets

        except Exception as e:
            print(f"❌ BigQuery 데이터 로딩 실패: {str(e)}")
            print("인증 파일 경로를 확인해주세요.")
            raise e

    def create_robust_models(self):
        """강건한 모델들 생성 (과적합 방지)"""
        print("🔄 강건한 모델들 생성 중...")

        self.models = {
            # 트리 기반 모델들 (과적합 방지 설정)
            "random_forest": RandomForestRegressor(
                n_estimators=100,  # 감소
                max_depth=10,  # 감소
                min_samples_split=10,  # 증가
                min_samples_leaf=5,  # 증가
                max_features="sqrt",  # 추가
                random_state=42,
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=100,  # 감소
                learning_rate=0.05,  # 감소
                max_depth=6,  # 감소
                min_samples_split=10,  # 증가
                subsample=0.8,  # 추가
                random_state=42,
            ),
            "xgboost": XGBRegressor(
                n_estimators=100,  # 감소
                learning_rate=0.05,  # 감소
                max_depth=6,  # 감소
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,  # L1 정규화
                reg_lambda=0.1,  # L2 정규화
                random_state=42,
            ),
            "lightgbm": LGBMRegressor(
                n_estimators=100,  # 감소
                learning_rate=0.05,  # 감소
                max_depth=6,  # 감소
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,  # L1 정규화
                reg_lambda=0.1,  # L2 정규화
                random_state=42,
                verbose=-1,
            ),
            # 선형 모델들 (정규화 강화)
            "ridge": Ridge(alpha=10.0),  # 정규화 강화
            "elastic_net": ElasticNet(alpha=1.0, l1_ratio=0.5),  # 정규화 강화
            # 서포트 벡터 머신 (정규화 강화)
            "svr": SVR(kernel="rbf", C=0.1, gamma="scale"),  # 정규화 강화
        }

        print(f"✅ {len(self.models)}개 강건한 모델 생성 완료")

    def prepare_robust_data(
        self, multimodal_data: Dict, targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """강건한 데이터 준비"""
        print("🔄 강건한 데이터 준비 중...")

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

        # RobustScaler로 정규화 (이상치에 강함)
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers["robust"] = scaler

        print(f"✅ 강건한 데이터 준비 완료: {X_scaled.shape}")
        return X_scaled, targets

    def train_models_with_cv(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """교차 검증으로 모델 훈련"""
        print("🚀 교차 검증으로 모델 훈련 시작...")

        # 5-fold 교차 검증
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        model_scores = {}

        for name, model in self.models.items():
            print(f"   훈련 중: {name}")

            try:
                # 교차 검증 점수 계산
                cv_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
                avg_score = cv_scores.mean()
                std_score = cv_scores.std()

                # 모델 훈련
                model.fit(X, y)

                model_scores[name] = {
                    "cv_mean": avg_score,
                    "cv_std": std_score,
                    "cv_scores": cv_scores,
                    "model": model,
                }

                self.cv_scores[name] = {
                    "mean": avg_score,
                    "std": std_score,
                    "scores": cv_scores,
                }

                print(f"     R²: {avg_score:.4f} (±{std_score:.4f})")

            except Exception as e:
                print(f"     ❌ 훈련 실패: {str(e)}")
                model_scores[name] = None

        # 성능 순으로 정렬
        valid_models = {k: v for k, v in model_scores.items() if v is not None}
        sorted_models = sorted(
            valid_models.items(), key=lambda x: x[1]["cv_mean"], reverse=True
        )

        print(f"✅ {len(valid_models)}개 모델 훈련 완료")
        print("📊 모델 성능 순위:")
        for i, (name, scores) in enumerate(sorted_models[:5], 1):
            print(f"   {i}. {name}: R² = {scores['cv_mean']:.4f}")

        return model_scores

    def select_stable_models(
        self, model_scores: Dict, stability_threshold: float = 0.02
    ) -> List[str]:
        """안정적인 모델들 선택"""
        print("🔄 안정적인 모델들 선택 중...")

        stable_models = []
        for name, scores in model_scores.items():
            if scores is not None:
                # 표준편차가 낮고 평균 점수가 높은 모델 선택
                if scores["cv_std"] < stability_threshold and scores["cv_mean"] > 0.3:
                    stable_models.append(name)
                    print(
                        f"   ✅ {name}: R² = {scores['cv_mean']:.4f} (±{scores['cv_std']:.4f})"
                    )

        print(f"✅ {len(stable_models)}개 안정적인 모델 선택 완료")
        return stable_models

    def create_robust_ensemble(
        self, X: np.ndarray, y: np.ndarray, model_scores: Dict, stable_models: List[str]
    ) -> Dict:
        """강건한 앙상블 생성"""
        print("🔄 강건한 앙상블 생성 중...")

        if len(stable_models) < 2:
            print("❌ 앙상블을 위한 충분한 안정적인 모델이 없습니다.")
            return None

        # 안정적인 모델들만 사용
        ensemble_models = {name: model_scores[name] for name in stable_models}

        # 가중치 계산 (성능과 안정성 모두 고려)
        weights = []
        for name in stable_models:
            scores = model_scores[name]
            # 성능과 안정성을 모두 고려한 점수
            stability_score = 1.0 / (1.0 + scores["cv_std"])
            performance_score = scores["cv_mean"]
            combined_score = performance_score * stability_score
            weights.append(combined_score)

        # 정규화
        weights = np.array(weights)
        weights = weights / weights.sum()

        self.ensemble_weights = dict(zip(stable_models, weights))

        print("✅ 강건한 앙상블 생성 완료:")
        for name, weight in self.ensemble_weights.items():
            print(f"   {name}: {weight:.4f}")

        # 앙상블 예측 생성
        predictions = []
        weights_list = []

        for name, weight in self.ensemble_weights.items():
            if name in model_scores and model_scores[name] is not None:
                model = model_scores[name]["model"]
                pred = model.predict(X)
                predictions.append(pred)
                weights_list.append(weight)

        if not predictions:
            print("❌ 유효한 예측값이 없습니다.")
            return None

        # 가중 평균으로 앙상블 예측
        predictions = np.array(predictions)
        weights_list = np.array(weights_list)

        ensemble_pred = np.average(predictions, axis=0, weights=weights_list)

        # 성능 메트릭 계산
        r2 = r2_score(y, ensemble_pred)
        mse = mean_squared_error(y, ensemble_pred)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(ensemble_pred - y))
        correlation = np.corrcoef(ensemble_pred, y)[0, 1]

        results = {
            "r2": r2,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "correlation": correlation,
            "predictions": ensemble_pred,
            "targets": y,
            "ensemble_weights": self.ensemble_weights,
            "stable_models": stable_models,
        }

        print(f"✅ 강건한 앙상블 평가 완료:")
        print(f"   R²: {r2:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   상관계수: {correlation:.4f}")

        return results

    def run_real_data_training(self, limit: int = 10000) -> Dict:
        """실제 데이터 전용 훈련 실행"""
        print("🚀 실제 BigQuery 데이터 전용 훈련 시스템 시작")
        print("=" * 60)

        # 1. 실제 BigQuery 데이터만 로딩 (시뮬레이션 데이터 없음)
        multimodal_data, targets = self.load_real_bigquery_data(limit)

        # 2. 강건한 모델들 생성
        self.create_robust_models()

        # 3. 데이터 준비
        X, y = self.prepare_robust_data(multimodal_data, targets)

        # 4. 교차 검증으로 모델 훈련
        model_scores = self.train_models_with_cv(X, y)

        # 5. 안정적인 모델들 선택
        stable_models = self.select_stable_models(model_scores)

        # 6. 강건한 앙상블 생성
        ensemble_results = self.create_robust_ensemble(
            X, y, model_scores, stable_models
        )

        # 7. 결과 저장
        results = {
            "ensemble_results": ensemble_results,
            "individual_models": {
                name: {
                    "cv_mean": scores["cv_mean"] if scores else None,
                    "cv_std": scores["cv_std"] if scores else None,
                }
                for name, scores in model_scores.items()
            },
            "stable_models": stable_models,
            "cv_scores": self.cv_scores,
            "data_info": {
                "n_samples": len(y),
                "n_features": X.shape[1],
                "n_models": len([m for m in model_scores.values() if m is not None]),
                "n_stable_models": len(stable_models),
                "data_source": "real_bigquery_only",
            },
        }

        # JSON으로 저장
        with open("real_data_only_results.json", "w") as f:

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
                elif isinstance(obj, dict):
                    return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_json_serializable(item) for item in obj]
                else:
                    return obj

            json_results = convert_to_json_serializable(results)
            json.dump(json_results, f, indent=2)

        print(f"✅ 실제 데이터 전용 훈련 완료!")
        if ensemble_results:
            print(f"   최종 R²: {ensemble_results['r2']:.4f}")
            print(f"   최종 RMSE: {ensemble_results['rmse']:.4f}")

        return results


def main():
    """메인 실행 함수"""
    print("🚀 실제 BigQuery 데이터 전용 훈련 시스템 - R² 0.70+ 도전")
    print("=" * 60)

    # 실제 데이터 전용 훈련 시스템 초기화
    trainer = RealDataOnlyTrainer()

    # 실제 데이터 전용 훈련 실행
    results = trainer.run_real_data_training(limit=10000)

    print("\n📊 실제 데이터 전용 훈련 결과:")
    if results["ensemble_results"]:
        print(f"   앙상블 R²: {results['ensemble_results']['r2']:.4f}")
        print(f"   앙상블 RMSE: {results['ensemble_results']['rmse']:.4f}")
        print(f"   상관계수: {results['ensemble_results']['correlation']:.4f}")
        print(f"   안정적인 모델 수: {len(results['stable_models'])}")


if __name__ == "__main__":
    main()
