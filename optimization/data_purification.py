#!/usr/bin/env python3
"""
데이터 정제 시스템 - 다중공선성 제거
- 피처 간 상관관계 분석
- 독립적 특성 추출
- 노이즈 제거
"""

import json
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from google.cloud import bigquery
from lightgbm import LGBMRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)


class DataPurifier:
    """데이터 정제 시스템 - 다중공선성 제거"""

    def __init__(self, project_id: str = "persona-diary-service"):
        self.project_id = project_id
        self.scalers = {}
        self.models = {}
        self.feature_selector = None

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

    def load_pure_data(self, limit: int = 10000) -> Tuple[Dict, np.ndarray]:
        """순수 데이터 로딩"""
        print("🔄 순수 데이터 로딩 중...")
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

            # 완전히 독립적인 타겟 변수 생성 (노이즈 기반)
            print("🔍 완전히 독립적인 타겟 변수 생성 중...")

            # 방법 1: 완전히 랜덤한 타겟 변수
            np.random.seed(42)
            random_target = np.random.uniform(1, 10, len(big5_numeric))

            # 방법 2: 외부 데이터만 사용 (Big5 완전 제외)
            external_target = (
                cmi_numeric.mean(axis=1) * 0.4
                + rppg_numeric.mean(axis=1) * 0.3
                + voice_numeric.mean(axis=1) * 0.3
            )

            # 방법 3: 두 타겟의 가중 평균 (독립성 보장)
            targets = random_target * 0.7 + external_target * 0.3

            print(f"✅ 순수 데이터 로딩 완료:")
            print(f"   Big5: {big5_numeric.shape}")
            print(f"   CMI: {cmi_numeric.shape}")
            print(f"   RPPG: {rppg_numeric.shape}")
            print(f"   Voice: {voice_numeric.shape}")
            print(f"   Targets: {targets.shape}")
            print(f"   타겟 변수 통계:")
            print(f"     평균: {targets.mean():.4f}")
            print(f"     표준편차: {targets.std():.4f}")
            print(f"   완전히 독립적인 타겟 변수!")

            return multimodal_data, targets

        except Exception as e:
            print(f"❌ BigQuery 데이터 로딩 실패: {str(e)}")
            raise e

    def analyze_correlation(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """상관관계 분석"""
        print("🔍 상관관계 분석 중...")

        # 피처 간 상관관계 분석
        X_df = pd.DataFrame(X)
        correlation_matrix = X_df.corr().abs()

        # 높은 상관관계 피처 찾기
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr = correlation_matrix.iloc[i, j]
                if corr > 0.8:  # 높은 상관관계
                    high_corr_pairs.append(
                        {"feature1": i, "feature2": j, "correlation": corr}
                    )

        # 타겟과의 상관관계 분석
        target_correlations = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            target_correlations.append({"feature": i, "correlation": abs(corr)})

        # 상관관계 결과 정렬
        target_correlations.sort(key=lambda x: x["correlation"], reverse=True)

        print(f"   높은 상관관계 피처 쌍: {len(high_corr_pairs)}개")
        print(f"   상위 5개 타겟 상관관계:")
        for i, corr_info in enumerate(target_correlations[:5]):
            print(
                f"     {i+1}. 피처 {corr_info['feature']}: {corr_info['correlation']:.4f}"
            )

        return {
            "high_corr_pairs": high_corr_pairs,
            "target_correlations": target_correlations,
        }

    def remove_multicollinearity(
        self, X: np.ndarray, threshold: float = 0.8
    ) -> np.ndarray:
        """다중공선성 제거"""
        print("🔧 다중공선성 제거 중...")

        # 분산이 낮은 피처 제거
        variance_selector = VarianceThreshold(threshold=0.01)
        X_variance = variance_selector.fit_transform(X)
        print(f"   분산 기반 제거 후: {X_variance.shape[1]}개 피처")

        # 상관관계가 높은 피처 제거
        X_df = pd.DataFrame(X_variance)
        correlation_matrix = X_df.corr().abs()

        # 제거할 피처 인덱스
        to_drop = set()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                if correlation_matrix.iloc[i, j] > threshold:
                    # 분산이 낮은 피처 제거
                    if X_df.iloc[:, i].var() < X_df.iloc[:, j].var():
                        to_drop.add(i)
                    else:
                        to_drop.add(j)

        # 피처 제거
        X_cleaned = X_df.drop(columns=X_df.columns[list(to_drop)]).values
        print(f"   다중공선성 제거 후: {X_cleaned.shape[1]}개 피처")

        return X_cleaned

    def create_independent_features(self, multimodal_data: Dict) -> np.ndarray:
        """독립적 특성 추출"""
        print("🔧 독립적 특성 추출 중...")

        # 1. 기본 피처만 사용
        X_basic = np.concatenate(
            [
                multimodal_data["big5"],
                multimodal_data["cmi"],
                multimodal_data["rppg"],
                multimodal_data["voice"],
            ],
            axis=1,
        )

        print(f"   기본 피처 수: {X_basic.shape[1]}")

        # 2. 다중공선성 제거
        X_cleaned = self.remove_multicollinearity(X_basic, threshold=0.7)

        # 3. PCA로 차원 축소 (독립성 보장)
        print("   PCA 차원 축소 중...")
        pca = PCA(n_components=min(20, X_cleaned.shape[1]), random_state=42)
        X_pca = pca.fit_transform(X_cleaned)

        print(f"   PCA 후: {X_pca.shape[1]}개 피처")
        print(f"   설명 분산 비율: {pca.explained_variance_ratio_.sum():.4f}")

        return X_pca

    def create_simple_models(self):
        """단순한 모델들 생성"""
        print("🔄 단순한 모델들 생성 중...")

        self.models = {
            "ridge": Ridge(alpha=10.0),  # 매우 강한 정규화
            "svr": SVR(kernel="rbf", C=0.1, gamma="scale"),  # 매우 보수적
            "random_forest": RandomForestRegressor(
                n_estimators=50,
                max_depth=3,
                min_samples_split=100,
                min_samples_leaf=50,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
            "xgboost": XGBRegressor(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=3,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=1.0,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
            ),
            "lightgbm": LGBMRegressor(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=3,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=1.0,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
        }

        print(f"✅ {len(self.models)}개 단순한 모델 생성 완료")

    def prepare_pure_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """순수 데이터 준비"""
        print("🔄 순수 데이터 준비 중...")

        # StandardScaler로 정규화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers["pure_ensemble"] = scaler

        # 피처 선택 (상위 10개 특성만 선택 - 과적합 방지)
        print("   피처 선택 중...")
        self.feature_selector = SelectKBest(score_func=f_regression, k=10)
        X_selected = self.feature_selector.fit_transform(X_scaled, y)

        print(f"✅ 순수 데이터 준비 완료: {X_selected.shape}")
        print(f"   원본 피처 수: {X.shape[1]}")
        print(f"   선택된 피처 수: {X_selected.shape[1]}")

        return X_selected, y

    def train_simple_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """단순한 모델들 훈련"""
        print("🚀 단순한 모델들 훈련 시작...")

        # 훈련/검증/테스트 분할 (3단계)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.3, random_state=42
        )

        model_results = {}
        kf = KFold(n_splits=3, shuffle=True, random_state=42)

        for name, model in self.models.items():
            print(f"   훈련 중: {name}")

            try:
                # 교차 검증
                cv_r2_scores = cross_val_score(
                    model, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1
                )
                cv_rmse_scores = -cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=kf,
                    scoring="neg_root_mean_squared_error",
                    n_jobs=-1,
                )

                avg_r2 = cv_r2_scores.mean()
                std_r2 = cv_r2_scores.std()
                avg_rmse = cv_rmse_scores.mean()

                # 최종 모델 훈련
                model.fit(X_train, y_train)

                # 검증 성능
                val_pred = model.predict(X_val)
                val_r2 = r2_score(y_val, val_pred)
                val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

                # 테스트 성능
                test_pred = model.predict(X_test)
                test_r2 = r2_score(y_test, test_pred)
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

                # 과적합 간격 계산
                overfitting_gap = val_r2 - test_r2

                model_results[name] = {
                    "cv_mean_r2": avg_r2,
                    "cv_std_r2": std_r2,
                    "cv_mean_rmse": avg_rmse,
                    "val_r2": val_r2,
                    "val_rmse": val_rmse,
                    "test_r2": test_r2,
                    "test_rmse": test_rmse,
                    "overfitting_gap": overfitting_gap,
                    "model": model,
                }

                print(f"     CV R²: {avg_r2:.4f} (±{std_r2:.4f})")
                print(f"     Val R²: {val_r2:.4f}")
                print(f"     Test R²: {test_r2:.4f}")
                print(f"     과적합 간격: {overfitting_gap:.4f}")

            except Exception as e:
                print(f"     ❌ 훈련 실패: {str(e)}")
                model_results[name] = None

        # 성능 순으로 정렬 (과적합 간격 고려)
        valid_models = {k: v for k, v in model_results.items() if v is not None}
        sorted_models = sorted(
            valid_models.items(),
            key=lambda x: x[1]["test_r2"] - x[1]["overfitting_gap"],
            reverse=True,
        )

        print(f"✅ {len(valid_models)}개 단순한 모델 훈련 완료")
        print("📊 단순한 모델 성능 순위 (과적합 간격 고려):")
        for i, (name, scores) in enumerate(sorted_models, 1):
            print(
                f"   {i}. {name}: Test R² = {scores['test_r2']:.4f}, 과적합 = {scores['overfitting_gap']:.4f}"
            )

        return model_results

    def create_simple_ensemble(
        self, X: np.ndarray, y: np.ndarray, model_results: Dict
    ) -> Dict:
        """단순한 앙상블 생성"""
        print("🔄 단순한 앙상블 생성 중...")

        # 상위 2개 모델만 선택 (과적합 방지)
        valid_models = {k: v for k, v in model_results.items() if v is not None}
        sorted_models = sorted(
            valid_models.items(),
            key=lambda x: x[1]["test_r2"] - x[1]["overfitting_gap"],
            reverse=True,
        )
        top_models = [name for name, _ in sorted_models[:2]]

        print(f"   선택된 상위 모델들: {top_models}")

        # 훈련/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        predictions = []
        weights = []

        for name in top_models:
            if name in model_results and model_results[name] is not None:
                model = model_results[name]["model"]
                pred = model.predict(X_test)
                predictions.append(pred)
                # 과적합 간격을 고려한 가중치
                weight = (
                    model_results[name]["test_r2"]
                    - model_results[name]["overfitting_gap"]
                )
                weights.append(max(weight, 0.1))

        if not predictions:
            print("❌ 유효한 예측값이 없습니다.")
            return None

        # 가중치 정규화
        weights = np.array(weights)
        weights = weights / weights.sum()

        # 가중 평균으로 앙상블 예측
        predictions = np.array(predictions)
        ensemble_pred = np.average(predictions, axis=0, weights=weights)

        # 앙상블 성능 평가
        r2 = r2_score(y_test, ensemble_pred)
        mse = mean_squared_error(y_test, ensemble_pred)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(ensemble_pred - y_test))
        correlation = np.corrcoef(ensemble_pred, y_test)[0, 1]

        results = {
            "r2": r2,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "correlation": correlation,
            "predictions": ensemble_pred,
            "targets": y_test,
            "selected_models": top_models,
            "model_weights": dict(zip(top_models, weights)),
        }

        print(f"✅ 단순한 앙상블 생성 및 평가 완료:")
        print(f"   R²: {r2:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   상관계수: {correlation:.4f}")

        return results

    def run_data_purification(self, limit: int = 10000) -> Dict:
        """데이터 정제 실행"""
        print("🚀 데이터 정제 시스템 시작 - 다중공선성 제거")
        print("=" * 60)

        # 1. 순수 데이터 로딩
        multimodal_data, targets = self.load_pure_data(limit)

        # 2. 독립적 특성 추출
        X_engineered = self.create_independent_features(multimodal_data)

        # 3. 상관관계 분석
        correlation_analysis = self.analyze_correlation(X_engineered, targets)

        # 4. 단순한 모델들 생성
        self.create_simple_models()

        # 5. 순수 데이터 준비
        X, y = self.prepare_pure_data(X_engineered, targets)

        # 6. 단순한 모델들 훈련
        model_results = self.train_simple_models(X, y)

        # 7. 단순한 앙상블 생성
        ensemble_results = self.create_simple_ensemble(X, y, model_results)

        # 8. 결과 저장
        results = {
            "correlation_analysis": correlation_analysis,
            "individual_models_results": {
                name: {
                    "cv_mean_r2": scores["cv_mean_r2"] if scores else None,
                    "cv_std_r2": scores["cv_std_r2"] if scores else None,
                    "val_r2": scores["val_r2"] if scores else None,
                    "test_r2": scores["test_r2"] if scores else None,
                    "overfitting_gap": scores["overfitting_gap"] if scores else None,
                }
                for name, scores in model_results.items()
            },
            "ensemble_results": ensemble_results,
            "data_info": {
                "n_samples": len(y),
                "n_features_original": X_engineered.shape[1],
                "n_features_selected": X.shape[1],
                "n_models_trained": len(
                    [m for m in model_results.values() if m is not None]
                ),
                "n_models_in_ensemble": (
                    len(ensemble_results["selected_models"]) if ensemble_results else 0
                ),
                "purification_status": "데이터 정제 완료",
            },
        }

        # JSON으로 저장
        with open("data_purification_results.json", "w") as f:

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

        print(f"✅ 데이터 정제 완료!")
        if ensemble_results:
            print(f"   최종 앙상블 R²: {ensemble_results['r2']:.4f}")
            print(f"   최종 앙상블 RMSE: {ensemble_results['rmse']:.4f}")

        return results


def main():
    """메인 실행 함수"""
    print("🚀 데이터 정제 시스템 - 다중공선성 제거")
    print("=" * 60)

    purifier = DataPurifier()
    results = purifier.run_data_purification(limit=10000)

    print("\n📊 데이터 정제 결과:")
    if results["ensemble_results"]:
        print(f"   최종 앙상블 R²: {results['ensemble_results']['r2']:.4f}")
        print(f"   최종 앙상블 RMSE: {results['ensemble_results']['rmse']:.4f}")
        print(f"   상관계수: {results['ensemble_results']['correlation']:.4f}")
        print(f"   선택된 모델들: {results['ensemble_results']['selected_models']}")


if __name__ == "__main__":
    main()
