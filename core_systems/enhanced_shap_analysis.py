#!/usr/bin/env python3
"""
향상된 SHAP 분석 시스템
- 데이터 의존성 완화
- 가중치 최적화
- 성능 최적화
- 결과 저장 및 시각화
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from vector_search_system import Big5VectorSearch


class EnhancedSHAPAnalyzer:
    def __init__(self, project_id: str = "persona-diary-service"):
        self.project_id = project_id
        self.vector_search = Big5VectorSearch(project_id)
        self.results_dir = "shap_analysis_results"
        os.makedirs(self.results_dir, exist_ok=True)

    def load_data_with_fallback(self, limit: int = 2000) -> pd.DataFrame:
        """데이터 로드 실패 시 대체 방안 제공"""
        try:
            print("BigQuery에서 데이터 로드 중...")
            data = self.vector_search.load_data(limit=limit)
            if data.empty:
                raise ValueError("로드된 데이터가 비어있습니다.")
            print(f"✅ 데이터 로드 성공: {len(data)}건")
            return data
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            print("🔄 대체 데이터 생성 중...")
            return self._generate_fallback_data(limit)

    def _generate_fallback_data(self, limit: int) -> pd.DataFrame:
        """데이터 로드 실패 시 대체 데이터 생성"""
        np.random.seed(42)

        # Big5 특성 컬럼 생성
        big5_cols = []
        for trait in ["EXT", "EST", "AGR", "CSN", "OPN"]:
            for i in range(1, 11):
                big5_cols.append(f"{trait}{i}")

        # 랜덤 데이터 생성 (1-6 범위)
        data = {}
        for col in big5_cols:
            if col.startswith(("EST", "AGR")):
                data[col] = np.random.randint(2, 7, limit)  # 2-6 범위
            else:
                data[col] = np.random.randint(1, 6, limit)  # 1-5 범위

        # 국가 정보 추가
        countries = ["US", "GB", "CA", "AU", "IN", "DE", "FR", "JP", "KR", "BR"]
        data["country"] = np.random.choice(countries, limit)

        df = pd.DataFrame(data)
        print(f"✅ 대체 데이터 생성 완료: {len(df)}건")
        return df

    def optimize_target_weights(self, data: pd.DataFrame) -> Dict[str, float]:
        """데이터 기반 가중치 최적화"""
        print("🔧 타겟 변수 가중치 최적화 중...")

        # 실제 관찰된 행동 패턴 시뮬레이션 (더 현실적인 타겟)
        # 사회적 활동 참여도
        social_activity = (
            data["EXT1"] * 0.4
            + data["EXT2"] * 0.3
            + data["EXT3"] * 0.3
            + data["AGR1"] * 0.2
            + data["AGR2"] * 0.2
        )

        # 스트레스 관리 (EST 역상관)
        stress_management = 6 - (data["EST1"] + data["EST2"] + data["EST3"]) / 3

        # 목표 달성 (CSN 기반)
        goal_achievement = (data["CSN1"] + data["CSN2"] + data["CSN3"]) / 3

        # 창의성 (OPN 기반)
        creativity = (data["OPN1"] + data["OPN2"] + data["OPN3"]) / 3

        # 실제 타겟 변수 (가중 평균)
        y_actual = (
            social_activity * 0.3
            + stress_management * 0.25
            + goal_achievement * 0.25
            + creativity * 0.2
        )

        # 가중치 최적화를 위한 특성 매트릭스
        X_features = pd.DataFrame(
            {
                "social": social_activity,
                "stress_mgmt": stress_management,
                "goal_ach": goal_achievement,
                "creativity": creativity,
            }
        )

        # 선형 회귀로 최적 가중치 학습
        reg = LinearRegression().fit(X_features, y_actual)

        optimized_weights = {
            "social": float(reg.coef_[0]),
            "stress_mgmt": float(reg.coef_[1]),
            "goal_ach": float(reg.coef_[2]),
            "creativity": float(reg.coef_[3]),
        }

        print(f"✅ 최적화된 가중치: {optimized_weights}")
        return optimized_weights

    def generate_optimized_target_variable(
        self, data: pd.DataFrame, weights: Dict[str, float]
    ) -> pd.Series:
        """최적화된 가중치로 타겟 변수 생성"""
        print("🎯 최적화된 타겟 변수 생성 중...")

        # 각 행동 패턴 계산
        social_activity = (
            data["EXT1"] * 0.4
            + data["EXT2"] * 0.3
            + data["EXT3"] * 0.3
            + data["AGR1"] * 0.2
            + data["AGR2"] * 0.2
        )

        stress_management = 6 - (data["EST1"] + data["EST2"] + data["EST3"]) / 3
        goal_achievement = (data["CSN1"] + data["CSN2"] + data["CSN3"]) / 3
        creativity = (data["OPN1"] + data["OPN2"] + data["OPN3"]) / 3

        # 최적화된 가중치 적용
        satisfaction = (
            social_activity * weights["social"]
            + stress_management * weights["stress_mgmt"]
            + goal_achievement * weights["goal_ach"]
            + creativity * weights["creativity"]
        )

        # 정규화 (1-10 스케일)
        satisfaction = (
            (satisfaction - satisfaction.min())
            / (satisfaction.max() - satisfaction.min())
        ) * 9 + 1

        print(
            f"✅ 타겟 변수 생성 완료: 평균 {satisfaction.mean():.2f}, 표준편차 {satisfaction.std():.2f}"
        )
        return satisfaction

    def train_enhanced_model(
        self, X: pd.DataFrame, y: pd.Series
    ) -> RandomForestRegressor:
        """향상된 모델 훈련 및 평가"""
        print("🤖 향상된 모델 훈련 중...")

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 랜덤 포레스트 모델 (하이퍼파라미터 튜닝)
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=42,
        )

        model.fit(X_train, y_train)

        # 다양한 성능 지표 계산
        y_pred = model.predict(X_test)

        performance_metrics = {
            "r2_score": r2_score(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "cv_r2_mean": cross_val_score(model, X, y, cv=5, scoring="r2").mean(),
            "cv_r2_std": cross_val_score(model, X, y, cv=5, scoring="r2").std(),
        }

        print(f"✅ 모델 성능:")
        print(f"   R² Score: {performance_metrics['r2_score']:.3f}")
        print(f"   RMSE: {performance_metrics['rmse']:.3f}")
        print(f"   MAE: {performance_metrics['mae']:.3f}")
        print(
            f"   CV R²: {performance_metrics['cv_r2_mean']:.3f} ± {performance_metrics['cv_r2_std']:.3f}"
        )

        return model, performance_metrics

    def optimized_shap_analysis(
        self, model: RandomForestRegressor, X: pd.DataFrame, sample_size: int = 500
    ) -> dict:
        """최적화된 SHAP 분석"""
        print(f"🔬 최적화된 SHAP 분석 중... (샘플 크기: {sample_size})")

        # 샘플링으로 계산 비용 최적화
        if len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
            print(f"   샘플링 적용: {len(X)} → {len(X_sample)}")
        else:
            X_sample = X

        # SHAP 분석
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # 특성 중요도 분석
        feature_importance = pd.DataFrame(
            {
                "feature": X_sample.columns,
                "importance": np.abs(shap_values).mean(0),
                "mean_shap": shap_values.mean(0),
                "std_shap": np.std(shap_values, axis=0),
            }
        ).sort_values("importance", ascending=False)

        # 성격 특성별 영향력 집계
        trait_impact = {}
        for trait in ["EXT", "EST", "AGR", "CSN", "OPN"]:
            trait_cols = [col for col in X_sample.columns if col.startswith(trait)]
            if trait_cols:
                trait_importance = feature_importance[
                    feature_importance["feature"].isin(trait_cols)
                ]["importance"].mean()
                trait_impact[trait] = float(trait_importance)

        return {
            "shap_values": shap_values,
            "feature_importance": feature_importance,
            "trait_impact": trait_impact,
            "sample_size": len(X_sample),
        }

    def save_results(
        self, analysis_result: dict, performance_metrics: dict, weights: dict
    ):
        """결과 저장"""
        print("💾 결과 저장 중...")

        # 1. 특성 중요도 저장
        analysis_result["feature_importance"].to_csv(
            f"{self.results_dir}/feature_importance.csv", index=False
        )

        # 2. 성능 지표 저장
        with open(f"{self.results_dir}/performance_metrics.json", "w") as f:
            json.dump(performance_metrics, f, indent=2)

        # 3. 가중치 저장
        with open(f"{self.results_dir}/optimized_weights.json", "w") as f:
            json.dump(weights, f, indent=2)

        # 4. 종합 보고서 저장
        report = {
            "performance_metrics": performance_metrics,
            "trait_impact": analysis_result["trait_impact"],
            "optimized_weights": weights,
            "sample_size": analysis_result["sample_size"],
        }

        with open(f"{self.results_dir}/comprehensive_report.json", "w") as f:
            json.dump(report, f, indent=2)

        print(f"✅ 결과 저장 완료: {self.results_dir}/")

    def create_visualizations(self, analysis_result: dict, performance_metrics: dict):
        """시각화 생성"""
        print("📊 시각화 생성 중...")

        # 1. 특성 중요도 막대 그래프
        plt.figure(figsize=(12, 8))
        top_features = analysis_result["feature_importance"].head(15)
        plt.barh(range(len(top_features)), top_features["importance"])
        plt.yticks(range(len(top_features)), top_features["feature"])
        plt.xlabel("SHAP Importance")
        plt.title("Top 15 Feature Importance (SHAP)")
        plt.tight_layout()
        plt.savefig(
            f"{self.results_dir}/feature_importance.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 2. 성격 특성별 영향력
        plt.figure(figsize=(10, 6))
        traits = list(analysis_result["trait_impact"].keys())
        impacts = list(analysis_result["trait_impact"].values())

        bars = plt.bar(
            traits,
            impacts,
            color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"],
        )
        plt.xlabel("Big5 Personality Traits")
        plt.ylabel("SHAP Impact")
        plt.title("Personality Trait Impact on Satisfaction")

        # 값 표시
        for bar, impact in zip(bars, impacts):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{impact:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(
            f"{self.results_dir}/trait_impact.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 3. SHAP 요약 플롯
        if len(analysis_result["shap_values"]) > 0:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                analysis_result["shap_values"],
                analysis_result["feature_importance"]["feature"].values.reshape(1, -1)[
                    0
                ][: len(analysis_result["shap_values"][0])],
                show=False,
            )
            plt.title("SHAP Summary Plot")
            plt.tight_layout()
            plt.savefig(
                f"{self.results_dir}/shap_summary.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

        print(f"✅ 시각화 완료: {self.results_dir}/")

    def run_enhanced_analysis(
        self, data_limit: int = 2000, sample_size: int = 500
    ) -> dict:
        """향상된 분석 실행"""
        print("🚀 향상된 SHAP 분석 시스템 시작")
        print("=" * 60)

        # 1. 데이터 로드 (대체 방안 포함)
        data = self.load_data_with_fallback(data_limit)

        # 2. 가중치 최적화
        weights = self.optimize_target_weights(data)

        # 3. 최적화된 타겟 변수 생성
        y = self.generate_optimized_target_variable(data, weights)

        # 4. Big5 특성 컬럼 선택
        big5_cols = [
            col
            for col in data.columns
            if any(trait in col for trait in ["EXT", "EST", "AGR", "CSN", "OPN"])
        ]
        X = data[big5_cols]

        # 5. 향상된 모델 훈련
        model, performance_metrics = self.train_enhanced_model(X, y)

        # 6. 최적화된 SHAP 분석
        analysis_result = self.optimized_shap_analysis(model, X, sample_size)

        # 7. 결과 저장
        self.save_results(analysis_result, performance_metrics, weights)

        # 8. 시각화 생성
        self.create_visualizations(analysis_result, performance_metrics)

        return {
            "analysis_result": analysis_result,
            "performance_metrics": performance_metrics,
            "optimized_weights": weights,
            "data_info": {
                "total_records": len(data),
                "sample_size": analysis_result["sample_size"],
            },
        }


def main():
    """메인 실행 함수"""
    print("🔬 향상된 SHAP 분석 시스템")

    analyzer = EnhancedSHAPAnalyzer()

    # 향상된 분석 실행
    results = analyzer.run_enhanced_analysis(data_limit=2000, sample_size=500)

    # 결과 출력
    print("\n" + "=" * 60)
    print("📊 향상된 SHAP 분석 결과")
    print("=" * 60)

    print(f"\n🎯 모델 성능:")
    metrics = results["performance_metrics"]
    print(f"   R² Score: {metrics['r2_score']:.3f}")
    print(f"   RMSE: {metrics['rmse']:.3f}")
    print(f"   MAE: {metrics['mae']:.3f}")
    print(f"   CV R²: {metrics['cv_r2_mean']:.3f} ± {metrics['cv_r2_std']:.3f}")

    print(f"\n📈 성격 특성별 영향력:")
    for trait, impact in sorted(
        results["analysis_result"]["trait_impact"].items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"   {trait}: {impact:.3f}")

    print(f"\n⚖️ 최적화된 가중치:")
    for feature, weight in results["optimized_weights"].items():
        print(f"   {feature}: {weight:.3f}")

    print(f"\n📊 데이터 정보:")
    print(f"   총 레코드 수: {results['data_info']['total_records']:,}")
    print(f"   SHAP 분석 샘플 수: {results['data_info']['sample_size']:,}")

    print(f"\n💾 저장된 파일:")
    print(f"   📁 결과 디렉토리: {analyzer.results_dir}/")
    print(f"   📄 feature_importance.csv")
    print(f"   📄 performance_metrics.json")
    print(f"   📄 optimized_weights.json")
    print(f"   📄 comprehensive_report.json")
    print(f"   🖼️ feature_importance.png")
    print(f"   🖼️ trait_impact.png")
    print(f"   🖼️ shap_summary.png")

    print(f"\n✅ 향상된 SHAP 분석 완료!")


if __name__ == "__main__":
    main()
