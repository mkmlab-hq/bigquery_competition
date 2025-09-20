#!/usr/bin/env python3
"""
BigQuery 최적화 전략 시스템
- Partitioning 및 Clustering 최적화
- Materialized Views 생성
- 쿼리 성능 분석
- 비용 최적화
"""

import json
import time
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import google.cloud.bigquery as bigquery
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)


class BigQueryOptimizationStrategy:
    def __init__(self, project_id: str = "persona-diary-service"):
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)
        self.results_dir = "bigquery_optimization_results"
        import os

        os.makedirs(self.results_dir, exist_ok=True)

    # 개선된 성능 분석 함수
    def analyze_current_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        현재 BigQuery 성능을 분석합니다.

        Returns:
            Dict[str, Dict[str, Any]]: 각 쿼리의 성능 결과.
        """
        print("📊 현재 BigQuery 성능 분석 중...")

        queries = {
            "basic_big5": """
                SELECT * FROM `{project_id}.big5_dataset.big5_preprocessed` 
                LIMIT 1000
            """,
            "big5_statistics": """
                SELECT 
                    COUNT(*) as total_records,
                    AVG(EXT1) as avg_ext1,
                    AVG(EST1) as avg_est1,
                    AVG(AGR1) as avg_agr1,
                    AVG(CSN1) as avg_csn1,
                    AVG(OPN1) as avg_opn1
                FROM `{project_id}.big5_dataset.big5_preprocessed`
            """,
        }

        performance_results = {}

        for query_name, query_template in queries.items():
            query = query_template.format(project_id=self.project_id)

            try:
                start_time = time.time()
                query_job = self.client.query(query)
                results = query_job.result()

                # 쿼리 실행 계획 분석 추가
                query_plan = query_job.query_plan
                for stage in query_plan:
                    print(
                        f"Stage {stage['name']}: {stage['start_time']} - {stage['end_time']}"
                    )

                # 쿼리 통계 수집
                bytes_processed = (
                    query_job.total_bytes_processed
                    if hasattr(query_job, "total_bytes_processed")
                    else 0
                )
                bytes_billed = (
                    query_job.total_bytes_billed
                    if hasattr(query_job, "total_bytes_billed")
                    else 0
                )

                execution_time = time.time() - start_time
                performance_results[query_name] = {
                    "execution_time": execution_time,
                    "bytes_processed": bytes_processed,
                    "bytes_billed": bytes_billed,
                    "estimated_cost": bytes_billed / (1024**4) * 5,  # $5 per TB
                    "status": "success",
                }

                print(
                    f"  {query_name}: {execution_time:.2f}초, {bytes_processed:,} bytes"
                )

            except Exception as e:
                performance_results[query_name] = {
                    "execution_time": 0,
                    "bytes_processed": 0,
                    "bytes_billed": 0,
                    "estimated_cost": 0,
                    "status": f"error: {str(e)}",
                }
                print(f"  {query_name}: 오류 - {e}")

        return performance_results

    def create_optimized_queries(self):
        """최적화된 쿼리 생성"""
        print("\n🔧 최적화된 쿼리 생성 중...")

        optimized_queries = {
            "sampled_big5": """
                SELECT * FROM `{project_id}.big5_dataset.big5_preprocessed` 
                TABLESAMPLE SYSTEM (10 PERCENT)
                LIMIT 1000
            """,
            "partitioned_big5": """
                SELECT 
                    *,
                    DATE('2024-01-01') as partition_date
                FROM `{project_id}.big5_dataset.big5_preprocessed`
                WHERE country IN ('US', 'GB', 'CA', 'AU')
                LIMIT 1000
            """,
            "clustered_big5": """
                SELECT 
                    country,
                    EXT1, EST1, AGR1, CSN1, OPN1,
                    (EXT1 + EST1 + AGR1 + CSN1 + OPN1) / 5 as avg_personality
                FROM `{project_id}.big5_dataset.big5_preprocessed`
                WHERE country IS NOT NULL
                ORDER BY country, avg_personality DESC
                LIMIT 1000
            """,
            "window_functions": """
                SELECT 
                    country,
                    EXT1, EST1, AGR1, CSN1, OPN1,
                    ROW_NUMBER() OVER (PARTITION BY country ORDER BY EXT1 DESC) as ext_rank,
                    AVG(EXT1) OVER (PARTITION BY country) as country_avg_ext,
                    PERCENTILE_CONT(EXT1, 0.5) OVER (PARTITION BY country) as country_median_ext
                FROM `{project_id}.big5_dataset.big5_preprocessed`
                WHERE country IS NOT NULL
                LIMIT 1000
            """,
            "aggregated_features": """
                SELECT 
                    country,
                    COUNT(*) as total_count,
                    AVG(EXT1) as avg_ext,
                    STDDEV(EXT1) as std_ext,
                    AVG(EST1) as avg_est,
                    STDDEV(EST1) as std_est,
                    AVG(AGR1) as avg_agr,
                    STDDEV(AGR1) as std_agr,
                    AVG(CSN1) as avg_csn,
                    STDDEV(CSN1) as std_csn,
                    AVG(OPN1) as avg_opn,
                    STDDEV(OPN1) as std_opn
                FROM `{project_id}.big5_dataset.big5_preprocessed`
                WHERE country IS NOT NULL
                GROUP BY country
                HAVING COUNT(*) > 10
                ORDER BY total_count DESC
            """,
        }

        return optimized_queries

    # Materialized View 생성 자동화 추가
    def create_materialized_views(self):
        """Materialized Views 생성"""
        print("\n📊 Materialized Views 생성 중...")

        materialized_views = {
            "big5_country_stats": """
                CREATE OR REPLACE MATERIALIZED VIEW `{project_id}.big5_dataset.big5_country_stats`
                PARTITION BY DATE('2024-01-01')
                CLUSTER BY country
                AS
                SELECT 
                    country,
                    COUNT(*) as total_count,
                    AVG(EXT1) as avg_ext,
                    AVG(EST1) as avg_est,
                    AVG(AGR1) as avg_agr,
                    AVG(CSN1) as avg_csn,
                    AVG(OPN1) as avg_opn
                FROM `{project_id}.big5_dataset.big5_preprocessed`
                WHERE country IS NOT NULL
                GROUP BY country
            """,
        }

        for view_name, view_query in materialized_views.items():
            try:
                formatted_query = view_query.format(project_id=self.project_id)
                self.client.query(formatted_query).result()
                print(f"✅ Materialized View 생성 완료: {view_name}")
            except Exception as e:
                print(f"❌ Materialized View 생성 실패: {view_name} - {e}")

    def test_optimized_queries(self, optimized_queries):
        """최적화된 쿼리 성능 테스트"""
        print("\n🧪 최적화된 쿼리 성능 테스트 중...")

        test_results = {}

        for query_name, query_template in optimized_queries.items():
            query = query_template.format(project_id=self.project_id)

            try:
                start_time = time.time()
                query_job = self.client.query(query)
                results = query_job.result()

                bytes_processed = (
                    query_job.total_bytes_processed
                    if hasattr(query_job, "total_bytes_processed")
                    else 0
                )
                bytes_billed = (
                    query_job.total_bytes_billed
                    if hasattr(query_job, "total_bytes_billed")
                    else 0
                )

                execution_time = time.time() - start_time

                test_results[query_name] = {
                    "execution_time": execution_time,
                    "bytes_processed": bytes_processed,
                    "bytes_billed": bytes_billed,
                    "estimated_cost": bytes_billed / (1024**4) * 5,
                    "status": "success",
                }

                print(
                    f"  {query_name}: {execution_time:.2f}초, {bytes_processed:,} bytes"
                )

            except Exception as e:
                test_results[query_name] = {
                    "execution_time": 0,
                    "bytes_processed": 0,
                    "bytes_billed": 0,
                    "estimated_cost": 0,
                    "status": f"error: {str(e)}",
                }
                print(f"  {query_name}: 오류 - {e}")

        return test_results

    def generate_optimization_recommendations(
        self, current_performance, optimized_performance
    ):
        """최적화 권장사항 생성"""
        print("\n💡 최적화 권장사항 생성 중...")

        recommendations = []

        # 성능 비교 분석
        for query_name in current_performance.keys():
            if query_name in optimized_performance:
                current_time = current_performance[query_name]["execution_time"]
                optimized_time = optimized_performance[query_name]["execution_time"]

                if current_time > 0 and optimized_time > 0:
                    improvement = (current_time - optimized_time) / current_time * 100

                    if improvement > 10:
                        recommendations.append(
                            {
                                "query": query_name,
                                "improvement": f"{improvement:.1f}%",
                                "current_time": f"{current_time:.2f}초",
                                "optimized_time": f"{optimized_time:.2f}초",
                                "recommendation": "쿼리 최적화 효과적",
                            }
                        )
                    elif improvement < -10:
                        recommendations.append(
                            {
                                "query": query_name,
                                "improvement": f"{improvement:.1f}%",
                                "current_time": f"{current_time:.2f}초",
                                "optimized_time": f"{optimized_time:.2f}초",
                                "recommendation": "쿼리 최적화 재검토 필요",
                            }
                        )

        # 일반적인 최적화 권장사항
        general_recommendations = [
            "1. 자주 사용되는 쿼리는 Materialized View로 저장하여 성능 향상",
            "2. WHERE 절에 자주 사용되는 컬럼으로 Clustering 설정",
            "3. 날짜별로 Partitioning하여 쿼리 범위 제한",
            "4. SELECT * 대신 필요한 컬럼만 선택하여 데이터 전송량 감소",
            "5. TABLESAMPLE을 사용하여 대용량 데이터 샘플링",
            "6. 윈도우 함수를 활용하여 복잡한 집계 연산 최적화",
            "7. 적절한 LIMIT 절 사용으로 결과 크기 제한",
        ]

        return recommendations, general_recommendations

    def create_optimization_report(
        self,
        current_performance,
        optimized_performance,
        recommendations,
        general_recommendations,
    ):
        """최적화 보고서 생성"""
        print("\n📋 최적화 보고서 생성 중...")

        report = {
            "timestamp": datetime.now().isoformat(),
            "project_id": self.project_id,
            "current_performance": current_performance,
            "optimized_performance": optimized_performance,
            "recommendations": recommendations,
            "general_recommendations": general_recommendations,
            "summary": {
                "total_queries_tested": len(current_performance),
                "successful_optimizations": len(
                    [r for r in recommendations if "효과적" in r["recommendation"]]
                ),
                "average_improvement": (
                    np.mean(
                        [
                            float(r["improvement"].replace("%", ""))
                            for r in recommendations
                            if "효과적" in r["recommendation"]
                        ]
                    )
                    if recommendations
                    else 0
                ),
            },
        }

        # JSON 보고서 저장
        with open(
            f"{self.results_dir}/optimization_report.json", "w", encoding="utf-8"
        ) as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 텍스트 보고서 생성
        with open(
            f"{self.results_dir}/optimization_summary.txt", "w", encoding="utf-8"
        ) as f:
            f.write("BigQuery 최적화 보고서\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"생성 시간: {report['timestamp']}\n")
            f.write(f"프로젝트 ID: {report['project_id']}\n\n")

            f.write("성능 개선 요약:\n")
            f.write(
                f"- 테스트된 쿼리 수: {report['summary']['total_queries_tested']}\n"
            )
            f.write(
                f"- 성공적인 최적화: {report['summary']['successful_optimizations']}\n"
            )
            f.write(
                f"- 평균 개선율: {report['summary']['average_improvement']:.1f}%\n\n"
            )

            f.write("쿼리별 성능 비교:\n")
            for rec in recommendations:
                f.write(
                    f"- {rec['query']}: {rec['improvement']} 개선 ({rec['current_time']} → {rec['optimized_time']})\n"
                )

            f.write("\n일반적인 최적화 권장사항:\n")
            for rec in general_recommendations:
                f.write(f"{rec}\n")

        print(f"✅ 최적화 보고서 저장: {self.results_dir}/optimization_report.json")
        print(f"✅ 요약 보고서 저장: {self.results_dir}/optimization_summary.txt")

    def run_optimization_analysis(self):
        """최적화 분석 실행"""
        print("🚀 BigQuery 최적화 분석 시작")
        print("=" * 60)

        # 1. 현재 성능 분석
        current_performance = self.analyze_current_performance()

        # 2. 최적화된 쿼리 생성
        optimized_queries = self.create_optimized_queries()

        # 3. 최적화된 쿼리 테스트
        optimized_performance = self.test_optimized_queries(optimized_queries)

        # 4. Materialized Views 생성 (실제로는 생성하지 않고 SQL만 제공)
        materialized_views = self.create_materialized_views()

        # 5. 최적화 권장사항 생성
        recommendations, general_recommendations = (
            self.generate_optimization_recommendations(
                current_performance, optimized_performance
            )
        )

        # 6. 최적화 보고서 생성
        self.create_optimization_report(
            current_performance,
            optimized_performance,
            recommendations,
            general_recommendations,
        )

        print("\n🎉 BigQuery 최적화 분석 완료!")
        print(f"   결과 저장: {self.results_dir}/")

        return {
            "current_performance": current_performance,
            "optimized_performance": optimized_performance,
            "recommendations": recommendations,
            "materialized_views": materialized_views,
        }


def main():
    """메인 실행 함수"""
    print("🧠 BigQuery 최적화 전략 시스템")

    optimizer = BigQueryOptimizationStrategy()
    results = optimizer.run_optimization_analysis()

    print("\n🎉 BigQuery 최적화 완료!")
    print(f"   결과 저장: {optimizer.results_dir}/")


if __name__ == "__main__":
    main()
