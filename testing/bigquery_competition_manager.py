#!/usr/bin/env python3
"""
BigQuery 대회 핵심 파일 통합 관리 시스템
- 모든 핵심 시스템을 한 곳에서 관리
- 통합 실행 및 모니터링
- 결과 통합 및 보고서 생성
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class BigQueryCompetitionManager:
    def __init__(self, project_id: str = "persona-diary-service"):
        self.project_id = project_id
        self.results_dir = "bigquery_competition_results"
        self.core_systems = {}
        self.results = {}
        
        # 결과 디렉토리 생성
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 핵심 시스템 파일들 정의
        self.core_files = {
            "vector_search": "vector_search_system.py",
            "ai_generate": "ai_generate_system.py", 
            "multimodal": "multimodal_integrated_system.py",
            "data_quality": "data_quality_improvement.py",
            "shap_analysis": "enhanced_shap_analysis.py",
            "clustering": "advanced_big5_clustering.py",
            "correlation": "advanced_correlation_analysis.py",
            "visualization": "advanced_big5_visualization.py",
            "bigquery_opt": "bigquery_optimization_strategy.py",
            "integration_test": "integration_test.py"
        }
        
        print("🚀 BigQuery 대회 통합 관리 시스템 초기화")
        print("=" * 60)

    def check_core_files(self):
        """핵심 파일 존재 여부 확인"""
        print("📁 핵심 파일 존재 여부 확인 중...")
        
        missing_files = []
        existing_files = []
        
        for system_name, filename in self.core_files.items():
            if os.path.exists(filename):
                existing_files.append(filename)
                print(f"  ✅ {filename}")
            else:
                missing_files.append(filename)
                print(f"  ❌ {filename} - 누락")
        
        print(f"\n📊 파일 상태: {len(existing_files)}개 존재, {len(missing_files)}개 누락")
        
        if missing_files:
            print("\n⚠️ 누락된 파일들:")
            for file in missing_files:
                print(f"  - {file}")
        
        return existing_files, missing_files

    def run_vector_search_system(self):
        """Vector Search 시스템 실행"""
        print("\n🔍 Vector Search 시스템 실행 중...")
        try:
            from vector_search_system import Big5VectorSearch
            
            vs = Big5VectorSearch(project_id=self.project_id)
            data = vs.load_data(limit=1000)
            
            # 샘플 타겟 사용자
            target_user = {
                "EXT": 4.5, "EST": 2.0, "AGR": 4.0, 
                "CSN": 3.5, "OPN": 4.8
            }
            
            # 유사 사용자 검색
            similar_users = vs.search_similar_users(target_user, data, top_k=10)
            
            self.results["vector_search"] = {
                "status": "success",
                "data_size": len(data),
                "similar_users_count": len(similar_users),
                "top_similarity": similar_users[0]["similarity_score"] if similar_users else 0
            }
            
            print(f"  ✅ Vector Search 완료: {len(similar_users)}명 유사 사용자 발견")
            
        except Exception as e:
            self.results["vector_search"] = {"status": "error", "error": str(e)}
            print(f"  ❌ Vector Search 실패: {e}")

    def run_ai_generate_system(self):
        """AI Generate 시스템 실행"""
        print("\n🤖 AI Generate 시스템 실행 중...")
        try:
            from ai_generate_system import AIGenerateSystem
            
            ai = AIGenerateSystem(project_id=self.project_id)
            
            # 샘플 타겟 사용자
            target_user = {
                "EXT": 4.5, "EST": 2.0, "AGR": 4.0, 
                "CSN": 3.5, "OPN": 4.8
            }
            
            # 종합 보고서 생성
            report = ai.generate_comprehensive_report(target_user)
            
            self.results["ai_generate"] = {
                "status": "success",
                "report_sections": len(report),
                "recommendations_count": len(report.get("personalized_recommendations", [])),
                "shap_insights_count": len(report.get("shap_insights", []))
            }
            
            print(f"  ✅ AI Generate 완료: {len(report)}개 섹션 생성")
            
        except Exception as e:
            self.results["ai_generate"] = {"status": "error", "error": str(e)}
            print(f"  ❌ AI Generate 실패: {e}")

    def run_clustering_analysis(self):
        """고급 클러스터링 분석 실행"""
        print("\n🎭 고급 클러스터링 분석 실행 중...")
        try:
            from advanced_big5_clustering import AdvancedBig5Clustering
            
            clustering = AdvancedBig5Clustering(project_id=self.project_id)
            results = clustering.run_advanced_clustering_analysis(limit=2000)
            
            if results:
                self.results["clustering"] = {
                    "status": "success",
                    "best_algorithm": results["best_algorithm"],
                    "clusters_count": len(results["cluster_characteristics"]),
                    "data_size": len(results["big5_df"])
                }
                print(f"  ✅ 클러스터링 완료: {results['best_algorithm']} 알고리즘")
            else:
                self.results["clustering"] = {"status": "error", "error": "클러스터링 실패"}
                print("  ❌ 클러스터링 실패")
                
        except Exception as e:
            self.results["clustering"] = {"status": "error", "error": str(e)}
            print(f"  ❌ 클러스터링 실패: {e}")

    def run_correlation_analysis(self):
        """고급 상관관계 분석 실행"""
        print("\n📊 고급 상관관계 분석 실행 중...")
        try:
            from advanced_correlation_analysis import AdvancedCorrelationAnalysis
            
            correlation = AdvancedCorrelationAnalysis(project_id=self.project_id)
            results = correlation.run_advanced_correlation_analysis(limit=2000)
            
            if results:
                self.results["correlation"] = {
                    "status": "success",
                    "pearson_correlations": len(results["pearson_correlations"]),
                    "nonlinear_relationships": len(results["nonlinear_relationships"]),
                    "high_mi_relationships": len(results["high_mi_relationships"]),
                    "countries_analyzed": len(results["country_analysis"])
                }
                print(f"  ✅ 상관관계 분석 완료: {len(results['pearson_correlations'])}개 상관관계")
            else:
                self.results["correlation"] = {"status": "error", "error": "상관관계 분석 실패"}
                print("  ❌ 상관관계 분석 실패")
                
        except Exception as e:
            self.results["correlation"] = {"status": "error", "error": str(e)}
            print(f"  ❌ 상관관계 분석 실패: {e}")

    def run_visualization_system(self):
        """고급 시각화 시스템 실행"""
        print("\n📈 고급 시각화 시스템 실행 중...")
        try:
            from advanced_big5_visualization import AdvancedBig5Visualization
            
            visualizer = AdvancedBig5Visualization(project_id=self.project_id)
            results = visualizer.create_comprehensive_visualization(limit=2000)
            
            if results:
                self.results["visualization"] = {
                    "status": "success",
                    "cluster_means_count": len(results["cluster_means"]),
                    "country_means_count": len(results["country_means"]),
                    "feature_importance_count": len(results["feature_importance"])
                }
                print(f"  ✅ 시각화 완료: {len(results['cluster_means'])}개 클러스터 시각화")
            else:
                self.results["visualization"] = {"status": "error", "error": "시각화 실패"}
                print("  ❌ 시각화 실패")
                
        except Exception as e:
            self.results["visualization"] = {"status": "error", "error": str(e)}
            print(f"  ❌ 시각화 실패: {e}")

    def run_bigquery_optimization(self):
        """BigQuery 최적화 분석 실행"""
        print("\n⚡ BigQuery 최적화 분석 실행 중...")
        try:
            from bigquery_optimization_strategy import BigQueryOptimizationStrategy
            
            optimizer = BigQueryOptimizationStrategy(project_id=self.project_id)
            results = optimizer.run_optimization_analysis()
            
            if results:
                self.results["bigquery_opt"] = {
                    "status": "success",
                    "queries_tested": len(results["current_performance"]),
                    "optimizations_successful": len(results["recommendations"]),
                    "materialized_views": len(results["materialized_views"])
                }
                print(f"  ✅ BigQuery 최적화 완료: {len(results['recommendations'])}개 최적화")
            else:
                self.results["bigquery_opt"] = {"status": "error", "error": "BigQuery 최적화 실패"}
                print("  ❌ BigQuery 최적화 실패")
                
        except Exception as e:
            self.results["bigquery_opt"] = {"status": "error", "error": str(e)}
            print(f"  ❌ BigQuery 최적화 실패: {e}")

    def run_integration_test(self):
        """통합 테스트 실행"""
        print("\n🔗 통합 테스트 실행 중...")
        try:
            from integration_test import test_integration
            
            # 통합 테스트 실행
            test_integration()
            
            self.results["integration_test"] = {
                "status": "success",
                "message": "통합 테스트 완료"
            }
            print("  ✅ 통합 테스트 완료")
            
        except Exception as e:
            self.results["integration_test"] = {"status": "error", "error": str(e)}
            print(f"  ❌ 통합 테스트 실패: {e}")

    def run_all_systems(self):
        """모든 핵심 시스템 실행"""
        print("🚀 모든 핵심 시스템 실행 시작")
        print("=" * 60)
        
        start_time = time.time()
        
        # 1. 파일 존재 여부 확인
        existing_files, missing_files = self.check_core_files()
        
        # 2. 핵심 시스템들 순차 실행
        self.run_vector_search_system()
        self.run_ai_generate_system()
        self.run_clustering_analysis()
        self.run_correlation_analysis()
        self.run_visualization_system()
        self.run_bigquery_optimization()
        self.run_integration_test()
        
        execution_time = time.time() - start_time
        
        # 3. 결과 요약
        self.create_summary_report(execution_time, existing_files, missing_files)
        
        print(f"\n🎉 모든 시스템 실행 완료! (총 {execution_time:.2f}초)")
        print(f"   결과 저장: {self.results_dir}/")

    def create_summary_report(self, execution_time: float, existing_files: List[str], missing_files: List[str]):
        """종합 보고서 생성"""
        print("\n📋 종합 보고서 생성 중...")
        
        # 성공/실패 통계
        successful_systems = [name for name, result in self.results.items() if result["status"] == "success"]
        failed_systems = [name for name, result in self.results.items() if result["status"] == "error"]
        
        summary_report = {
            "timestamp": datetime.now().isoformat(),
            "project_id": self.project_id,
            "execution_time": execution_time,
            "file_status": {
                "existing_files": existing_files,
                "missing_files": missing_files,
                "total_files": len(self.core_files)
            },
            "system_results": self.results,
            "summary": {
                "total_systems": len(self.results),
                "successful_systems": len(successful_systems),
                "failed_systems": len(failed_systems),
                "success_rate": len(successful_systems) / len(self.results) * 100 if self.results else 0
            }
        }
        
        # JSON 보고서 저장
        with open(f"{self.results_dir}/competition_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_report, f, indent=2, ensure_ascii=False)
        
        # 텍스트 요약 보고서 생성
        with open(f"{self.results_dir}/competition_summary.txt", "w", encoding="utf-8") as f:
            f.write("BigQuery 대회 핵심 시스템 종합 보고서\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"실행 시간: {summary_report['timestamp']}\n")
            f.write(f"프로젝트 ID: {summary_report['project_id']}\n")
            f.write(f"총 실행 시간: {execution_time:.2f}초\n\n")
            
            f.write("파일 상태:\n")
            f.write(f"- 존재하는 파일: {len(existing_files)}개\n")
            f.write(f"- 누락된 파일: {len(missing_files)}개\n\n")
            
            f.write("시스템 실행 결과:\n")
            f.write(f"- 총 시스템: {summary_report['summary']['total_systems']}개\n")
            f.write(f"- 성공: {summary_report['summary']['successful_systems']}개\n")
            f.write(f"- 실패: {summary_report['summary']['failed_systems']}개\n")
            f.write(f"- 성공률: {summary_report['summary']['success_rate']:.1f}%\n\n")
            
            f.write("시스템별 상세 결과:\n")
            for system_name, result in self.results.items():
                status = "✅ 성공" if result["status"] == "success" else "❌ 실패"
                f.write(f"- {system_name}: {status}\n")
                if result["status"] == "error":
                    f.write(f"  오류: {result['error']}\n")
        
        print(f"✅ 종합 보고서 저장: {self.results_dir}/competition_summary.json")
        print(f"✅ 요약 보고서 저장: {self.results_dir}/competition_summary.txt")

    def show_system_status(self):
        """시스템 상태 표시"""
        print("\n📊 현재 시스템 상태")
        print("=" * 40)
        
        for system_name, result in self.results.items():
            status = "✅ 성공" if result["status"] == "success" else "❌ 실패"
            print(f"{system_name:20} : {status}")
            
            if result["status"] == "error":
                print(f"{'':20}   오류: {result['error']}")

    def run_quick_test(self):
        """빠른 테스트 실행 (핵심 시스템만)"""
        print("⚡ 빠른 테스트 실행 (핵심 시스템만)")
        print("=" * 50)
        
        self.run_vector_search_system()
        self.run_ai_generate_system()
        self.run_integration_test()
        
        print("\n🎉 빠른 테스트 완료!")

    def create_file_structure(self):
        """파일 구조 생성"""
        print("\n📁 BigQuery 대회 파일 구조 생성 중...")
        
        structure = {
            "bigquery_competition/": {
                "core_systems/": {
                    "vector_search_system.py": "벡터 검색 시스템",
                    "ai_generate_system.py": "AI 생성 시스템", 
                    "multimodal_integrated_system.py": "멀티모달 통합 시스템",
                    "enhanced_shap_analysis.py": "SHAP 분석 시스템"
                },
                "analysis/": {
                    "advanced_big5_clustering.py": "고급 클러스터링",
                    "advanced_correlation_analysis.py": "상관관계 분석",
                    "advanced_big5_visualization.py": "시각화 시스템"
                },
                "optimization/": {
                    "bigquery_optimization_strategy.py": "BigQuery 최적화",
                    "data_quality_improvement.py": "데이터 품질 개선"
                },
                "testing/": {
                    "integration_test.py": "통합 테스트",
                    "bigquery_competition_manager.py": "통합 관리자"
                },
                "results/": {
                    "advanced_clustering_results/": "클러스터링 결과",
                    "correlation_analysis_results/": "상관관계 분석 결과",
                    "advanced_visualization_results/": "시각화 결과",
                    "bigquery_optimization_results/": "BigQuery 최적화 결과",
                    "bigquery_competition_results/": "통합 결과"
                }
            }
        }
        
        # 구조를 JSON으로 저장
        with open(f"{self.results_dir}/file_structure.json", "w", encoding="utf-8") as f:
            json.dump(structure, f, indent=2, ensure_ascii=False)
        
        print("✅ 파일 구조 생성 완료")
        return structure


def main():
    """메인 실행 함수"""
    print("🧠 BigQuery 대회 통합 관리 시스템")
    print("=" * 50)
    
    manager = BigQueryCompetitionManager()
    
    while True:
        print("\n📋 메뉴를 선택하세요:")
        print("1. 모든 시스템 실행")
        print("2. 빠른 테스트 (핵심 시스템만)")
        print("3. 시스템 상태 확인")
        print("4. 파일 구조 생성")
        print("5. 종료")
        
        choice = input("\n선택 (1-5): ").strip()
        
        if choice == "1":
            manager.run_all_systems()
        elif choice == "2":
            manager.run_quick_test()
        elif choice == "3":
            manager.show_system_status()
        elif choice == "4":
            manager.create_file_structure()
        elif choice == "5":
            print("👋 시스템을 종료합니다.")
            break
        else:
            print("❌ 잘못된 선택입니다. 1-5 중에서 선택하세요.")


if __name__ == "__main__":
    main()
