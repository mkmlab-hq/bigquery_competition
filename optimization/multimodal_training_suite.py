#!/usr/bin/env python3
"""
멀티모달 훈련 통합 스위트
- 모든 멀티모달 훈련 시스템 통합
- 자동화된 훈련 파이프라인
- 성능 비교 및 벤치마킹
"""

import json
import os
import subprocess
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


class MultimodalTrainingSuite:
    """멀티모달 훈련 통합 스위트"""

    def __init__(self, project_id: str = "persona-diary-service"):
        self.project_id = project_id
        self.training_results = {}
        self.performance_comparison = {}

        print("🚀 멀티모달 훈련 통합 스위트 초기화")
        print("=" * 60)

    def check_dependencies(self) -> Dict[str, bool]:
        """의존성 확인"""
        print("🔍 의존성 확인 중...")

        dependencies = {
            "torch": False,
            "sklearn": False,
            "pandas": False,
            "numpy": False,
            "matplotlib": False,
            "seaborn": False,
            "tqdm": False,
        }

        for dep in dependencies.keys():
            try:
                __import__(dep)
                dependencies[dep] = True
                print(f"   ✅ {dep}")
            except ImportError:
                print(f"   ❌ {dep}")

        return dependencies

    def run_advanced_training(self, n_samples: int = 10000, epochs: int = 100) -> Dict:
        """고급 멀티모달 훈련 실행"""
        print("\n🧠 고급 멀티모달 훈련 실행")
        print("-" * 40)

        try:
            # advanced_multimodal_training.py 실행
            result = subprocess.run(
                [sys.executable, "f:\\workspace\\bigquery_competition\\optimization\\advanced_multimodal_training.py"],
                capture_output=True,
                text=True,
                timeout=1800,
                encoding="utf-8",  # Explicitly set encoding to UTF-8
                errors="replace"  # Replace invalid characters
            )  # 30분 타임아웃

            if result.returncode == 0:
                print("✅ 고급 훈련 완료")

                # 결과 파일 확인
                if os.path.exists("multimodal_training_results.json"):
                    with open("multimodal_training_results.json", "r") as f:
                        training_data = json.load(f)

                    self.training_results["advanced_training"] = {
                        "status": "success",
                        "r2_score": training_data["evaluation_results"]["r2"],
                        "rmse": training_data["evaluation_results"]["rmse"],
                        "mae": training_data["evaluation_results"]["mae"],
                        "modality_weights": training_data["evaluation_results"][
                            "modality_weights"
                        ],
                        "epochs_trained": training_data["training_results"][
                            "total_epochs"
                        ],
                    }
                else:
                    self.training_results["advanced_training"] = {
                        "status": "completed_but_no_results_file"
                    }
            else:
                print(f"❌ 고급 훈련 실패: {result.stderr}")
                self.training_results["advanced_training"] = {
                    "status": "failed",
                    "error": result.stderr,
                }

        except subprocess.TimeoutExpired:
            print("⏰ 고급 훈련 타임아웃 (30분)")
            self.training_results["advanced_training"] = {"status": "timeout"}
        except Exception as e:
            print(f"❌ 고급 훈련 오류: {e}")
            self.training_results["advanced_training"] = {
                "status": "error",
                "error": str(e),
            }

    def run_realtime_learning(self, n_samples: int = 1000) -> Dict:
        """실시간 학습 실행"""
        print("\n⚡ 실시간 학습 실행")
        print("-" * 40)

        try:
            # real_time_multimodal_learning.py 실행
            result = subprocess.run(
                [sys.executable, "f:\\workspace\\bigquery_competition\\optimization\\real_time_multimodal_learning.py"],
                capture_output=True,
                text=True,
                timeout=600,
                encoding="utf-8",
                errors="replace"
            )  # 10분 타임아웃

            if result.returncode == 0:
                print("✅ 실시간 학습 완료")
                self.training_results["realtime_learning"] = {
                    "status": "success",
                    "output": result.stdout,
                }
            else:
                print(f"❌ 실시간 학습 실패: {result.stderr}")
                self.training_results["realtime_learning"] = {
                    "status": "failed",
                    "error": result.stderr,
                }

        except subprocess.TimeoutExpired:
            print("⏰ 실시간 학습 타임아웃 (10분)")
            self.training_results["realtime_learning"] = {"status": "timeout"}
        except Exception as e:
            print(f"❌ 실시간 학습 오류: {e}")
            self.training_results["realtime_learning"] = {
                "status": "error",
                "error": str(e),
            }

    def run_performance_optimization(
        self, n_samples: int = 50000, epochs: int = 50
    ) -> Dict:
        """성능 최적화 실행"""
        print("\n⚡ 성능 최적화 실행")
        print("-" * 40)

        try:
            # multimodal_performance_optimizer.py 실행
            result = subprocess.run(
                [sys.executable, "f:\\workspace\\bigquery_competition\\optimization\\multimodal_performance_optimizer.py"],
                capture_output=True,
                text=True,
                timeout=1800,
                encoding="utf-8",
                errors="replace"
            )  # 30분 타임아웃

            if result.returncode == 0:
                print("✅ 성능 최적화 완료")

                # 결과 파일 확인
                if os.path.exists("multimodal_optimization_results.json"):
                    with open("multimodal_optimization_results.json", "r") as f:
                        optimization_data = json.load(f)

                    self.training_results["performance_optimization"] = {
                        "status": "success",
                        "torchscript_throughput": optimization_data[
                            "optimization_results"
                        ]["torchscript_inference"]["throughput"],
                        "speedup": optimization_data["optimization_results"][
                            "torchscript_inference"
                        ]["speedup"],
                        "optimal_batch_size": optimization_data["optimization_results"][
                            "batch_optimization"
                        ]["optimal_batch_size"],
                        "memory_usage_mb": optimization_data["optimization_results"][
                            "memory_usage_mb"
                        ],
                    }
                else:
                    self.training_results["performance_optimization"] = {
                        "status": "completed_but_no_results_file"
                    }
            else:
                print(f"❌ 성능 최적화 실패: {result.stderr}")
                self.training_results["performance_optimization"] = {
                    "status": "failed",
                    "error": result.stderr,
                }

        except subprocess.TimeoutExpired:
            print("⏰ 성능 최적화 타임아웃 (30분)")
            self.training_results["performance_optimization"] = {"status": "timeout"}
        except Exception as e:
            print(f"❌ 성능 최적화 오류: {e}")
            self.training_results["performance_optimization"] = {
                "status": "error",
                "error": str(e),
            }

    def create_comprehensive_comparison(
        self, save_path: str = "multimodal_training_comparison.png"
    ):
        """종합 비교 분석 생성"""
        print("\n📊 종합 비교 분석 생성 중...")

        # 성공한 훈련 결과만 필터링
        successful_results = {
            k: v
            for k, v in self.training_results.items()
            if v.get("status") == "success"
        }

        if not successful_results:
            print("❌ 성공한 훈련 결과가 없습니다.")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. 훈련 상태 요약
        axes[0, 0].axis("off")
        status_summary = []
        for method, result in self.training_results.items():
            status = result.get("status", "unknown")
            status_summary.append(f"{method.replace('_', ' ').title()}: {status}")

        axes[0, 0].text(
            0.1,
            0.9,
            "\n".join(status_summary),
            transform=axes[0, 0].transAxes,
            fontsize=12,
            verticalalignment="top",
            fontfamily="monospace",
        )
        axes[0, 0].set_title("Training Status Summary")

        # 2. 성능 메트릭 비교 (고급 훈련이 있는 경우)
        if "advanced_training" in successful_results:
            metrics = ["r2_score", "rmse", "mae"]
            values = [
                successful_results["advanced_training"].get(metric, 0)
                for metric in metrics
            ]

            axes[0, 1].bar(metrics, values)
            axes[0, 1].set_title("Advanced Training Performance")
            axes[0, 1].set_ylabel("Value")
            axes[0, 1].tick_params(axis="x", rotation=45)

            # 값 표시
            for i, v in enumerate(values):
                axes[0, 1].text(
                    i, v + max(values) * 0.01, f"{v:.3f}", ha="center", va="bottom"
                )

        # 3. 모달리티 가중치 (고급 훈련이 있는 경우)
        if "advanced_training" in successful_results:
            modality_weights = successful_results["advanced_training"].get(
                "modality_weights", {}
            )
            if modality_weights:
                modalities = list(modality_weights.keys())
                weights = list(modality_weights.values())

                axes[0, 2].bar(modalities, weights)
                axes[0, 2].set_title("Modality Weights")
                axes[0, 2].set_ylabel("Weight")
                axes[0, 2].tick_params(axis="x", rotation=45)

        # 4. 성능 최적화 결과 (성능 최적화가 있는 경우)
        if "performance_optimization" in successful_results:
            opt_results = successful_results["performance_optimization"]

            # 추론 속도 비교
            methods = ["Basic", "TorchScript"]
            throughputs = [
                opt_results.get("torchscript_throughput", 0)
                / opt_results.get("speedup", 1),
                opt_results.get("torchscript_throughput", 0),
            ]

            axes[1, 0].bar(methods, throughputs)
            axes[1, 0].set_title("Inference Speed Comparison")
            axes[1, 0].set_ylabel("Samples/Second")

            # 속도 향상 표시
            speedup = opt_results.get("speedup", 1)
            axes[1, 0].text(
                0.5,
                max(throughputs) * 0.8,
                f"Speedup: {speedup:.2f}x",
                ha="center",
                va="center",
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            )

        # 5. 메모리 사용량 (성능 최적화가 있는 경우)
        if "performance_optimization" in successful_results:
            memory_usage = successful_results["performance_optimization"].get(
                "memory_usage_mb", 0
            )

            axes[1, 1].bar(["Memory Usage"], [memory_usage])
            axes[1, 1].set_title("Memory Usage")
            axes[1, 1].set_ylabel("MB")
            axes[1, 1].text(
                0,
                memory_usage + memory_usage * 0.01,
                f"{memory_usage:.1f} MB",
                ha="center",
                va="bottom",
            )

        # 6. 최적 배치 크기 (성능 최적화가 있는 경우)
        if "performance_optimization" in successful_results:
            optimal_batch_size = successful_results["performance_optimization"].get(
                "optimal_batch_size", 0
            )

            axes[1, 2].bar(["Optimal Batch Size"], [optimal_batch_size])
            axes[1, 2].set_title("Optimal Batch Size")
            axes[1, 2].set_ylabel("Batch Size")
            axes[1, 2].text(
                0,
                optimal_batch_size + optimal_batch_size * 0.01,
                f"{optimal_batch_size}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✅ 종합 비교 분석 저장: {save_path}")

    def generate_training_report(
        self, save_path: str = "multimodal_training_report.md"
    ):
        """훈련 보고서 생성"""
        print(f"\n📝 훈련 보고서 생성 중...")

        report_content = f"""# 멀티모달 훈련 통합 보고서

## 📊 훈련 개요
- **프로젝트 ID**: {self.project_id}
- **생성 시간**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **총 훈련 방법**: {len(self.training_results)}개

## 🧠 훈련 결과 상세

"""

        for method, result in self.training_results.items():
            method_name = method.replace("_", " ").title()
            status = result.get("status", "unknown")

            report_content += f"### {method_name}\n"
            report_content += f"- **상태**: {status}\n"

            if status == "success":
                if "r2_score" in result:
                    report_content += f"- **R² Score**: {result['r2_score']:.4f}\n"
                if "rmse" in result:
                    report_content += f"- **RMSE**: {result['rmse']:.4f}\n"
                if "mae" in result:
                    report_content += f"- **MAE**: {result['mae']:.4f}\n"
                if "torchscript_throughput" in result:
                    report_content += f"- **TorchScript Throughput**: {result['torchscript_throughput']:.1f} samples/sec\n"
                if "speedup" in result:
                    report_content += f"- **Speedup**: {result['speedup']:.2f}x\n"
                if "optimal_batch_size" in result:
                    report_content += (
                        f"- **Optimal Batch Size**: {result['optimal_batch_size']}\n"
                    )
                if "memory_usage_mb" in result:
                    report_content += (
                        f"- **Memory Usage**: {result['memory_usage_mb']:.1f} MB\n"
                    )
            elif status == "failed":
                report_content += (
                    f"- **오류**: {result.get('error', 'Unknown error')}\n"
                )
            elif status == "timeout":
                report_content += f"- **상태**: 타임아웃으로 인한 중단\n"

            report_content += "\n"

        # 성공한 훈련 요약
        successful_count = sum(
            1
            for result in self.training_results.values()
            if result.get("status") == "success"
        )

        report_content += f"""## 📈 훈련 요약
- **성공한 훈련**: {successful_count}/{len(self.training_results)}개
- **성공률**: {successful_count/len(self.training_results)*100:.1f}%

## 🎯 권장사항

"""

        if "advanced_training" in self.training_results:
            if self.training_results["advanced_training"].get("status") == "success":
                r2_score = self.training_results["advanced_training"].get("r2_score", 0)
                if r2_score > 0.8:
                    report_content += (
                        "- ✅ 고급 훈련이 우수한 성능을 보입니다 (R² > 0.8)\n"
                    )
                elif r2_score > 0.6:
                    report_content += "- ⚠️ 고급 훈련 성능이 보통입니다 (R² > 0.6)\n"
                else:
                    report_content += (
                        "- ❌ 고급 훈련 성능이 개선이 필요합니다 (R² < 0.6)\n"
                    )

        if "performance_optimization" in self.training_results:
            if (
                self.training_results["performance_optimization"].get("status")
                == "success"
            ):
                speedup = self.training_results["performance_optimization"].get(
                    "speedup", 1
                )
                if speedup > 2.0:
                    report_content += (
                        "- ✅ 성능 최적화가 효과적입니다 (속도 향상 > 2x)\n"
                    )
                else:
                    report_content += "- ⚠️ 성능 최적화 효과가 제한적입니다\n"

        report_content += f"""
## 📁 생성된 파일
- `multimodal_training_comparison.png`: 종합 비교 분석 차트
- `multimodal_training_results.json`: 고급 훈련 결과 (있는 경우)
- `multimodal_optimization_results.json`: 성능 최적화 결과 (있는 경우)
- `realtime_performance.png`: 실시간 학습 성능 차트 (있는 경우)

---
*이 보고서는 멀티모달 훈련 통합 스위트에 의해 자동 생성되었습니다.*
"""

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"✅ 훈련 보고서 저장: {save_path}")

    def run_comprehensive_training_suite(
        self,
        advanced_samples: int = 10000,
        realtime_samples: int = 1000,
        optimization_samples: int = 50000,
    ) -> Dict:
        """종합 훈련 스위트 실행"""
        print("🚀 멀티모달 종합 훈련 스위트 시작")
        print("=" * 60)

        start_time = time.time()

        # 1. 의존성 확인
        dependencies = self.check_dependencies()
        missing_deps = [dep for dep, status in dependencies.items() if not status]
        if missing_deps:
            print(f"⚠️ 누락된 의존성: {missing_deps}")
            print("   일부 훈련이 실패할 수 있습니다.")

        # 2. 고급 멀티모달 훈련
        self.run_advanced_training(advanced_samples)

        # 3. 실시간 학습
        self.run_realtime_learning(realtime_samples)

        # 4. 성능 최적화
        self.run_performance_optimization(optimization_samples)

        # 5. 종합 비교 분석
        self.create_comprehensive_comparison()

        # 6. 훈련 보고서 생성
        self.generate_training_report()

        total_time = time.time() - start_time

        # 최종 결과
        final_results = {
            "training_results": self.training_results,
            "dependencies": dependencies,
            "total_time_minutes": total_time / 60,
            "successful_trainings": sum(
                1
                for result in self.training_results.values()
                if result.get("status") == "success"
            ),
            "total_trainings": len(self.training_results),
        }

        # JSON으로 저장
        with open("multimodal_training_suite_results.json", "w") as f:
            json.dump(final_results, f, indent=2, default=str)

        print(f"\n🎉 종합 훈련 스위트 완료!")
        print(f"   총 소요 시간: {total_time/60:.1f}분")
        print(
            f"   성공한 훈련: {final_results['successful_trainings']}/{final_results['total_trainings']}개"
        )
        print(
            f"   성공률: {final_results['successful_trainings']/final_results['total_trainings']*100:.1f}%"
        )

        return final_results


def main():
    """메인 실행 함수"""
    print("🧠 멀티모달 훈련 통합 스위트")
    print("=" * 60)

    # 훈련 스위트 초기화
    suite = MultimodalTrainingSuite()

    # 종합 훈련 실행
    results = suite.run_comprehensive_training_suite(
        advanced_samples=10000, realtime_samples=1000, optimization_samples=50000
    )

    print("\n🎯 최종 결과:")
    for method, result in results["training_results"].items():
        status = result.get("status", "unknown")
        print(f"   {method.replace('_', ' ').title()}: {status}")


if __name__ == "__main__":
    main()



