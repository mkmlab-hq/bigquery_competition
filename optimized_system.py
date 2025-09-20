#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized BigQuery Competition System
- Memory optimization
- Parallel processing
- Performance monitoring
"""

import gc
import os
import sys
import time
from typing import Any, Dict, List

import psutil
from memory_optimizer import MemoryOptimizer
from parallel_processor import ParallelProcessor


class OptimizedSystem:
    """최적화된 시스템 클래스"""

    def __init__(self):
        self.memory_optimizer = MemoryOptimizer()
        self.parallel_processor = ParallelProcessor()
        self.performance_log = []

    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 확인"""
        memory_info = self.memory_optimizer.get_memory_info()
        memory_status = self.memory_optimizer.check_memory_status()
        system_info = self.parallel_processor.get_system_info()

        return {
            "memory_status": memory_status,
            "memory_used_percent": memory_info["used_percent"],
            "available_memory_gb": memory_info["available_gb"],
            "cpu_count": system_info["cpu_count"],
            "max_workers": system_info["max_workers"],
            "timestamp": time.time(),
        }

    def run_optimized_system(self) -> Dict[str, Any]:
        """최적화된 시스템 실행"""
        print("=" * 60)
        print("OPTIMIZED BIGQUERY COMPETITION SYSTEM")
        print("=" * 60)

        start_time = time.time()

        # 1. 시스템 상태 확인
        print("\n[STEP 1] System Status Check")
        status = self.get_system_status()
        print(f"Memory Status: {status['memory_status']}")
        print(f"Memory Usage: {status['memory_used_percent']:.1f}%")
        print(f"Available Memory: {status['available_memory_gb']:.2f} GB")
        print(f"CPU Count: {status['cpu_count']}")

        # 2. 메모리 최적화 (필요한 경우)
        if status["memory_status"] in ["HIGH", "CRITICAL"]:
            print("\n[STEP 2] Memory Optimization")
            cleanup_result = self.memory_optimizer.cleanup_memory()
            print(f"Memory freed: {cleanup_result['freed_mb']:.2f} MB")

        # 3. 병렬 처리 최적화
        print("\n[STEP 3] Parallel Processing Optimization")
        optimization = self.parallel_processor.optimize_workload(10000)
        print(f"Recommended Workers: {optimization['recommended_workers']}")
        print(f"Recommended Batch Size: {optimization['recommended_batch_size']}")
        print(f"Use Threads: {optimization['use_threads']}")

        # 4. 시스템 실행 시뮬레이션
        print("\n[STEP 4] System Execution Simulation")
        systems = [
            "advanced_multimodal_training",
            "real_time_multimodal_learning",
            "performance_monitoring",
            "bigquery_optimization",
            "vector_search",
        ]

        for i, system in enumerate(systems, 1):
            print(f"  {i}. {system}...")
            time.sleep(0.1)  # 시뮬레이션

            # 메모리 상태 모니터링
            current_status = self.get_system_status()
            if current_status["memory_status"] == "CRITICAL":
                print(f"    Memory cleanup triggered...")
                self.memory_optimizer.cleanup_memory()

        # 5. 성능 측정
        end_time = time.time()
        execution_time = end_time - start_time

        final_status = self.get_system_status()

        # 결과 저장
        result = {
            "execution_time": execution_time,
            "final_memory_status": final_status["memory_status"],
            "final_memory_usage": final_status["memory_used_percent"],
            "systems_completed": len(systems),
            "optimization_applied": status["memory_status"] in ["HIGH", "CRITICAL"],
            "timestamp": time.time(),
        }

        self.performance_log.append(result)

        return result

    def generate_report(self) -> str:
        """성능 보고서 생성"""
        if not self.performance_log:
            return "No performance data available"

        latest = self.performance_log[-1]

        report = f"""
OPTIMIZED SYSTEM PERFORMANCE REPORT
====================================
Execution Time: {latest['execution_time']:.2f} seconds
Memory Status: {latest['final_memory_status']}
Memory Usage: {latest['final_memory_usage']:.1f}%
Systems Completed: {latest['systems_completed']}
Optimization Applied: {latest['optimization_applied']}

PERFORMANCE RATING:
"""

        if latest["execution_time"] < 30:
            report += "EXCELLENT - Under 30 seconds"
        elif latest["execution_time"] < 45:
            report += "GOOD - Under 45 seconds"
        else:
            report += "NEEDS IMPROVEMENT - Over 45 seconds"

        return report


def main():
    """메인 함수"""
    system = OptimizedSystem()

    # 최적화된 시스템 실행
    result = system.run_optimized_system()

    # 보고서 생성
    report = system.generate_report()
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)

    # 최종 상태 확인
    final_status = system.get_system_status()
    print(f"\nFinal System Status:")
    print(f"Memory: {final_status['memory_used_percent']:.1f}%")
    print(f"Status: {final_status['memory_status']}")
    print(f"CPU: {final_status['cpu_count']} cores")

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
