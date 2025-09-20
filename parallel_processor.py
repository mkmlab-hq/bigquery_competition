#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel Processing System
- Multi-core utilization
- Parallel data processing
- Performance optimization
"""

import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Dict, List

import psutil


class ParallelProcessor:
    """병렬 처리 클래스"""

    def __init__(self):
        self.cpu_count = psutil.cpu_count()
        self.max_workers = min(self.cpu_count, 8)  # 최대 8개 워커

    def get_system_info(self) -> Dict[str, Any]:
        """시스템 정보 가져오기"""
        memory = psutil.virtual_memory()

        return {
            "cpu_count": self.cpu_count,
            "max_workers": self.max_workers,
            "memory_total_gb": memory.total / 1024 / 1024 / 1024,
            "memory_available_gb": memory.available / 1024 / 1024 / 1024,
            "memory_used_percent": memory.percent,
        }

    def process_data_parallel(
        self, data: List[Any], process_func: Callable, use_threads: bool = False
    ) -> List[Any]:
        """데이터 병렬 처리"""
        print(f"Processing {len(data)} items in parallel...")
        print(f"Using {'threads' if use_threads else 'processes'}")
        print(f"Workers: {self.max_workers}")

        start_time = time.time()

        try:
            if use_threads:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    results = list(executor.map(process_func, data))
            else:
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    results = list(executor.map(process_func, data))

            end_time = time.time()
            execution_time = end_time - start_time

            print(f"Parallel processing completed in {execution_time:.2f} seconds")
            return results

        except Exception as e:
            print(f"Parallel processing failed: {e}")
            return []

    def batch_process(
        self, data: List[Any], process_func: Callable, batch_size: int = 100
    ) -> List[Any]:
        """배치 처리"""
        print(f"Batch processing {len(data)} items (batch size: {batch_size})")

        results = []
        total_batches = (len(data) + batch_size - 1) // batch_size

        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            batch_num = i // batch_size + 1

            print(f"Processing batch {batch_num}/{total_batches}...")

            # 배치 처리
            batch_results = [process_func(item) for item in batch]
            results.extend(batch_results)

            # 메모리 정리
            if batch_num % 5 == 0:
                import gc

                gc.collect()

        return results

    def optimize_workload(self, data_size: int) -> Dict[str, Any]:
        """워크로드 최적화"""
        memory = psutil.virtual_memory()

        # 메모리 사용률에 따른 최적화
        if memory.percent > 85:
            recommended_workers = max(2, self.cpu_count // 4)
            recommended_batch_size = 50
            use_threads = True
        elif memory.percent > 70:
            recommended_workers = max(4, self.cpu_count // 2)
            recommended_batch_size = 100
            use_threads = False
        else:
            recommended_workers = self.max_workers
            recommended_batch_size = 200
            use_threads = False

        return {
            "recommended_workers": recommended_workers,
            "recommended_batch_size": recommended_batch_size,
            "use_threads": use_threads,
            "estimated_time": data_size / (recommended_workers * 100),
            "memory_efficient": memory.percent < 70,
        }


def sample_process_function(item: Any) -> Any:
    """샘플 처리 함수"""
    # 시뮬레이션된 처리
    time.sleep(0.01)
    return f"processed_{item}"


def main():
    """메인 함수"""
    print("=" * 60)
    print("PARALLEL PROCESSING SYSTEM")
    print("=" * 60)

    processor = ParallelProcessor()

    # 시스템 정보
    info = processor.get_system_info()
    print(f"CPU Count: {info['cpu_count']}")
    print(f"Max Workers: {info['max_workers']}")
    print(f"Memory: {info['memory_used_percent']:.1f}% used")

    # 테스트 데이터 생성
    test_data = list(range(1000))
    print(f"\nTest Data: {len(test_data)} items")

    # 워크로드 최적화
    optimization = processor.optimize_workload(len(test_data))
    print(f"\nOptimization Recommendations:")
    print(f"Workers: {optimization['recommended_workers']}")
    print(f"Batch Size: {optimization['recommended_batch_size']}")
    print(f"Use Threads: {optimization['use_threads']}")
    print(f"Memory Efficient: {optimization['memory_efficient']}")

    # 병렬 처리 테스트
    print(f"\nTesting parallel processing...")
    results = processor.process_data_parallel(
        test_data[:100],  # 작은 샘플로 테스트
        sample_process_function,
        use_threads=optimization["use_threads"],
    )

    print(f"Processed {len(results)} items successfully")

    # 배치 처리 테스트
    print(f"\nTesting batch processing...")
    batch_results = processor.batch_process(
        test_data[:200],  # 작은 샘플로 테스트
        sample_process_function,
        batch_size=optimization["recommended_batch_size"],
    )

    print(f"Batch processed {len(batch_results)} items successfully")

    print("\n" + "=" * 60)
    print("PARALLEL PROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
