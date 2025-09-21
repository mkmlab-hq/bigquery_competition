#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory Optimization System
- Memory usage monitoring
- Data streaming implementation
- Memory cleanup automation
"""

import gc
import os
import sys
import time
from typing import Any, Dict, List

import psutil


class MemoryOptimizer:
    """메모리 최적화 클래스"""

    def __init__(self):
        self.memory_threshold = 80  # 80% 메모리 사용률 임계값
        self.cleanup_threshold = 90  # 90% 메모리 사용률 정리 임계값

    def get_memory_info(self) -> Dict[str, float]:
        """메모리 정보 가져오기"""
        memory = psutil.virtual_memory()
        process = psutil.Process(os.getpid())

        return {
            "total_gb": memory.total / 1024 / 1024 / 1024,
            "available_gb": memory.available / 1024 / 1024 / 1024,
            "used_percent": memory.percent,
            "process_mb": process.memory_info().rss / 1024 / 1024,
        }

    def check_memory_status(self) -> str:
        """메모리 상태 확인"""
        info = self.get_memory_info()

        if info["used_percent"] >= self.cleanup_threshold:
            return "CRITICAL"
        elif info["used_percent"] >= self.memory_threshold:
            return "HIGH"
        else:
            return "NORMAL"

    def cleanup_memory(self) -> Dict[str, Any]:
        """메모리 정리"""
        print("Cleaning up memory...")

        # 가비지 컬렉션 실행
        collected = gc.collect()

        # 메모리 정보 (정리 전후)
        before = self.get_memory_info()
        time.sleep(0.1)  # 잠시 대기
        after = self.get_memory_info()

        return {
            "before_mb": before["process_mb"],
            "after_mb": after["process_mb"],
            "freed_mb": before["process_mb"] - after["process_mb"],
            "collected_objects": collected,
        }

    def optimize_data_loading(self, data_size: int) -> Dict[str, Any]:
        """데이터 로딩 최적화"""
        print(f"Optimizing data loading for {data_size} items...")

        # 배치 크기 계산 (메모리 사용률에 따라 조정)
        info = self.get_memory_info()

        if info["used_percent"] > 80:
            batch_size = max(100, data_size // 10)
        elif info["used_percent"] > 60:
            batch_size = max(200, data_size // 5)
        else:
            batch_size = max(500, data_size // 2)

        return {
            "recommended_batch_size": batch_size,
            "estimated_batches": (data_size + batch_size - 1) // batch_size,
            "memory_efficient": info["used_percent"] < 70,
        }

    def monitor_memory(self, duration: int = 60) -> List[Dict[str, Any]]:
        """메모리 모니터링"""
        print(f"Monitoring memory for {duration} seconds...")

        records = []
        start_time = time.time()

        while time.time() - start_time < duration:
            info = self.get_memory_info()
            records.append(
                {
                    "timestamp": time.time(),
                    "used_percent": info["used_percent"],
                    "process_mb": info["process_mb"],
                }
            )
            time.sleep(1)

        return records


def main():
    """메인 함수"""
    print("=" * 60)
    print("MEMORY OPTIMIZATION SYSTEM")
    print("=" * 60)

    optimizer = MemoryOptimizer()

    # 현재 메모리 상태
    info = optimizer.get_memory_info()
    status = optimizer.check_memory_status()

    print(f"Total Memory: {info['total_gb']:.2f} GB")
    print(f"Available Memory: {info['available_gb']:.2f} GB")
    print(f"Memory Usage: {info['used_percent']:.1f}%")
    print(f"Process Memory: {info['process_mb']:.2f} MB")
    print(f"Status: {status}")

    # 메모리 정리 (필요한 경우)
    if status in ["HIGH", "CRITICAL"]:
        cleanup_result = optimizer.cleanup_memory()
        print(f"\nMemory Cleanup Results:")
        print(f"Freed: {cleanup_result['freed_mb']:.2f} MB")
        print(f"Collected Objects: {cleanup_result['collected_objects']}")

    # 데이터 로딩 최적화 제안
    optimization = optimizer.optimize_data_loading(10000)
    print(f"\nData Loading Optimization:")
    print(f"Recommended Batch Size: {optimization['recommended_batch_size']}")
    print(f"Estimated Batches: {optimization['estimated_batches']}")
    print(f"Memory Efficient: {optimization['memory_efficient']}")

    print("\n" + "=" * 60)
    print("MEMORY OPTIMIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()



