#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Benchmark Test
- Execution time measurement
- Memory usage monitoring
- System performance analysis
"""

import os
import sys
import time

import psutil

# UTF-8 인코딩 설정
if sys.platform.startswith("win"):
    import codecs

    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())


def measure_performance():
    """성능 측정 함수"""
    print("=" * 60)
    print("[BENCHMARK] BigQuery Competition System Performance Test")
    print("=" * 60)

    # 메모리 사용량 측정 (시작)
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    # 실행 시간 측정
    start_time = time.time()

    try:
        # 시스템 실행
        exec(open("run_enhanced_systems.py").read())

        end_time = time.time()
        execution_time = end_time - start_time

        # 메모리 사용량 측정 (종료)
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = mem_after - mem_before

        # 결과 출력
        print("\n" + "=" * 60)
        print("[RESULTS] Performance Benchmark Results")
        print("=" * 60)
        print(f"Execution Time: {execution_time:.2f} seconds")
        print(f"Memory Usage: {memory_used:.2f} MB")
        print(f"Peak Memory: {mem_after:.2f} MB")
        print(f"CPU Usage: {psutil.cpu_percent()}%")
        print("=" * 60)

        # 성능 평가
        if execution_time < 30:
            print("[RATING] EXCELLENT - Under 30 seconds")
        elif execution_time < 45:
            print("[RATING] GOOD - Under 45 seconds")
        else:
            print("[RATING] NEEDS IMPROVEMENT - Over 45 seconds")

    except Exception as e:
        print(f"[ERROR] Execution failed: {e}")
        return False

    return True


if __name__ == "__main__":
    measure_performance()
