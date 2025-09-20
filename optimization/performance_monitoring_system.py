#!/usr/bin/env python3
"""
고급 성능 모니터링 시스템
- 실시간 메트릭 수집 및 분석
- 자동 알림 및 복구 시스템
- 대시보드 및 시각화
"""

import json
import os
import time
import warnings
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)


class PerformanceMonitor:
    """고급 성능 모니터링 시스템"""

    def __init__(
        self,
        monitoring_window=1000,
        alert_threshold=0.1,
        recovery_threshold=0.2,
        save_interval=100,
    ):
        self.monitoring_window = monitoring_window
        self.alert_threshold = alert_threshold
        self.recovery_threshold = recovery_threshold
        self.save_interval = save_interval

        # 성능 메트릭 저장소
        self.metrics_history = deque(maxlen=monitoring_window)
        self.alerts_history = deque(maxlen=1000)
        self.recovery_attempts = deque(maxlen=100)

        # 실시간 모니터링 상태
        self.is_monitoring = False
        self.monitoring_start_time = None
        self.last_save_time = time.time()

        # 성능 기준선
        self.baseline_metrics = {}
        self.performance_trends = {}

        print("🔍 성능 모니터링 시스템 초기화 완료")

    def start_monitoring(self):
        """모니터링 시작"""
        self.is_monitoring = True
        self.monitoring_start_time = time.time()
        print("🚀 실시간 성능 모니터링 시작")

    def stop_monitoring(self):
        """모니터링 중지"""
        self.is_monitoring = False
        print("⏹️ 성능 모니터링 중지")

    def record_metrics(
        self,
        model_name: str,
        metrics: Dict[str, float],
        additional_info: Optional[Dict] = None,
    ):
        """성능 메트릭 기록"""
        if not self.is_monitoring:
            return

        timestamp = datetime.now()
        record = {
            "timestamp": timestamp,
            "model_name": model_name,
            "metrics": metrics,
            "additional_info": additional_info or {},
        }

        self.metrics_history.append(record)

        # 성능 분석 및 알림
        self._analyze_performance(record)

        # 주기적 저장
        if len(self.metrics_history) % self.save_interval == 0:
            self._save_metrics()

    def _analyze_performance(self, record: Dict):
        """성능 분석 및 알림 생성"""
        if len(self.metrics_history) < 10:
            return

        # 최근 성능 추세 분석
        recent_metrics = [r["metrics"] for r in list(self.metrics_history)[-10:]]
        self._analyze_trends(recent_metrics)

        # 성능 저하 탐지
        if self._detect_performance_degradation(record):
            self._generate_alert("PERFORMANCE_DEGRADATION", record)

        # 메모리 사용량 모니터링
        if self._detect_memory_issues(record):
            self._generate_alert("MEMORY_ISSUE", record)

        # 학습률 모니터링
        if self._detect_learning_rate_issues(record):
            self._generate_alert("LEARNING_RATE_ISSUE", record)

    def _analyze_trends(self, recent_metrics: List[Dict]):
        """성능 추세 분석"""
        if len(recent_metrics) < 5:
            return

        # 주요 메트릭 추출
        mae_values = [m.get("mae", 0) for m in recent_metrics]
        mse_values = [m.get("mse", 0) for m in recent_metrics]
        loss_values = [m.get("loss", 0) for m in recent_metrics]

        # 추세 계산
        mae_trend = np.polyfit(range(len(mae_values)), mae_values, 1)[0]
        mse_trend = np.polyfit(range(len(mse_values)), mse_values, 1)[0]
        loss_trend = np.polyfit(range(len(loss_values)), loss_values, 1)[0]

        self.performance_trends = {
            "mae_trend": mae_trend,
            "mse_trend": mse_trend,
            "loss_trend": loss_trend,
            "overall_trend": (mae_trend + mse_trend + loss_trend) / 3,
        }

    def _detect_performance_degradation(self, record: Dict) -> bool:
        """성능 저하 탐지"""
        if len(self.metrics_history) < 20:
            return False

        current_mae = record["metrics"].get("mae", 0)
        historical_mae = [r["metrics"].get("mae", 0) for r in list(self.metrics_history)[-20:]]

        # 기준선 설정 (처음 10개 데이터의 평균)
        baseline = np.mean(historical_mae[:10])
        current_avg = np.mean(historical_mae[-5:])

        # 성능 저하 임계값 체크
        degradation_ratio = current_avg / baseline if baseline > 0 else 1.0
        return degradation_ratio > (1 + self.alert_threshold)

    def _detect_memory_issues(self, record: Dict) -> bool:
        """메모리 사용량 문제 탐지"""
        memory_usage = record.get("additional_info", {}).get("memory_usage", 0)
        return memory_usage > 0.9  # 90% 이상 사용 시 알림

    def _detect_learning_rate_issues(self, record: Dict) -> bool:
        """학습률 문제 탐지"""
        learning_rate = record.get("additional_info", {}).get("learning_rate", 0)
        return learning_rate < 1e-6 or learning_rate > 1.0

    def _generate_alert(self, alert_type: str, record: Dict):
        """알림 생성"""
        alert = {
            "timestamp": datetime.now(),
            "alert_type": alert_type,
            "severity": self._get_alert_severity(alert_type),
            "record": record,
            "recommendations": self._get_recommendations(alert_type),
        }

        self.alerts_history.append(alert)
        self._log_alert(alert)

    def _get_alert_severity(self, alert_type: str) -> str:
        """알림 심각도 결정"""
        severity_map = {
            "PERFORMANCE_DEGRADATION": "HIGH",
            "MEMORY_ISSUE": "CRITICAL",
            "LEARNING_RATE_ISSUE": "MEDIUM",
        }
        return severity_map.get(alert_type, "LOW")

    def _get_recommendations(self, alert_type: str) -> List[str]:
        """알림별 권장사항"""
        recommendations = {
            "PERFORMANCE_DEGRADATION": [
                "학습률 조정을 고려하세요",
                "모델 복잡도 검토가 필요합니다",
                "데이터 품질을 확인하세요",
                "정규화 강도를 조정하세요",
            ],
            "MEMORY_ISSUE": [
                "배치 크기를 줄이세요",
                "모델 크기를 축소하세요",
                "메모리 정리를 실행하세요",
                "GPU 메모리 사용량을 모니터링하세요",
            ],
            "LEARNING_RATE_ISSUE": [
                "학습률을 적절한 범위로 조정하세요",
                "적응형 학습률 스케줄러를 사용하세요",
                "그래디언트 클리핑을 적용하세요",
            ],
        }
        return recommendations.get(alert_type, ["시스템 상태를 확인하세요"])

    def _log_alert(self, alert: Dict):
        """알림 로깅"""
        severity_emoji = {
            "CRITICAL": "🚨",
            "HIGH": "⚠️",
            "MEDIUM": "🔶",
            "LOW": "ℹ️",
        }

        emoji = severity_emoji.get(alert["severity"], "ℹ️")
        print(f"{emoji} {alert['alert_type']} - {alert['severity']}")
        print(f"   시간: {alert['timestamp']}")
        print(f"   권장사항: {', '.join(alert['recommendations'])}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 정보 반환"""
        if not self.metrics_history:
            return {"status": "no_data"}

        recent_metrics = [r["metrics"] for r in list(self.metrics_history)[-10:]]
        all_metrics = [r["metrics"] for r in self.metrics_history]

        # 기본 통계
        mae_values = [m.get("mae", 0) for m in all_metrics]
        mse_values = [m.get("mse", 0) for m in all_metrics]
        loss_values = [m.get("loss", 0) for m in all_metrics]

        summary = {
            "monitoring_duration": time.time() - self.monitoring_start_time if self.monitoring_start_time else 0,
            "total_records": len(self.metrics_history),
            "recent_performance": {
                "mae": {
                    "current": mae_values[-1] if mae_values else 0,
                    "average": np.mean(mae_values[-10:]) if len(mae_values) >= 10 else np.mean(mae_values),
                    "trend": self.performance_trends.get("mae_trend", 0),
                },
                "mse": {
                    "current": mse_values[-1] if mse_values else 0,
                    "average": np.mean(mse_values[-10:]) if len(mse_values) >= 10 else np.mean(mse_values),
                    "trend": self.performance_trends.get("mse_trend", 0),
                },
                "loss": {
                    "current": loss_values[-1] if loss_values else 0,
                    "average": np.mean(loss_values[-10:]) if len(loss_values) >= 10 else np.mean(loss_values),
                    "trend": self.performance_trends.get("loss_trend", 0),
                },
            },
            "alerts": {
                "total": len(self.alerts_history),
                "recent": len([a for a in self.alerts_history if (datetime.now() - a["timestamp"]).seconds < 3600]),
                "critical": len([a for a in self.alerts_history if a["severity"] == "CRITICAL"]),
            },
            "overall_trend": self.performance_trends.get("overall_trend", 0),
        }

        return summary

    def create_performance_dashboard(self, save_path: str = "performance_dashboard.png"):
        """성능 대시보드 생성"""
        if not self.metrics_history:
            print("❌ 표시할 데이터가 없습니다")
            return

        # 데이터 준비
        timestamps = [r["timestamp"] for r in self.metrics_history]
        mae_values = [r["metrics"].get("mae", 0) for r in self.metrics_history]
        mse_values = [r["metrics"].get("mse", 0) for r in self.metrics_history]
        loss_values = [r["metrics"].get("loss", 0) for r in self.metrics_history]

        # 대시보드 생성
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("실시간 성능 모니터링 대시보드", fontsize=16, fontweight="bold")

        # 1. MAE 추이
        axes[0, 0].plot(timestamps, mae_values, "b-", linewidth=2, label="MAE")
        axes[0, 0].set_title("Mean Absolute Error (MAE)")
        axes[0, 0].set_ylabel("MAE")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis="x", rotation=45)

        # 2. MSE 추이
        axes[0, 1].plot(timestamps, mse_values, "r-", linewidth=2, label="MSE")
        axes[0, 1].set_title("Mean Squared Error (MSE)")
        axes[0, 1].set_ylabel("MSE")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis="x", rotation=45)

        # 3. Loss 추이
        axes[1, 0].plot(timestamps, loss_values, "g-", linewidth=2, label="Loss")
        axes[1, 0].set_title("Training Loss")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis="x", rotation=45)

        # 4. 성능 요약
        summary = self.get_performance_summary()
        axes[1, 1].axis("off")
        
        summary_text = f"""
성능 요약:
• 모니터링 시간: {summary['monitoring_duration']:.1f}초
• 총 기록 수: {summary['total_records']}
• 현재 MAE: {summary['recent_performance']['mae']['current']:.4f}
• 현재 MSE: {summary['recent_performance']['mse']['current']:.4f}
• 현재 Loss: {summary['recent_performance']['loss']['current']:.4f}
• 전체 추세: {summary['overall_trend']:.4f}
• 알림 수: {summary['alerts']['total']}
• 최근 알림: {summary['alerts']['recent']}
• 심각한 알림: {summary['alerts']['critical']}
        """
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"📊 성능 대시보드 저장: {save_path}")

    def _save_metrics(self):
        """메트릭 저장"""
        if not self.metrics_history:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_metrics_{timestamp}.json"

        # JSON 직렬화 가능한 형태로 변환
        save_data = {
            "monitoring_info": {
                "start_time": self.monitoring_start_time,
                "total_records": len(self.metrics_history),
                "monitoring_window": self.monitoring_window,
            },
            "metrics_history": [
                {
                    "timestamp": r["timestamp"].isoformat(),
                    "model_name": r["model_name"],
                    "metrics": r["metrics"],
                    "additional_info": r["additional_info"],
                }
                for r in self.metrics_history
            ],
            "alerts_history": [
                {
                    "timestamp": a["timestamp"].isoformat(),
                    "alert_type": a["alert_type"],
                    "severity": a["severity"],
                    "recommendations": a["recommendations"],
                }
                for a in self.alerts_history
            ],
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"💾 메트릭 저장: {filename}")

    def export_performance_report(self, save_path: str = "performance_report.html"):
        """성능 보고서 HTML 내보내기"""
        if not self.metrics_history:
            print("❌ 내보낼 데이터가 없습니다")
            return

        summary = self.get_performance_summary()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>성능 모니터링 보고서</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metric {{ margin: 10px 0; padding: 10px; border-left: 4px solid #007acc; }}
                .alert {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ff6b6b; }}
                .summary {{ background-color: #e8f4fd; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 성능 모니터링 보고서</h1>
                <p>생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="summary">
                <h2>📊 성능 요약</h2>
                <p><strong>모니터링 시간:</strong> {summary['monitoring_duration']:.1f}초</p>
                <p><strong>총 기록 수:</strong> {summary['total_records']}</p>
                <p><strong>현재 MAE:</strong> {summary['recent_performance']['mae']['current']:.4f}</p>
                <p><strong>현재 MSE:</strong> {summary['recent_performance']['mse']['current']:.4f}</p>
                <p><strong>전체 추세:</strong> {summary['overall_trend']:.4f}</p>
            </div>
            
            <div class="metric">
                <h2>⚠️ 알림 현황</h2>
                <p><strong>총 알림 수:</strong> {summary['alerts']['total']}</p>
                <p><strong>최근 알림:</strong> {summary['alerts']['recent']}</p>
                <p><strong>심각한 알림:</strong> {summary['alerts']['critical']}</p>
            </div>
        </body>
        </html>
        """

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"📄 성능 보고서 저장: {save_path}")


class ModelPerformanceTracker:
    """모델별 성능 추적 시스템"""

    def __init__(self):
        self.model_metrics = {}
        self.comparison_results = {}

    def track_model(self, model_name: str, metrics: Dict[str, float]):
        """모델 성능 추적"""
        if model_name not in self.model_metrics:
            self.model_metrics[model_name] = []

        self.model_metrics[model_name].append({
            "timestamp": datetime.now(),
            "metrics": metrics,
        })

    def compare_models(self, model_names: List[str]) -> Dict[str, Any]:
        """모델 성능 비교"""
        if len(model_names) < 2:
            return {"error": "최소 2개 모델이 필요합니다"}

        comparison = {}
        for model_name in model_names:
            if model_name not in self.model_metrics:
                continue

            model_data = self.model_metrics[model_name]
            if not model_data:
                continue

            # 최근 성능 계산
            recent_metrics = model_data[-10:] if len(model_data) >= 10 else model_data
            mae_values = [m["metrics"].get("mae", 0) for m in recent_metrics]
            mse_values = [m["metrics"].get("mse", 0) for m in recent_metrics]

            comparison[model_name] = {
                "avg_mae": np.mean(mae_values),
                "avg_mse": np.mean(mse_values),
                "std_mae": np.std(mae_values),
                "std_mse": np.std(mse_values),
                "total_updates": len(model_data),
            }

        # 최고 성능 모델 찾기
        if comparison:
            best_mae_model = min(comparison.keys(), key=lambda x: comparison[x]["avg_mae"])
            best_mse_model = min(comparison.keys(), key=lambda x: comparison[x]["avg_mse"])

            comparison["best_models"] = {
                "mae": best_mae_model,
                "mse": best_mse_model,
            }

        self.comparison_results = comparison
        return comparison

    def get_model_ranking(self) -> List[Tuple[str, float]]:
        """모델 순위 반환 (MAE 기준)"""
        if not self.comparison_results:
            return []

        rankings = []
        for model_name, metrics in self.comparison_results.items():
            if model_name == "best_models":
                continue
            rankings.append((model_name, metrics["avg_mae"]))

        return sorted(rankings, key=lambda x: x[1])


def main():
    """메인 실행 함수"""
    print("🔍 고급 성능 모니터링 시스템")
    print("=" * 50)

    # 모니터링 시스템 초기화
    monitor = PerformanceMonitor()
    tracker = ModelPerformanceTracker()

    # 모니터링 시작
    monitor.start_monitoring()

    # 시뮬레이션 데이터로 테스트
    print("📊 시뮬레이션 데이터로 테스트 중...")
    
    for i in range(50):
        # 가상의 성능 메트릭 생성
        mae = 0.5 + 0.1 * np.sin(i * 0.1) + np.random.normal(0, 0.05)
        mse = mae ** 2 + np.random.normal(0, 0.01)
        loss = mse + np.random.normal(0, 0.02)

        metrics = {
            "mae": max(0, mae),
            "mse": max(0, mse),
            "loss": max(0, loss),
        }

        additional_info = {
            "learning_rate": 0.001 * (0.95 ** (i // 10)),
            "memory_usage": 0.3 + 0.1 * np.random.random(),
            "batch_size": 32,
        }

        # 메트릭 기록
        monitor.record_metrics("test_model", metrics, additional_info)
        tracker.track_model("test_model", metrics)

        time.sleep(0.1)  # 시뮬레이션 지연

    # 성능 요약 출력
    summary = monitor.get_performance_summary()
    print("\n📈 성능 요약:")
    print(f"   모니터링 시간: {summary['monitoring_duration']:.1f}초")
    print(f"   총 기록 수: {summary['total_records']}")
    print(f"   현재 MAE: {summary['recent_performance']['mae']['current']:.4f}")
    print(f"   전체 추세: {summary['overall_trend']:.4f}")

    # 대시보드 생성
    monitor.create_performance_dashboard()
    monitor.export_performance_report()

    # 모니터링 중지
    monitor.stop_monitoring()

    print("\n✅ 성능 모니터링 시스템 테스트 완료!")


if __name__ == "__main__":
    main()
