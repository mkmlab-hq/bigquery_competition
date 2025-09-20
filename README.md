# BigQuery 대회 핵심 시스템

## 📁 디렉토리 구조

### 🔧 Core Systems (핵심 시스템)
- `vector_search_system.py` - Big5 벡터 검색 시스템
- `ai_generate_system.py` - AI 생성 및 추천 시스템
- `multimodal_integrated_system.py` - 멀티모달 통합 시스템
- `enhanced_shap_analysis.py` - SHAP 기반 모델 해석성 분석

### 📊 Analysis (분석 시스템)
- `advanced_big5_clustering.py` - 고급 클러스터링 분석
- `advanced_correlation_analysis.py` - 상관관계 분석
- `advanced_big5_visualization.py` - 고급 시각화 시스템
- `pure_big5_analysis_system.py` - 순수 Big5 분석 시스템

### ⚡ Optimization (최적화)
- `bigquery_optimization_strategy.py` - BigQuery 최적화 전략
- `data_quality_improvement.py` - 데이터 품질 개선
- `realistic_shap_analysis.py` - 현실적 SHAP 분석

### 🧪 Testing (테스트)
- `integration_test.py` - 통합 테스트
- `bigquery_competition_manager.py` - 통합 관리 시스템

### 📊 Data (데이터)
- `test_query.sql` - BigQuery 테스트 쿼리
- `analyze_big5_data.sql` - Big5 데이터 분석 쿼리

## 🚀 빠른 시작

### 1. 통합 관리 시스템 실행
```bash
python bigquery_competition_manager.py
```

### 2. 개별 시스템 실행
```bash
# Vector Search
python core_systems/vector_search_system.py

# AI Generate
python core_systems/ai_generate_system.py

# 클러스터링 분석
python analysis/advanced_big5_clustering.py

# 시각화
python analysis/advanced_big5_visualization.py
```

## 📋 시스템 요구사항

- Python 3.8+
- Google Cloud BigQuery
- 필요한 라이브러리:
  - pandas, numpy, scikit-learn
  - matplotlib, seaborn, plotly
  - shap, google-cloud-bigquery

## 🎯 주요 기능

1. **Vector Search**: Big5 특성 기반 유사 사용자 검색
2. **AI Generate**: 개인화된 추천 및 조언 생성
3. **Clustering**: 6개 클러스터 기반 성격 분류
4. **Correlation**: Pearson, Spearman, Mutual Information 분석
5. **Visualization**: Radar Chart, Heatmap, SHAP 시각화
6. **BigQuery Optimization**: 쿼리 성능 최적화

## 📊 결과 파일

모든 분석 결과는 `results/` 디렉토리에 저장됩니다:
- 클러스터링 결과: `advanced_clustering_results/`
- 상관관계 분석: `correlation_analysis_results/`
- 시각화 결과: `advanced_visualization_results/`
- BigQuery 최적화: `bigquery_optimization_results/`

## 🔧 설정

프로젝트 ID를 변경하려면 각 파일의 `project_id` 변수를 수정하세요:
```python
project_id = "your-project-id"
```

생성일: 2025-09-21 00:39:32
