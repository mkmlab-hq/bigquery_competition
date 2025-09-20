@echo off
echo BigQuery 대회 핵심 시스템 실행
echo ================================

echo.
echo 1. 통합 관리 시스템 실행
python bigquery_competition_manager.py

echo.
echo 2. 개별 시스템 실행 옵션:
echo    - python core_systems/vector_search_system.py
echo    - python core_systems/ai_generate_system.py
echo    - python analysis/advanced_big5_clustering.py
echo    - python analysis/advanced_big5_visualization.py

pause
