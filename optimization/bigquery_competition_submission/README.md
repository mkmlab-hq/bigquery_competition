# BigQuery 대회 제출 자료

## 팀 정보
- **팀명**: MKM Lab AI 기술부
- **제출일**: 2025-01-12
- **접근법**: 예측에서 발견으로 - 페르소나 기반 솔루션

## 핵심 발견
우리는 기존의 예측 모델 접근법이 실패할 수밖에 없는 데이터의 구조적 한계를 인정하고, 
대신 비지도학습을 통한 '발견' 접근법으로 전환하여 7개의 뚜렷한 사용자 페르소나를 발견했습니다.

## 주요 결과
- **7개 페르소나 발견**: 모두 '스트레스받는 성취가' 계열
- **단일 거대 페르소나**: 7개 세부 그룹이 하나의 거대 페르소나로 통합
- **명확한 타겟 전략**: 복잡한 다중 페르소나 대신 단일 타겟 집중

## 파일 구조
- `final_report.json`: 최종 분석 결과 및 페르소나 프로필
- `persona_discovery_operation.py`: 페르소나 발견 메인 코드
- `simplified_insight_finalization.py`: 인사이트 최종화 코드
- `final_report_corrections.py`: 보고서 수정 및 개선 코드
- `submission_summary.json`: 제출 요약 정보

## 실행 방법
1. BigQuery 인증 설정
2. `persona_discovery_operation.py` 실행
3. `simplified_insight_finalization.py` 실행
4. `final_report_corrections.py` 실행

## 핵심 인사이트
모든 사용자가 '스트레스받는 성취가'라는 하나의 거대 페르소나에 속하며, 
이는 높은 스트레스 수준과 높은 활동성을 동시에 보이는 특별한 집단입니다.
이 발견은 맞춤형 헬스케어 서비스 설계의 핵심이 됩니다.
