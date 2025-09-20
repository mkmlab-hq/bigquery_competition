-- Big5 데이터 전체 구조 분석
SELECT COUNT(*) as total_records,
  COUNT(DISTINCT country) as unique_countries,
  MIN(EXT1) as min_ext_score,
  MAX(EXT1) as max_ext_score,
  AVG(EXT1) as avg_ext_score
FROM `persona-diary-service.big5_dataset.big5_preprocessed`