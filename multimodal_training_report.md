# 멀티모달 훈련 통합 보고서

## 📊 훈련 개요
- **프로젝트 ID**: persona-diary-service
- **생성 시간**: 2025-09-21 03:30:45
- **총 훈련 방법**: 3개

## 🧠 훈련 결과 상세

### Advanced Training
- **상태**: failed
- **오류**: 
Epoch 1/100:   0%|          | 0/110 [00:00<?, ?it/s]
Epoch 1/100:   0%|          | 0/110 [00:01<?, ?it/s]
Traceback (most recent call last):
  File "f:\workspace\bigquery_competition\optimization\advanced_multimodal_training.py", line 832, in <module>
    main()
  File "f:\workspace\bigquery_competition\optimization\advanced_multimodal_training.py", line 822, in main
    results = trainer.run_comprehensive_training(n_samples=10000, epochs=100)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "f:\workspace\bigquery_competition\optimization\advanced_multimodal_training.py", line 770, in run_comprehensive_training
    training_results = self.train_model(train_loader, val_loader, epochs)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "f:\workspace\bigquery_competition\optimization\advanced_multimodal_training.py", line 479, in train_model
    outputs, attention_info, dynamic_weights = self.model(
                                               ^^^^^^^^^^^
  File "F:\Python\Python311\Lib\site-packages\torch\nn\modules\module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Python\Python311\Lib\site-packages\torch\nn\modules\module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "f:\workspace\bigquery_competition\optimization\advanced_multimodal_training.py", line 198, in forward
    attended_features, cross_attention_weights = self.cross_attention(
                                                 ^^^^^^^^^^^^^^^^^^^^^
  File "F:\Python\Python311\Lib\site-packages\torch\nn\modules\module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Python\Python311\Lib\site-packages\torch\nn\modules\module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Python\Python311\Lib\site-packages\torch\nn\modules\activation.py", line 1380, in forward
    attn_output, attn_output_weights = F.multi_head_attention_forward(
                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Python\Python311\Lib\site-packages\torch\nn\functional.py", line 6304, in multi_head_attention_forward
    q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Python\Python311\Lib\site-packages\torch\nn\functional.py", line 5713, in _in_projection_packed
    kv_proj = linear(k, w_kv, b_kv)
              ^^^^^^^^^^^^^^^^^^^^^
RuntimeError: mat1 and mat2 shapes cannot be multiplied (192x128 and 64x128)


### Realtime Learning
- **상태**: success

### Performance Optimization
- **상태**: success
- **TorchScript Throughput**: 111710.6 samples/sec
- **Speedup**: 0.94x
- **Optimal Batch Size**: 512
- **Memory Usage**: 464.1 MB

## 📈 훈련 요약
- **성공한 훈련**: 2/3개
- **성공률**: 66.7%

## 🎯 권장사항

- ⚠️ 성능 최적화 효과가 제한적입니다

## 📁 생성된 파일
- `multimodal_training_comparison.png`: 종합 비교 분석 차트
- `multimodal_training_results.json`: 고급 훈련 결과 (있는 경우)
- `multimodal_optimization_results.json`: 성능 최적화 결과 (있는 경우)
- `realtime_performance.png`: 실시간 학습 성능 차트 (있는 경우)

---
*이 보고서는 멀티모달 훈련 통합 스위트에 의해 자동 생성되었습니다.*
