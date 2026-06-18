# CPU LightGBM Fallback Report

Terraform da duoc chuyen sang phuong an CPU theo Phan 7: Amazon Linux 2023 + `r5.xlarge` cho CPU node va `t3.micro` cho bastion. Ly do dung `r5.xlarge` thay vi `r5.2xlarge`: AWS account hien co quota 8 vCPU; `r5.2xlarge` dung het 8 vCPU nen khong tao them bastion duoc. `r5.xlarge` van co 32 GB RAM va chi dung 4 vCPU, tong voi bastion la 6 vCPU.

Endpoint health da chay thanh cong:

```text
http://ai-inference-alb-1ccc8d5d-1858038333.us-east-1.elb.amazonaws.com/health
HTTP 200: {"status":"ok","service":"lightgbm-cpu"}
```

Benchmark da chay tren CPU node va luu ket qua vao `benchmark_result.json`. Lan chay hien tai dung dataset fallback `synthetic_fraud_like` vi chua cau hinh Kaggle credentials tren EC2.

| Metric | Result |
|---|---:|
| Rows | 284807 |
| Features | 30 |
| Load data time | 0.5236 s |
| Training time | 2.3614 s |
| Best iteration | 57 |
| AUC-ROC | 0.631897 |
| Accuracy | 0.993136 |
| F1-score | 0.034568 |
| Precision | 0.538462 |
| Recall | 0.017857 |
| Inference latency, 1 row | 1.3353 ms |
| Inference throughput, 1000 rows | 435480.89 rows/s |
