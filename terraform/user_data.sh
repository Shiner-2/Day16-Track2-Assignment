#!/bin/bash
set -euo pipefail

exec > >(tee /var/log/user-data.log | logger -t user-data -s 2>/dev/console) 2>&1

echo "Starting CPU LightGBM benchmark setup"

dnf update -y
dnf install -y python3 python3-pip python3-virtualenv unzip libgomp

install -d -o ec2-user -g ec2-user /home/ec2-user/ml-benchmark
cd /home/ec2-user/ml-benchmark

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install lightgbm scikit-learn pandas numpy kaggle

cat > benchmark.py <<'PY'
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


RESULT_PATH = Path("benchmark_result.json")
DATA_PATH = Path("creditcard.csv")


def now():
    return time.perf_counter()


def load_data():
    start = now()
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        y = df["Class"].astype(int)
        x = df.drop(columns=["Class"])
        source = "creditcard.csv"
    else:
        x_np, y_np = make_classification(
            n_samples=284_807,
            n_features=30,
            n_informative=20,
            n_redundant=5,
            weights=[0.998, 0.002],
            random_state=42,
        )
        x = pd.DataFrame(x_np, columns=[f"feature_{i}" for i in range(x_np.shape[1])])
        y = pd.Series(y_np, name="Class")
        source = "synthetic_fraud_like"
    return x, y, source, now() - start


def main():
    x, y, source, load_seconds = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=-1,
        random_state=42,
    )

    train_start = now()
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_test, y_test)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )
    training_seconds = now() - train_start

    proba = model.predict_proba(x_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    one_row = x_test.iloc[[0]]
    batch = x_test.iloc[:1000]

    latency_start = now()
    model.predict_proba(one_row)
    one_row_latency_ms = (now() - latency_start) * 1000

    throughput_start = now()
    model.predict_proba(batch)
    batch_seconds = now() - throughput_start

    results = {
        "dataset": source,
        "rows": int(len(x)),
        "features": int(x.shape[1]),
        "load_data_seconds": round(load_seconds, 4),
        "training_seconds": round(training_seconds, 4),
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "auc_roc": round(float(roc_auc_score(y_test, proba)), 6),
        "accuracy": round(float(accuracy_score(y_test, pred)), 6),
        "f1_score": round(float(f1_score(y_test, pred, zero_division=0)), 6),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 6),
        "inference_latency_1_row_ms": round(one_row_latency_ms, 4),
        "inference_throughput_1000_rows_per_second": round(1000 / batch_seconds, 2),
    }

    RESULT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
PY

cat > run_benchmark.sh <<'SH'
#!/bin/bash
set -euo pipefail
cd /home/ec2-user/ml-benchmark
exec .venv/bin/python benchmark.py
SH

cat > health_server.py <<'PY'
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok","service":"lightgbm-cpu"}')
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        return


HTTPServer(("0.0.0.0", 8000), Handler).serve_forever()
PY

cat > /etc/systemd/system/lightgbm-health.service <<'SERVICE'
[Unit]
Description=LightGBM CPU node health endpoint
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/ml-benchmark
ExecStart=/home/ec2-user/ml-benchmark/.venv/bin/python /home/ec2-user/ml-benchmark/health_server.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICE

chown -R ec2-user:ec2-user /home/ec2-user/ml-benchmark
chmod +x /home/ec2-user/ml-benchmark/run_benchmark.sh
systemctl daemon-reload
systemctl enable --now lightgbm-health.service

echo "CPU LightGBM setup complete. Run: cd ~/ml-benchmark && ./run_benchmark.sh"
