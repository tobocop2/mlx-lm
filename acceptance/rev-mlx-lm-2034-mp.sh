#!/usr/bin/env bash
set -u
cd src
export HF_HOME=. METAL_DEVICE_WRAPPER_TYPE=1 METAL_DEBUG_ERROR_MODE=0
echo "REVIEW-HEAD sha=$(git rev-parse HEAD) tree=$(git rev-parse 'HEAD^{tree}')"
curl -s -o test_data.zip -L https://github.com/ml-explore/mlx-lm/releases/download/test_data/test_data.zip
unzip -q test_data.zip
uv run mlx.launch -n 2 python tests/model_parallel_tests.py > ../mp.log 2>&1
rc=$?
echo "STEP_RESULT name=model_parallel_tests rc=$rc"
grep -E "^(Ran [0-9]+ tests|OK|FAILED)" ../mp.log | sort | uniq -c | sed 's/^/STEP_RESULT mp-summary /'
grep -E "^(ERROR|FAIL):" ../mp.log | sort | uniq -c | sed 's/^/STEP_RESULT mp-red /'
grep -E "^[A-Za-z]*(Error|Exception)[:(]" ../mp.log | sort | uniq -c | sed 's/^/STEP_RESULT mp-exc /' | cut -c1-300
grep -E "exited with code" ../mp.log | sed 's/^/STEP_RESULT mp-exit /'
echo "ACCEPTANCE-DONE rc=$rc"
