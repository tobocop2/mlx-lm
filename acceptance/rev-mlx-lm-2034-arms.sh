#!/usr/bin/env bash
# Review job for ml-explore/mlx-lm#2034 (rev-mlx-lm-2034): gate steps, mutation arms, flake loop.
# Runs from the harness workspace root; the ref under test is checked out in src/.
# ONLY: "[GATE] <regex on arm ids>" (the word GATE adds the gate steps and the flake loop).
set -u
HERE="$(pwd)"
cd src
export HF_HOME=. METAL_DEVICE_WRAPPER_TYPE=1 METAL_DEBUG_ERROR_MODE=0
spec="${ONLY:-}"
gate=0
case "$spec" in GATE*) gate=1; spec="${spec#GATE}";; esac
spec="${spec# }"
export ARMS_ONLY="${spec:-.*}"
echo "REVIEW-HEAD sha=$(git rev-parse HEAD) tree=$(git rev-parse 'HEAD^{tree}') gate=$gate arms_only=$ARMS_ONLY"
curl -s -o test_data.zip -L https://github.com/ml-explore/mlx-lm/releases/download/test_data/test_data.zip
unzip -q test_data.zip
echo "REVIEW test_data entries: $(ls | wc -l)"
if [ "$gate" = 1 ]; then
  uv run python -m unittest discover -v tests > ../full.log 2>&1
  rc=$?
  echo "STEP_RESULT name=unittest_discover rc=$rc ran=$(grep -E '^Ran [0-9]+ tests' ../full.log | tail -1) fails=$(grep -cE '^(FAIL|ERROR):' ../full.log) $(tail -1 ../full.log)"
  grep -E '^(FAIL|ERROR):' ../full.log | sort | uniq -c | head -20
  uv run mlx.launch -n 2 python tests/model_parallel_tests.py > ../mp.log 2>&1
  rc=$?
  echo "STEP_RESULT name=model_parallel_tests rc=$rc ran=$(grep -E '^Ran [0-9]+ tests' ../mp.log | tr '\n' '|') $(tail -3 ../mp.log | tr '\n' '|')"
  grep -E '^(FAIL|ERROR):' ../mp.log | sort | uniq -c | head
fi
uv run python "$HERE/acceptance/rev-mlx-lm-2034-arms.py"
echo "ARMS-DONE rc=$?"
if [ "$gate" = 1 ]; then
  KS="-k TestServerPromptTooLong -k TestDeclaredContextLength -k TestModelProviderContextLength -k TestCheckPromptLength -k test_prompt_too_long_is_a_bad_request -k test_other_generation_errors_stay_not_found"
  flake() {
    for i in 1 2 3 4 5 6 7 8 9 10; do
      uv run python -m unittest discover -v -s tests -p test_server.py $KS > "../flake-$1-$i.log" 2>&1
      echo "FLAKE stream=$1 run=$i rc=$? $(grep -E '^Ran [0-9]+ tests' ../flake-$1-$i.log | tr '\n' ' ')$(grep -E '^(FAIL|ERROR):' ../flake-$1-$i.log | tr '\n' ' ')"
    done
  }
  uv run python -m unittest discover -s tests -p test_server.py > ../flake-load.log 2>&1 &
  lp=$!
  flake 1 &
  p1=$!
  flake 2 &
  p2=$!
  wait $p1 $p2 $lp
  echo "FLAKE-LOAD rc-log: $(grep -E '^Ran [0-9]+ tests' ../flake-load.log | tr '\n' ' ') $(tail -1 ../flake-load.log)"
fi
echo "ACCEPTANCE-DONE rc=0"
