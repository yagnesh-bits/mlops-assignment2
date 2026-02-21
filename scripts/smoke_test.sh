#!/usr/bin/env bash
# scripts/smoke_test.sh
# Post-deploy smoke tests – called by CI/CD after deployment.
# Fails the pipeline (exit 1) if any check fails.

set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"
MAX_RETRIES=10
RETRY_DELAY=5

echo "=================================================="
echo "  Smoke Tests – Cats vs Dogs API"
echo "  Target: $API_URL"
echo "=================================================="

# ── Helper functions ──────────────────────────────────────────────────────────

pass() { echo "✅  PASS: $1"; }
fail() { echo "❌  FAIL: $1"; exit 1; }

# ── Wait for service to be ready ──────────────────────────────────────────────

echo ""
echo "⏳  Waiting for API to become healthy..."
for i in $(seq 1 $MAX_RETRIES); do
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/health" || echo "000")
  if [ "$STATUS" = "200" ]; then
    pass "API is reachable (HTTP 200)"
    break
  fi
  if [ "$i" = "$MAX_RETRIES" ]; then
    fail "API did not become healthy after $MAX_RETRIES attempts (last status: $STATUS)"
  fi
  echo "   Attempt $i/$MAX_RETRIES – status $STATUS – retrying in ${RETRY_DELAY}s..."
  sleep $RETRY_DELAY
done

# ── Test 1: Health endpoint ───────────────────────────────────────────────────

echo ""
echo "── Test 1: Health Endpoint ──────────────────────"
HEALTH=$(curl -sf "$API_URL/health")
echo "   Response: $HEALTH"

STATUS_FIELD=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
[ "$STATUS_FIELD" = "healthy" ] && pass "status == healthy" || fail "status != healthy (got: $STATUS_FIELD)"

MODEL_LOADED=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin)['model_loaded'])")
[ "$MODEL_LOADED" = "True" ] && pass "model_loaded == True" || fail "model_loaded != True"

# ── Test 2: Prediction endpoint with a synthetic image ───────────────────────

echo ""
echo "── Test 2: Prediction Endpoint ──────────────────"

# Create a minimal JPEG test image using Python
TMPIMG=$(mktemp --suffix=.jpg)
python3 - <<EOF
from PIL import Image
import numpy as np
arr = np.random.randint(0, 255, (224, 224, 3), dtype="uint8")
Image.fromarray(arr, "RGB").save("$TMPIMG", "JPEG")
print("Test image created: $TMPIMG")
EOF

PREDICT_RESP=$(curl -sf -X POST "$API_URL/predict" \
  -F "file=@$TMPIMG;type=image/jpeg")
echo "   Response: $PREDICT_RESP"
rm -f "$TMPIMG"

LABEL=$(echo "$PREDICT_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['label'])")
[[ "$LABEL" == "cat" || "$LABEL" == "dog" ]] && pass "label is valid ($LABEL)" || fail "label invalid: $LABEL"

CONFIDENCE=$(echo "$PREDICT_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['confidence'])")
python3 -c "assert 0 <= float('$CONFIDENCE') <= 1, 'confidence out of range'" && \
  pass "confidence in [0,1] ($CONFIDENCE)" || fail "confidence out of range: $CONFIDENCE"

# ── Test 3: Metrics endpoint ──────────────────────────────────────────────────

echo ""
echo "── Test 3: Prometheus Metrics Endpoint ──────────"
METRICS_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/metrics")
[ "$METRICS_STATUS" = "200" ] && pass "Metrics endpoint returns 200" || fail "Metrics endpoint returned $METRICS_STATUS"

# ── Test 4: Invalid file type returns 422 ────────────────────────────────────

echo ""
echo "── Test 4: Invalid Input Handling ───────────────"
INVALID_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_URL/predict" \
  -F "file=@/dev/null;type=text/plain")
[ "$INVALID_STATUS" = "422" ] && pass "Invalid file returns 422" || fail "Expected 422, got $INVALID_STATUS"

echo ""
echo "=================================================="
echo "  All smoke tests passed! ✅"
echo "=================================================="
