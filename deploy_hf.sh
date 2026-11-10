#!/usr/bin/env bash
# Usage: ./deploy_hf.sh <hf-username>/<space-name>
# Requires: `hf auth login` already done. Uploads a clean snapshot (no git history).
set -euo pipefail
SPACE="${1:?usage: ./deploy_hf.sh <user>/<space>}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cat > "$STAGE/README.md" <<EOF
---
title: PhotoRevive Colorizer
emoji: 🎨
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

Automatic black-and-white photo colorization with DDColor.
EOF

cp Dockerfile requirements-space.txt app.py index.html "$STAGE/"
mkdir -p "$STAGE/services/ddcolor"
cp -r services/ddcolor/ddcolor services/ddcolor/basicsr "$STAGE/services/ddcolor/"
find "$STAGE" -name __pycache__ -type d -prune -exec rm -rf {} +

hf repo create "$SPACE" --repo-type space --space_sdk docker --exist-ok
hf upload "$SPACE" "$STAGE" . --repo-type space --commit-message "Deploy"
echo "Live at: https://huggingface.co/spaces/$SPACE"
