#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-./data/COCO2017}"
mkdir -p "${TARGET}"
cd "${TARGET}"

fetch() {
  local url="$1"
  local file="$2"
  if [[ -f "${file}" ]]; then
    echo "[skip] ${file} already exists"
    return
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -c "${url}" -O "${file}"
  else
    curl -L --retry 3 -C - "${url}" -o "${file}"
  fi
}

fetch "http://images.cocodataset.org/zips/train2017.zip" "train2017.zip"
fetch "http://images.cocodataset.org/zips/val2017.zip" "val2017.zip"
fetch "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" "annotations_trainval2017.zip"

unzip -q -n train2017.zip
unzip -q -n val2017.zip
unzip -q -n annotations_trainval2017.zip

echo "COCO2017 ready at: $(pwd)"
