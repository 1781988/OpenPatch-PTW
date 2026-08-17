#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-./data/COCO2017}"
mkdir -p "${TARGET}"
cd "${TARGET}"

fetch() {
  local url="$1"
  local file="$2"
  if [[ -s "${file}" ]]; then
    echo "[skip] ${file} already exists"
    return
  fi
  if command -v wget >/dev/null 2>&1; then
    wget --continue --tries=5 --timeout=60 "${url}" -O "${file}"
  else
    curl --location --fail --retry 5 --retry-delay 5 --continue-at - "${url}" -o "${file}"
  fi
}

fetch "https://images.cocodataset.org/zips/train2017.zip" "train2017.zip"
fetch "https://images.cocodataset.org/zips/val2017.zip" "val2017.zip"
fetch "https://images.cocodataset.org/annotations/annotations_trainval2017.zip" "annotations_trainval2017.zip"

unzip -q -n train2017.zip
unzip -q -n val2017.zip
unzip -q -n annotations_trainval2017.zip

for required in train2017 val2017 annotations/instances_train2017.json annotations/instances_val2017.json; do
  [[ -e "${required}" ]] || { echo "Missing after extraction: ${required}" >&2; exit 1; }
done

echo "COCO2017 ready at: $(pwd)"
