#!/usr/bin/env bash
# Creates a zip of the project to upload to Google Colab.
set -euo pipefail

ZIP_NAME="gpu_sorting_colab.zip"

cd "$(dirname "$0")"

zip -r "$ZIP_NAME" \
    src/ \
    include/ \
    data/ \
    Makefile \
    --exclude "*.DS_Store" \
    --exclude "bin/*"

echo "Created: $ZIP_NAME"
echo "Upload this file in the Colab notebook (colab_runner.ipynb)."
