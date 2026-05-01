#!/usr/bin/env bash
# Creates the submission zip in the required Nume_Grupa format.
set -euo pipefail

# ── EDIT THESE ──────────────────────────────────────────────
STUDENT_NAME="Tudose_Alexandru"
GRUPA="10LF333"
# ────────────────────────────────────────────────────────────

FOLDER="${STUDENT_NAME}_${GRUPA}"
ZIP_NAME="${FOLDER}.zip"
TMPDIR_PATH=$(mktemp -d)
DEST="$TMPDIR_PATH/$FOLDER"

cd "$(dirname "$0")"

mkdir -p "$DEST"

# Copy source files (professor expects .cpp/.c/.cu)
cp src/*.cu "$DEST/"

# Copy readme (Task 2 document)
if [ -f docs/readme.pdf ]; then
    cp docs/readme.pdf "$DEST/"
elif [ -f docs/readme.doc ]; then
    cp docs/readme.doc "$DEST/"
else
    echo "WARNING: docs/readme.pdf or docs/readme.doc not found — submission incomplete."
fi

# Copy plagiarism screenshot(s) if present
for f in *.png *.jpg *.jpeg *.webp; do
    [ -f "$f" ] && cp "$f" "$DEST/"
done

(cd "$TMPDIR_PATH" && zip -r - "$FOLDER") > "$ZIP_NAME"
rm -rf "$TMPDIR_PATH"

echo "Created: $ZIP_NAME"
echo "Upload this to the Elearning platform before 3 Mai 2026, 23:59."
