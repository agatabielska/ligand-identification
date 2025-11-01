#!/bin/bash
set -euo pipefail

# Get script directory
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/"

# Configuration
LIGAND_GROUPS="${SCRIPT_DIR}../../data/ligand_groups.txt"
XRAY_BLOBS_DIR="${SCRIPT_DIR}../../data/xray_blobs"
RARE_DIR="${XRAY_BLOBS_DIR}/RARE"
MAX_WORKERS="${1:-16}"  # Default 16 workers, can override with first argument

# Validation
if [[ ! -f "$LIGAND_GROUPS" ]]; then
    echo "Error: Ligand groups file not found: $LIGAND_GROUPS"
    exit 1
fi

if [[ ! -d "$XRAY_BLOBS_DIR" ]]; then
    echo "Error: X-ray blobs directory not found: $XRAY_BLOBS_DIR"
    exit 1
fi

echo "Starting ligand folder organization..."
echo "Using $MAX_WORKERS workers"
echo "Ligand groups file: $LIGAND_GROUPS"
echo "X-ray blobs directory: $XRAY_BLOBS_DIR"
echo ""

# Create a temporary sorted file for fast lookups with grep
TEMP_LIGANDS=$(mktemp)
trap "rm -f $TEMP_LIGANDS" EXIT

# Sort ligand names for binary search with grep
sort -u "$LIGAND_GROUPS" | grep -v '^[[:space:]]*$' > "$TEMP_LIGANDS"

ligand_count=$(wc -l < "$TEMP_LIGANDS")
echo "Loaded $ligand_count valid ligand names"
echo ""

# Function to process a single folder
process_folder() {
    local folder="$1"
    local ligands_file="$2"
    local parent_dir="$3"
    local folder_name=$(basename "$folder")

    # Check if folder name is in valid ligands list using grep
    if grep -qFx "$folder_name" "$ligands_file"; then
        echo "keep:$folder_name"
        return 0
    fi

    # Move all files from this folder to parent directory
    local file_count=0
    for file in "$folder"/*; do
        [[ -e "$file" ]] || continue  # Skip if no files
        mv "$file" "$parent_dir/"
        ((file_count++))
    done

    # Remove empty folder
    if [[ $file_count -gt 0 ]]; then
        rmdir "$folder" 2>/dev/null || true
        echo "removed:$folder_name:$file_count"
    fi
}

export -f process_folder

# Step 1: Process all subdirectories in parallel
echo "Step 1: Processing subdirectories..."
find "$XRAY_BLOBS_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | \
    xargs -0 -P "$MAX_WORKERS" -I {} bash -c 'process_folder "$1" "$2" "$3"' _ {} "$TEMP_LIGANDS" "$XRAY_BLOBS_DIR" > /tmp/folder_processing_xray.log

# Count results
kept_count=$(grep -c "^keep:" /tmp/folder_processing_xray.log || true)
removed_count=$(grep -c "^removed:" /tmp/folder_processing_xray.log || true)

echo "Kept $kept_count valid folders, removed $removed_count invalid folders"
echo ""

# Step 2: Move all remaining loose files to RARE directory
echo "Step 2: Moving loose files to RARE directory..."

# Count loose files (excluding directories)
loose_files=$(find "$XRAY_BLOBS_DIR" -maxdepth 1 -type f | wc -l)

if [[ $loose_files -gt 0 ]]; then
    # Create RARE directory
    mkdir -p "$RARE_DIR"

    # Move all loose files to RARE
    find "$XRAY_BLOBS_DIR" -maxdepth 1 -type f -exec mv {} "$RARE_DIR/" \;

    echo "Moved $loose_files loose files to RARE directory"
else
    echo "No loose files found to move"
fi

# Clean up
rm -f /tmp/folder_processing_xray.log

echo ""
echo "Organization complete!"
echo ""
echo "Summary:"
echo "--------"
echo "Valid ligand folders retained: $(find "$XRAY_BLOBS_DIR" -mindepth 1 -maxdepth 1 -type d ! -name "RARE" | wc -l)"
if [[ -d "$RARE_DIR" ]]; then
    echo "Files in RARE directory: $(find "$RARE_DIR" -maxdepth 1 -type f | wc -l)"
else
    echo "Files in RARE directory: 0"
fi

exit 0