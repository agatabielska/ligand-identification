#!/usr/bin/bash

# Get Common Ligands Script
# Identifies ligands with 100 or more total files across cryoem_blobs and xray_blobs
# Outputs a list of common ligands to common_ligands.txt


# Get script directory
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/"

# Configuration
CRYOEM_DIR="${SCRIPT_DIR}../../data/cryoem_blobs"
OUTPUT_FILE="${SCRIPT_DIR}../../data/ligand_groups.txt"
THRESHOLD=50

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Get Common Ligands Script"
echo "=========================================="
echo "Threshold: >= $THRESHOLD files"
echo "Script directory: $SCRIPT_DIR"

# Change to script directory
cd "$SCRIPT_DIR"

# Validate inputs
if [[ ! -d "$CRYOEM_DIR" ]]; then
    echo -e "${RED}❌ Error: CryoEM directory not found: $CRYOEM_DIR${NC}"
    exit 1
fi


# Group .npz files into folders by ligand name
echo -e "\n${BLUE}[1/3] Grouping .npz files by ligand name...${NC}"

read GROUPED_COUNT SKIPPED_COUNT < <(python3 - "$CRYOEM_DIR" << 'EOF'
import os, sys
from pathlib import Path

cryoem_dir = Path(sys.argv[1])
grouped = 0
skipped = 0

# Scan root-level .npz files and group by ligand
moves = {}
for f in cryoem_dir.iterdir():
    if f.is_file() and f.suffix == '.npz':
        parts = f.stem.split('_')
        if len(parts) < 2:
            skipped += 1
            continue
        ligand = parts[-1]          # last segment → e.g. ADP
        moves.setdefault(ligand, []).append(f)

# Create each ligand dir once, then bulk-rename its files
for ligand, files in moves.items():
    target = cryoem_dir / ligand
    target.mkdir(exist_ok=True)
    for f in files:
        f.rename(target / f.name)
        grouped += 1

print(grouped, skipped)
EOF
)

echo "  ✅ Moved $GROUPED_COUNT files into ligand subdirectories"
[[ $SKIPPED_COUNT -gt 0 ]] && echo "  ⚠️  Skipped $SKIPPED_COUNT files with unparseable names"


echo -e "\n${BLUE}[2/3] Counting files in CryoEM ligand folders...${NC}"

# Create temporary file for counting
TEMP_COUNT_FILE=$(mktemp)
trap "rm -f $TEMP_COUNT_FILE" EXIT

# Count files in CryoEM directory
if [[ -d "$CRYOEM_DIR" ]]; then
    CRYOEM_FOLDER_COUNT=0

    # Find all subdirectories (ligand folders)
    while IFS= read -r -d '' ligand_folder; do
        ligand_name=$(basename "$ligand_folder")

        # Count .npz files in this folder
        file_count=$(find "$ligand_folder" -maxdepth 1 -type f -name "*.npz" | wc -l)

        if [[ $file_count -gt 0 ]]; then
            # Check if ligand already exists in temp file
            if grep -q "^${ligand_name} " "$TEMP_COUNT_FILE"; then
                # Add to existing count
                existing_count=$(grep "^${ligand_name} " "$TEMP_COUNT_FILE" | awk '{print $2}')
                new_count=$((existing_count + file_count))
                sed -i "s/^${ligand_name} ${existing_count}$/${ligand_name} ${new_count}/" "$TEMP_COUNT_FILE"
            else
                # New ligand, add to file
                echo "$ligand_name $file_count" >> "$TEMP_COUNT_FILE"
            fi
            ((CRYOEM_FOLDER_COUNT++))
        fi
    done < <(find "$CRYOEM_DIR" -mindepth 1 -maxdepth 1 -type d -print0)

    echo "Processed $CRYOEM_FOLDER_COUNT ligand folders from CryoEM"
else
    echo "Skipping CryoEM (directory not found)"
fi


echo -e "\n${BLUE}[3/3] Filtering common ligands (>= $THRESHOLD files)...${NC}"

# Filter ligands with count >= threshold and save to output file
> "$OUTPUT_FILE"
COMMON_COUNT=0
TOTAL_LIGANDS=$(wc -l < "$TEMP_COUNT_FILE")

echo "Ligands with >= $THRESHOLD files:" > /tmp/common_ligands_report.txt
echo "================================" >> /tmp/common_ligands_report.txt

while read -r ligand_name count; do
    if [[ $count -ge $THRESHOLD ]]; then
        echo "$ligand_name" >> "$OUTPUT_FILE"
        echo "  $ligand_name: $count files" >> /tmp/common_ligands_report.txt
        ((COMMON_COUNT++))
    fi
done < <(sort -k2 -n -r "$TEMP_COUNT_FILE")

echo -e "\n=========================================="
echo -e "${GREEN}✓ Analysis complete!${NC}"
echo ""
echo "Statistics:"
echo "  Total unique ligands: $TOTAL_LIGANDS"
echo "  Common ligands (>= $THRESHOLD files): $COMMON_COUNT"
echo "  Rare ligands (< $THRESHOLD files): $((TOTAL_LIGANDS - COMMON_COUNT))"
echo ""
echo "Output files:"
echo "  Common ligands list: $(realpath "$OUTPUT_FILE")"
echo "  Detailed report: /tmp/common_ligands_report.txt"
echo ""

if [[ $COMMON_COUNT -gt 0 ]]; then
    echo -e "${BLUE}Common ligands (sorted by count):${NC}"
    while read -r ligand; do
        count=$(grep "^${ligand} " "$TEMP_COUNT_FILE" | awk '{print $2}')
        echo "  $ligand ($count files)"
    done < "$OUTPUT_FILE"
else
    echo -e "${YELLOW}No ligands have >= $THRESHOLD files!${NC}"
fi

echo "=========================================="