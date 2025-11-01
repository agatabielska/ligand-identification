#!/usr/bin/bash

# Get Common Ligands Script
# Identifies ligands with 100 or more total files across cryoem_blobs and xray_blobs
# Outputs a list of common ligands to common_ligands.txt


# Get script directory
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/"

# Configuration
CRYOEM_DIR="${SCRIPT_DIR}../../data/cryoem_blobs"
XRAY_DIR="${SCRIPT_DIR}../../data/xray_blobs"
OUTPUT_FILE="${SCRIPT_DIR}../../data/ligand_groups.txt"
THRESHOLD=100

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
    echo -e "${YELLOW}⚠ Warning: CryoEM directory not found: $CRYOEM_DIR${NC}"
    echo "  Will only count from X-ray directory"
fi

if [[ ! -d "$XRAY_DIR" ]]; then
    echo -e "${YELLOW}⚠ Warning: X-ray directory not found: $XRAY_DIR${NC}"
    echo "  Will only count from CryoEM directory"
fi

if [[ ! -d "$CRYOEM_DIR" ]] && [[ ! -d "$XRAY_DIR" ]]; then
    echo -e "${RED}❌ Error: Neither CryoEM nor X-ray directories found${NC}"
    exit 1
fi

# Create temporary file for counting
TEMP_COUNT_FILE=$(mktemp)
trap "rm -f $TEMP_COUNT_FILE" EXIT

echo -e "\n${BLUE}[1/3] Counting files in X-ray ligand folders...${NC}"

# Count files in X-ray directory
if [[ -d "$XRAY_DIR" ]]; then
    XRAY_FOLDER_COUNT=0

    # Find all subdirectories (ligand folders)
    while IFS= read -r -d '' ligand_folder; do
        ligand_name=$(basename "$ligand_folder")

        # Count .npz files in this folder
        file_count=$(find "$ligand_folder" -maxdepth 1 -type f -name "*.npz" | wc -l)

        if [[ $file_count -gt 0 ]]; then
            echo "$ligand_name $file_count" >> "$TEMP_COUNT_FILE"
            ((XRAY_FOLDER_COUNT++))
        fi
    done < <(find "$XRAY_DIR" -mindepth 1 -maxdepth 1 -type d -print0)

    echo "Processed $XRAY_FOLDER_COUNT ligand folders from X-ray"
else
    echo "Skipping X-ray (directory not found)"
fi

echo -e "\n${BLUE}[2/3] Counting files in CryoEM ligand folders...${NC}"

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