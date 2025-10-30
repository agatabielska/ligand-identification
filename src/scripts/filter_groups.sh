#!/bin/bash

# Cluster Consolidation Script
# Consolidates small clusters (<100 files) and organizes rare ligands

# Don't exit on error - we want to see what's happening
set +e

# Get script directory
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/"

# Configuration
DATA_DIR="${SCRIPT_DIR}../../data/cryoem_blobs"
MIN_CLUSTER_SIZE="${1:-100}"  # Default 100 files minimum, can override with first argument

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Cluster Consolidation & Rare Ligand Organizer"
echo "=========================================="
echo "Script directory: $SCRIPT_DIR"
echo "Working directory: $DATA_DIR"
echo "Minimum cluster size: $MIN_CLUSTER_SIZE files"

# Validate data directory
if [[ ! -d "$DATA_DIR" ]]; then
    echo -e "${RED}❌ Error: cryoem_blobs directory not found: $DATA_DIR${NC}"
    exit 1
fi

# Change to cryoem_blobs directory
cd "$DATA_DIR"
echo "Working in: $(pwd)"

echo -e "\n${BLUE}[1/3] Scanning folders in cryoem_blobs directory...${NC}"

# Find all subdirectories in data folder
mapfile -t CLUSTER_DIRS < <(find . -mindepth 1 -maxdepth 1 -type d | sort)
TOTAL_CLUSTERS=${#CLUSTER_DIRS[@]}

if [[ $TOTAL_CLUSTERS -eq 0 ]]; then
    echo -e "${YELLOW}⚠ No folders found in cryoem_blobs directory${NC}"
else
    echo "Found $TOTAL_CLUSTERS folders to analyze"
fi

echo -e "\n${BLUE}[2/3] Processing folders (consolidating small clusters)...${NC}"
echo "Starting to process folders..."

CONSOLIDATED_FOLDERS=0
CONSOLIDATED_FILES=0
KEPT_FOLDERS=0
PROCESSED=0

for cluster_dir in "${CLUSTER_DIRS[@]}"; do
    result=$((PROCESSED + 1))
    PROCESSED=$result
    # Remove leading './' from dirname
    cluster_name="${cluster_dir#./}"

    echo -e "${CYAN}DEBUG: Processing folder $PROCESSED: '$cluster_name' (path: '$cluster_dir')${NC}"

    # Progress indicator every 10 folders (more frequent for debugging)
    if (( PROCESSED % 10 == 0 )); then
        echo -e "${YELLOW}  Progress: $PROCESSED/$TOTAL_CLUSTERS folders processed...${NC}"
    fi

    # Count files in the folder (all files, not just .npz)
    echo -e "${CYAN}DEBUG: Running find on '$cluster_dir'${NC}"
    file_count=$(find "$cluster_dir" -maxdepth 1 -type f 2>/dev/null | wc -l)
    echo -e "${CYAN}DEBUG: Found $file_count files${NC}"

    if [[ $file_count -lt $MIN_CLUSTER_SIZE ]]; then
        echo -e "${CYAN}  Consolidating: $cluster_name ($file_count files)${NC}"

        # Move all files from cluster back to cryoem_blobs directory
        moved=0
        failed=0

        if [[ $file_count -gt 0 ]]; then
            # Get all files at once
            mapfile -t files < <(find "$cluster_dir" -maxdepth 1 -type f 2>/dev/null)

            # Debug
            if (( PROCESSED <= 3 )); then
                echo -e "${CYAN}    Moving ${#files[@]} files...${NC}"
            fi

            for file in "${files[@]}"; do
                filename=$(basename "$file")
                if mv "$file" "./$filename" 2>/dev/null; then
                    ((moved++))
                else
                    ((failed++))
                    echo -e "${RED}    ⚠ Failed to move: $filename${NC}"
                fi
            done

            # Debug
            if (( PROCESSED <= 3 )); then
                echo -e "${CYAN}    Moved: $moved, Failed: $failed${NC}"
            fi
        fi

        # Remove now-empty cluster directory
        if rmdir "$cluster_dir" 2>/dev/null; then
            ((CONSOLIDATED_FOLDERS++))
            CONSOLIDATED_FILES=$((CONSOLIDATED_FILES + moved))
            if [[ $moved -gt 0 ]]; then
                echo -e "${GREEN}    ✓ Moved $moved files, removed folder${NC}"
            else
                echo -e "${GREEN}    ✓ Removed empty folder${NC}"
            fi
            if [[ $failed -gt 0 ]]; then
                echo -e "${YELLOW}    ⚠ Failed to move $failed files${NC}"
            fi
        else
            echo -e "${RED}    ⚠ Could not remove folder (not empty or permission issue)${NC}"
        fi
    else
        ((KEPT_FOLDERS++))
        if (( PROCESSED <= 10 || PROCESSED % 50 == 0 )); then
            echo -e "${GREEN}  Keeping: $cluster_name ($file_count files)${NC}"
        fi
    fi
done

echo -e "${YELLOW}Finished processing all $PROCESSED folders${NC}"

echo -e "\n${CYAN}Summary of consolidation:${NC}"
echo "  - Total folders processed: $PROCESSED"
echo "  - Kept folders (≥$MIN_CLUSTER_SIZE files): $KEPT_FOLDERS"
echo "  - Consolidated folders: $CONSOLIDATED_FOLDERS"
echo "  - Files moved back to cryoem_blobs directory: $CONSOLIDATED_FILES"

echo -e "\n${BLUE}[3/3] Creating RARE folder and organizing loose files...${NC}"

RARE_DIR="./RARE"
mkdir -p "$RARE_DIR"
echo "Created/verified folder: $RARE_DIR"

# Find all files directly in cryoem_blobs directory (not in subdirectories)
echo "Scanning for loose files in cryoem_blobs directory..."
mapfile -t LOOSE_FILES < <(find . -maxdepth 1 -type f 2>/dev/null)
LOOSE_COUNT=${#LOOSE_FILES[@]}

if [[ $LOOSE_COUNT -eq 0 ]]; then
    echo -e "${GREEN}✓ No loose files found - all organized!${NC}"
else
    echo "Found $LOOSE_COUNT loose files to move to RARE"

    RARE_MOVED=0
    RARE_ERRORS=0

    for file in "${LOOSE_FILES[@]}"; do
        filename=$(basename "$file")
        if mv "$file" "$RARE_DIR/$filename" 2>/dev/null; then
            ((RARE_MOVED++))
            if (( RARE_MOVED % 1000 == 0 )); then
                echo -e "${CYAN}  Progress: $RARE_MOVED/$LOOSE_COUNT files moved...${NC}"
            fi
        else
            ((RARE_ERRORS++))
            echo -e "${RED}  ⚠ Failed to move: $filename${NC}"
        fi
    done

    echo -e "${GREEN}✓ Moved $RARE_MOVED files to RARE${NC}"

    if [[ $RARE_ERRORS -gt 0 ]]; then
        echo -e "${YELLOW}⚠ Encountered $RARE_ERRORS errors${NC}"
    fi
fi

# Final count of RARE folder
RARE_FINAL_COUNT=$(find "$RARE_DIR" -maxdepth 1 -type f 2>/dev/null | wc -l)

echo -e "\n=========================================="
echo -e "${GREEN}✓ Consolidation complete!${NC}"
echo ""
echo "Final structure:"
echo "  - Large clusters (≥$MIN_CLUSTER_SIZE files): $KEPT_FOLDERS folders"
echo "  - RARE folder: $RARE_FINAL_COUNT files"
echo "  - Consolidated: $CONSOLIDATED_FILES files from $CONSOLIDATED_FOLDERS small clusters"
echo ""
echo "  cryoem_blobs directory: $(realpath .)"
echo "  RARE directory: $(realpath "$RARE_DIR")"
echo "=========================================="