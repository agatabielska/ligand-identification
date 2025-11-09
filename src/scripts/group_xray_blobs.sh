#!/usr/bin/bash

# Ligand File Organizer by Cluster (Updated)
# Creates train/holdout folder structure based on CSV files
# Organizes .npz files into folders based on ligand groups

# Get script directory
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/"

# Configuration
TRAIN_CSV="${SCRIPT_DIR}../../data/xray_train.csv"
HOLDOUT_CSV="${SCRIPT_DIR}../../data/xray_holdout.csv"
SOURCE_DIR="${SCRIPT_DIR}../../data/xray_blobs"
MAX_WORKERS="${1:-16}"  # Default 16 workers, can override with first argument

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "X-ray Ligand Folder Structure Setup"
echo "=========================================="
echo "Script directory: $SCRIPT_DIR"

# Change to script directory
cd "$SCRIPT_DIR"

# Validate inputs
if [[ ! -d "$SOURCE_DIR" ]]; then
    echo -e "${RED}✖ Error: Source directory not found: $SOURCE_DIR${NC}"
    exit 1
fi

if [[ ! -f "$TRAIN_CSV" ]]; then
    echo -e "${RED}✖ Error: Training CSV not found: $TRAIN_CSV${NC}"
    exit 1
fi

if [[ ! -f "$HOLDOUT_CSV" ]]; then
    echo -e "${RED}✖ Error: Holdout CSV not found: $HOLDOUT_CSV${NC}"
    exit 1
fi

echo -e "\n${BLUE}[Step 1/3] Creating main directories...${NC}"

# Create xray_train and xray_holdout directories
TRAIN_DIR="${SOURCE_DIR}/xray_train"
HOLDOUT_DIR="${SOURCE_DIR}/xray_holdout"

if [[ -d "$TRAIN_DIR" ]]; then
    echo "  xray_train/ - already exists"
else
    mkdir -p "$TRAIN_DIR"
    echo -e "  ${GREEN}✓ Created xray_train/${NC}"
fi

if [[ -d "$HOLDOUT_DIR" ]]; then
    echo "  xray_holdout/ - already exists"
else
    mkdir -p "$HOLDOUT_DIR"
    echo -e "  ${GREEN}✓ Created xray_holdout/${NC}"
fi

echo -e "\n${BLUE}[Step 2/3] Extracting unique ligand groups from xray_train.csv...${NC}"

# Extract unique ligand groups from training CSV (skip header, get 2nd column)
TEMP_LIGANDS=$(mktemp)

awk -F',' '
NR == 1 { next }  # Skip header
{
    ligand = $2
    # Trim whitespace and quotes
    gsub(/^[ \t\r\n"]+|[ \t\r\n"]+$/, "", ligand)
    if (ligand != "") {
        print ligand
    }
}' "$TRAIN_CSV" | sort -u > "$TEMP_LIGANDS"

UNIQUE_GROUPS=$(wc -l < "$TEMP_LIGANDS")
echo "Found $UNIQUE_GROUPS unique ligand groups"

echo -e "\n${BLUE}[Step 3/3] Creating ligand group folders...${NC}"

TRAIN_CREATED=0
TRAIN_EXISTS=0
HOLDOUT_CREATED=0
HOLDOUT_EXISTS=0

while IFS= read -r group; do
    if [[ -z "$group" ]]; then
        continue
    fi

    # Sanitize folder name (replace problematic characters)
    safe_group=$(echo "$group" | tr '/' '_' | tr '\\' '_')

    # Create in xray_train
    if [[ -d "$TRAIN_DIR/$safe_group" ]]; then
        ((TRAIN_EXISTS++))
    else
        mkdir -p "$TRAIN_DIR/$safe_group"
        ((TRAIN_CREATED++))
    fi

    # Create in xray_holdout
    if [[ -d "$HOLDOUT_DIR/$safe_group" ]]; then
        ((HOLDOUT_EXISTS++))
    else
        mkdir -p "$HOLDOUT_DIR/$safe_group"
        ((HOLDOUT_CREATED++))
    fi

done < "$TEMP_LIGANDS"

# Cleanup
rm -f "$TEMP_LIGANDS"

echo -e "\n=========================================="
echo -e "${GREEN}✓ Folder structure setup complete!${NC}"
echo ""
echo "xray_train/ folders:"
echo "  - Already existing: $TRAIN_EXISTS"
echo "  - Newly created: $TRAIN_CREATED"
echo "  - Total: $UNIQUE_GROUPS"
echo ""
echo "xray_holdout/ folders:"
echo "  - Already existing: $HOLDOUT_EXISTS"
echo "  - Newly created: $HOLDOUT_CREATED"
echo "  - Total: $UNIQUE_GROUPS"
echo ""
echo "Directory structure:"
echo "  $(realpath "$SOURCE_DIR")"
echo "  ├── xray_train/"
echo "  │   └── [${UNIQUE_GROUPS} ligand group folders]"
echo "  └── xray_holdout/"
echo "      └── [${UNIQUE_GROUPS} ligand group folders]"
echo "=========================================="

# ============================================================
# PART 2: Move files to appropriate folders
# ============================================================

echo -e "\n${BLUE}[Step 4/6] Finding .npz files in root directory...${NC}"

# Find all .npz files in root directory only (not in subdirectories)
mapfile -t ROOT_FILES < <(find "$SOURCE_DIR" -maxdepth 1 -type f -name "*.npz")
TOTAL_ROOT_FILES=${#ROOT_FILES[@]}

echo "Found $TOTAL_ROOT_FILES .npz files in root directory"

if [[ $TOTAL_ROOT_FILES -eq 0 ]]; then
    echo -e "${GREEN}✓ No files to organize (all files already in folders)${NC}"
    echo "=========================================="
    exit 0
fi

echo -e "\n${BLUE}[Step 5/6] Processing training set files...${NC}"

# Build filename -> ligand group mapping from xray_train.csv
TEMP_TRAIN_MAP=$(mktemp)

awk -F',' '
NR == 1 { next }  # Skip header
{
    filename = $1
    ligand = $2
    # Trim whitespace and quotes
    gsub(/^[ \t\r\n"]+|[ \t\r\n"]+$/, "", filename)
    gsub(/^[ \t\r\n"]+|[ \t\r\n"]+$/, "", ligand)
    if (filename != "" && ligand != "") {
        print filename ":" ligand
    }
}' "$TRAIN_CSV" > "$TEMP_TRAIN_MAP"

TRAIN_MAPPINGS=$(wc -l < "$TEMP_TRAIN_MAP")
echo "Loaded $TRAIN_MAPPINGS filename mappings from xray_train.csv"

# Function to move file to training folder
move_to_train() {
    local file="$1"
    local filename=$(basename "$file")

    # Look up ligand group from mapping
    local ligand=$(grep "^${filename}:" /tmp/train_map_xray.txt 2>/dev/null | cut -d':' -f2)

    if [[ -n "$ligand" ]]; then
        local safe_ligand=$(echo "$ligand" | tr '/' '_' | tr '\\' '_')
        local dest="${SOURCE_DIR}/xray_train/${safe_ligand}/${filename}"

        # Check if file already exists at destination
        if [[ -f "$dest" ]]; then
            rm "$file"
            echo "train_duplicate:$filename:$ligand"
            return
        fi

        if mv "$file" "$dest" 2>/dev/null; then
            echo "train_moved:$filename:$ligand"
        else
            echo "train_error:$filename:$ligand"
        fi
    fi
}

export -f move_to_train
export SOURCE_DIR

# Copy mapping to /tmp for access in parallel processes
cp "$TEMP_TRAIN_MAP" /tmp/train_map_xray.txt

# Process files in parallel with xargs
echo "Using xargs with $MAX_WORKERS workers for training set..."
printf '%s\n' "${ROOT_FILES[@]}" | \
    xargs -P "$MAX_WORKERS" -I {} bash -c 'move_to_train "$@"' _ {} > /tmp/train_results_xray.txt

wait

# Count results
TRAIN_MOVED=0
TRAIN_DUPLICATES=0
TRAIN_ERRORS=0

while IFS=':' read -r status filename ligand; do
    case "$status" in
        train_moved)
            ((TRAIN_MOVED++))
            ;;
        train_duplicate)
            ((TRAIN_DUPLICATES++))
            ;;
        train_error)
            ((TRAIN_ERRORS++))
            ;;
    esac
done < /tmp/train_results_xray.txt

echo -e "${GREEN}✓ Training set processing complete${NC}"
echo "  - Files moved: $TRAIN_MOVED"
echo "  - Duplicates removed: $TRAIN_DUPLICATES"
echo "  - Errors: $TRAIN_ERRORS"

echo -e "\n${BLUE}[Step 6/6] Processing holdout set files...${NC}"

# Re-scan for remaining files in root directory
mapfile -t REMAINING_FILES < <(find "$SOURCE_DIR" -maxdepth 1 -type f -name "*.npz")
REMAINING_COUNT=${#REMAINING_FILES[@]}

echo "Found $REMAINING_COUNT .npz files remaining in root directory"

if [[ $REMAINING_COUNT -eq 0 ]]; then
    echo -e "${GREEN}✓ No remaining files to process${NC}"
    rm -f "$TEMP_TRAIN_MAP" /tmp/train_map_xray.txt /tmp/train_results_xray.txt
    echo "=========================================="
    exit 0
fi

# Build filename -> ligand group mapping from xray_holdout.csv
TEMP_HOLDOUT_MAP=$(mktemp)

awk -F',' '
NR == 1 { next }  # Skip header
{
    filename = $1
    ligand = $2
    # Trim whitespace and quotes
    gsub(/^[ \t\r\n"]+|[ \t\r\n"]+$/, "", filename)
    gsub(/^[ \t\r\n"]+|[ \t\r\n"]+$/, "", ligand)
    if (filename != "" && ligand != "") {
        print filename ":" ligand
    }
}' "$HOLDOUT_CSV" > "$TEMP_HOLDOUT_MAP"

HOLDOUT_MAPPINGS=$(wc -l < "$TEMP_HOLDOUT_MAP")
echo "Loaded $HOLDOUT_MAPPINGS filename mappings from xray_holdout.csv"

# Function to move file to holdout folder
move_to_holdout() {
    local file="$1"
    local filename=$(basename "$file")

    # Look up ligand group from mapping
    local ligand=$(grep "^${filename}:" /tmp/holdout_map_xray.txt 2>/dev/null | cut -d':' -f2)

    if [[ -n "$ligand" ]]; then
        local safe_ligand=$(echo "$ligand" | tr '/' '_' | tr '\\' '_')
        local dest="${SOURCE_DIR}/xray_holdout/${safe_ligand}/${filename}"

        # Check if file already exists at destination
        if [[ -f "$dest" ]]; then
            rm "$file"
            echo "holdout_duplicate:$filename:$ligand"
            return
        fi

        if mv "$file" "$dest" 2>/dev/null; then
            echo "holdout_moved:$filename:$ligand"
        else
            echo "holdout_error:$filename:$ligand"
        fi
    else
        # File not in holdout mapping - leave in root
        echo "unmatched:$filename"
    fi
}

export -f move_to_holdout

# Copy mapping to /tmp for access in parallel processes
cp "$TEMP_HOLDOUT_MAP" /tmp/holdout_map_xray.txt

# Process files in parallel with xargs
echo "Using xargs with $MAX_WORKERS workers for holdout set..."
printf '%s\n' "${REMAINING_FILES[@]}" | \
    xargs -P "$MAX_WORKERS" -I {} bash -c 'move_to_holdout "$@"' _ {} > /tmp/holdout_results_xray.txt

wait

# Count results
HOLDOUT_MOVED=0
HOLDOUT_DUPLICATES=0
HOLDOUT_ERRORS=0
UNMATCHED=0

while IFS=':' read -r status filename ligand; do
    case "$status" in
        holdout_moved)
            ((HOLDOUT_MOVED++))
            ;;
        holdout_duplicate)
            ((HOLDOUT_DUPLICATES++))
            ;;
        holdout_error)
            ((HOLDOUT_ERRORS++))
            ;;
        unmatched)
            ((UNMATCHED++))
            ;;
    esac
done < /tmp/holdout_results_xray.txt

echo -e "${GREEN}✓ Holdout set processing complete${NC}"
echo "  - Files moved: $HOLDOUT_MOVED"
echo "  - Duplicates removed: $HOLDOUT_DUPLICATES"
echo "  - Errors: $HOLDOUT_ERRORS"
echo "  - Unmatched (stayed in root): $UNMATCHED"

# Final summary
echo -e "\n=========================================="
echo -e "${GREEN}✓ File organization complete!${NC}"
echo ""
echo "Training set:"
echo "  - Files moved: $TRAIN_MOVED"
echo "  - Duplicates: $TRAIN_DUPLICATES"
echo "  - Errors: $TRAIN_ERRORS"
echo ""
echo "Holdout set:"
echo "  - Files moved: $HOLDOUT_MOVED"
echo "  - Duplicates: $HOLDOUT_DUPLICATES"
echo "  - Errors: $HOLDOUT_ERRORS"
echo ""
echo "Files remaining in root: $UNMATCHED"
echo "=========================================="

# Cleanup
rm -f "$TEMP_TRAIN_MAP" "$TEMP_HOLDOUT_MAP"
rm -f /tmp/train_map_xray.txt /tmp/holdout_map_xray.txt
rm -f /tmp/train_results_xray.txt /tmp/holdout_results_xray.txt

exit 0