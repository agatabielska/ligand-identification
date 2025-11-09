#!/usr/bin/bash

# CryoEM Ligand File Organizer
# Organizes .npz files into folders based on xray training groups
# Uses ligand_mapping.csv for ligand to group matching
# Moves unmatched files to RARE_LIGAND folder

# Get script directory
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/"

# Configuration
DATA_DIR="${SCRIPT_DIR}../../data"
XRAY_TRAIN_DIR="${DATA_DIR}/xray_blobs/xray_train"
CRYOEM_DIR="${DATA_DIR}/cryoem_blobs"
XRAY_GROUPS_FILE="${DATA_DIR}/xray_groups.txt"
LIGAND_MAPPING_CSV="${DATA_DIR}/ligand_mapping.csv"
MAX_WORKERS="${1:-16}"  # Default 16 workers, can override with first argument

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "CryoEM Ligand File Organizer"
echo "=========================================="
echo "Script directory: $SCRIPT_DIR"

# Change to script directory
cd "$SCRIPT_DIR"

# Validate inputs
if [[ ! -d "$XRAY_TRAIN_DIR" ]]; then
    echo -e "${RED}✖ Error: X-ray training directory not found: $XRAY_TRAIN_DIR${NC}"
    exit 1
fi

if [[ ! -d "$CRYOEM_DIR" ]]; then
    echo -e "${RED}✖ Error: CryoEM directory not found: $CRYOEM_DIR${NC}"
    exit 1
fi

if [[ ! -f "$LIGAND_MAPPING_CSV" ]]; then
    echo -e "${RED}✖ Error: Ligand mapping CSV not found: $LIGAND_MAPPING_CSV${NC}"
    exit 1
fi

echo -e "\n${BLUE}[Step 1/6] Extracting group names from xray_train...${NC}"

# Get all folder names from xray_train directory
find "$XRAY_TRAIN_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort > "$XRAY_GROUPS_FILE"

GROUP_COUNT=$(wc -l < "$XRAY_GROUPS_FILE")
echo "Extracted $GROUP_COUNT group names from xray_train"
echo "Saved to: $(realpath "$XRAY_GROUPS_FILE")"

if [[ $GROUP_COUNT -eq 0 ]]; then
    echo -e "${RED}✖ Error: No groups found in xray_train directory${NC}"
    exit 1
fi

echo -e "\n${BLUE}[Step 2/6] Creating group folders in cryoem_blobs...${NC}"

FOLDERS_CREATED=0
FOLDERS_EXISTING=0

while IFS= read -r group; do
    if [[ -z "$group" ]]; then
        continue
    fi

    if [[ -d "$CRYOEM_DIR/$group" ]]; then
        ((FOLDERS_EXISTING++))
    else
        mkdir -p "$CRYOEM_DIR/$group"
        ((FOLDERS_CREATED++))
    fi
done < "$XRAY_GROUPS_FILE"

# Also create RARE_LIGAND folder
if [[ -d "$CRYOEM_DIR/RARE_LIGAND" ]]; then
    echo "RARE_LIGAND folder already exists"
else
    mkdir -p "$CRYOEM_DIR/RARE_LIGAND"
    echo -e "${GREEN}✓ Created RARE_LIGAND folder${NC}"
fi

echo "Group folders status:"
echo "  - Already existing: $FOLDERS_EXISTING"
echo "  - Newly created: $FOLDERS_CREATED"
echo "  - Total: $GROUP_COUNT"

echo -e "\n${BLUE}[Step 3/6] Loading ligand mapping from CSV...${NC}"

# Build ligand -> group mapping from ligand_mapping.csv
# Column 2 = ligand name, Column 3 = group name
TEMP_LIGAND_MAP=$(mktemp)

awk -F',' '
NR == 1 { next }  # Skip header
{
    ligand = $2
    group = $3
    # Trim whitespace and quotes
    gsub(/^[ \t\r\n"]+|[ \t\r\n"]+$/, "", ligand)
    gsub(/^[ \t\r\n"]+|[ \t\r\n"]+$/, "", group)
    if (ligand != "" && group != "") {
        print ligand ":" group
    }
}' "$LIGAND_MAPPING_CSV" > "$TEMP_LIGAND_MAP"

MAPPING_COUNT=$(wc -l < "$TEMP_LIGAND_MAP")
echo "Loaded $MAPPING_COUNT ligand-to-group mappings from CSV"
echo "Sample mappings (first 5):"
head -n 5 "$TEMP_LIGAND_MAP" | sed 's/^/  /'

# Copy mapping to /tmp for access in parallel processes
cp "$TEMP_LIGAND_MAP" /tmp/cryoem_ligand_map.txt

# Also copy group list to /tmp for validation
cp "$XRAY_GROUPS_FILE" /tmp/cryoem_valid_groups.txt

echo -e "\n${BLUE}[Step 4/6] Finding and organizing .npz files...${NC}"

# Find all .npz files in root cryoem_blobs directory
mapfile -t NPZ_FILES < <(find "$CRYOEM_DIR" -maxdepth 1 -type f -name "*.npz")
TOTAL_FILES=${#NPZ_FILES[@]}

echo "Found $TOTAL_FILES .npz files to organize"

if [[ $TOTAL_FILES -eq 0 ]]; then
    echo -e "${YELLOW}⚠ No files to organize${NC}"

    # Still proceed to cleanup empty folders even if no files to move
    MOVED=0
    DUPLICATES=0
    ERRORS=0
    UNMATCHED=0
    NO_FOLDER=0
    RARE_MOVED=0
    RARE_DUPLICATES=0
else

# Function to organize a single file
organize_file() {
    local file="$1"
    local filename=$(basename "$file")

    # Extract ligand name (part after last underscore before .npz)
    # Remove .npz extension first
    local base="${filename%.npz}"

    # Get part after last underscore
    local ligand="${base##*_}"

    # Look up group from mapping
    local group=$(grep "^${ligand}:" /tmp/cryoem_ligand_map.txt 2>/dev/null | cut -d':' -f2)

    if [[ -z "$group" ]]; then
        # No mapping found - will be moved to RARE_LIGAND later
        echo "unmatched:$filename:$ligand:"
        return
    fi

    # Check if group folder exists (is in valid groups list)
    if grep -qx "$group" /tmp/cryoem_valid_groups.txt; then
        local dest="$CRYOEM_DIR/$group/$filename"

        # Check if file already exists at destination
        if [[ -f "$dest" ]]; then
            rm "$file"
            echo "duplicate:$filename:$ligand:$group"
            return
        fi

        if mv "$file" "$dest" 2>/dev/null; then
            echo "moved:$filename:$ligand:$group"
        else
            echo "error:$filename:$ligand:$group"
        fi
    else
        # Group not in valid groups - will be moved to RARE_LIGAND later
        echo "no_folder:$filename:$ligand:$group"
    fi
}

export -f organize_file
export CRYOEM_DIR

# Process files in parallel with xargs
echo "Using xargs with $MAX_WORKERS workers..."
printf '%s\n' "${NPZ_FILES[@]}" | \
    xargs -P "$MAX_WORKERS" -I {} bash -c 'organize_file "$@"' _ {} > /tmp/cryoem_results.txt

wait

# Count results
MOVED=0
DUPLICATES=0
ERRORS=0
UNMATCHED=0
NO_FOLDER=0

while IFS=':' read -r status filename ligand group; do
    case "$status" in
        moved)
            ((MOVED++))
            ;;
        duplicate)
            ((DUPLICATES++))
            ;;
        error)
            ((ERRORS++))
            ;;
        unmatched)
            ((UNMATCHED++))
            ;;
        no_folder)
            ((NO_FOLDER++))
            ;;
    esac
done < /tmp/cryoem_results.txt

echo -e "${GREEN}✓ File organization complete${NC}"
echo "  - Files moved to groups: $MOVED"
echo "  - Duplicates removed: $DUPLICATES"
echo "  - Errors: $ERRORS"
echo "  - No mapping found: $UNMATCHED"
echo "  - Mapping exists but no folder: $NO_FOLDER"

echo -e "\n${BLUE}[Step 5/6] Moving remaining files to RARE_LIGAND...${NC}"

# Find all remaining .npz files in root directory
mapfile -t REMAINING_FILES < <(find "$CRYOEM_DIR" -maxdepth 1 -type f -name "*.npz")
REMAINING_COUNT=${#REMAINING_FILES[@]}

echo "Found $REMAINING_COUNT files remaining in root directory"

if [[ $REMAINING_COUNT -gt 0 ]]; then
    RARE_MOVED=0
    RARE_DUPLICATES=0

    for file in "${REMAINING_FILES[@]}"; do
        filename=$(basename "$file")
        dest="$CRYOEM_DIR/RARE_LIGAND/$filename"

        if [[ -f "$dest" ]]; then
            rm "$file"
            ((RARE_DUPLICATES++))
        elif mv "$file" "$dest" 2>/dev/null; then
            ((RARE_MOVED++))
        fi
    done

    echo -e "${GREEN}✓ Moved $RARE_MOVED files to RARE_LIGAND${NC}"
    if [[ $RARE_DUPLICATES -gt 0 ]]; then
        echo "  - Duplicates removed: $RARE_DUPLICATES"
    fi
else
    echo "No files to move to RARE_LIGAND"
fi

fi  # End of if block for TOTAL_FILES check

echo -e "\n${BLUE}[Step 6/6] Cleaning up empty folders...${NC}"

EMPTY_FOLDERS=0

# Check each group folder (not RARE_LIGAND)
while IFS= read -r group; do
    if [[ -z "$group" ]]; then
        continue
    fi

    folder="$CRYOEM_DIR/$group"

    # Check if folder exists and is empty
    if [[ -d "$folder" ]]; then
        file_count=$(find "$folder" -maxdepth 1 -type f -name "*.npz" | wc -l)

        if [[ $file_count -eq 0 ]]; then
            rmdir "$folder" 2>/dev/null && ((EMPTY_FOLDERS++))
        fi
    fi
done < "$XRAY_GROUPS_FILE"

if [[ $EMPTY_FOLDERS -gt 0 ]]; then
    echo -e "${GREEN}✓ Removed $EMPTY_FOLDERS empty folders${NC}"
else
    echo "No empty folders to remove"
fi

# Final summary
echo -e "\n=========================================="
echo -e "${GREEN}✓ CryoEM organization complete!${NC}"
echo ""
echo "Statistics:"
echo "  - Files organized into groups: $MOVED"
echo "  - Files moved to RARE_LIGAND: $RARE_MOVED"
echo "  - Duplicates removed: $((DUPLICATES + RARE_DUPLICATES))"
echo "  - Errors: $ERRORS"
echo "  - Empty folders removed: $EMPTY_FOLDERS"
echo ""
echo "Location: $(realpath "$CRYOEM_DIR")"
echo "=========================================="

# Cleanup
rm -f "$TEMP_LIGAND_MAP"
rm -f /tmp/cryoem_ligand_map.txt /tmp/cryoem_valid_groups.txt
rm -f /tmp/cryoem_results.txt

exit 0