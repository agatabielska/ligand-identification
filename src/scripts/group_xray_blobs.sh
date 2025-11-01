#!/usr/bin/bash

# Ligand File Organizer by Cluster (Optimized)
# Organizes .npz files into folders based on ligand_mapping.csv
# Creates individual folders for unmatched ligands



# Get script directory
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/"

# Configuration
MAPPING_CSV="${SCRIPT_DIR}../../data/ligand_mapping.csv"
REQUIRED_LIGANDS="${SCRIPT_DIR}../../data/required_ligands_xray.txt"
SOURCE_DIR="${SCRIPT_DIR}../../data/xray_blobs"
MAX_WORKERS="${1:-16}"  # Default 16 workers, can override with first argument

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Ligand File Organizer by Cluster"
echo "=========================================="
echo "Script directory: $SCRIPT_DIR"

# Change to script directory
cd "$SCRIPT_DIR"

# Validate inputs
if [[ ! -f "$MAPPING_CSV" ]]; then
    echo -e "${RED}❌ Error: Mapping CSV not found: $MAPPING_CSV${NC}"
    exit 1
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
    echo -e "${RED}❌ Error: Source directory not found: $SOURCE_DIR${NC}"
    exit 1
fi

echo -e "\n${BLUE}[1/5] Scanning .npz files...${NC}"

# Find all .npz files FIRST (before loading CSV)
mapfile -t NPZ_FILES < <(find "$SOURCE_DIR" -maxdepth 1 -type f -name "*.npz")
TOTAL_FILES=${#NPZ_FILES[@]}

if [[ $TOTAL_FILES -eq 0 ]]; then
    echo -e "${YELLOW}⚠ No .npz files found to organize${NC}"
    exit 0
fi

echo "Found $TOTAL_FILES .npz files to organize"

echo -e "\n${BLUE}[2/5] Extracting required ligands from filenames...${NC}"

# Check if required_ligands_xray.txt exists
if [[ -f "$REQUIRED_LIGANDS" ]]; then
    echo "Loading required ligands from $REQUIRED_LIGANDS"
else
    # Extract unique ligands from filenames using parallel processing
    TEMP_LIGANDS=$(mktemp)

    extract_ligand() {
        local file="$1"
        local filename=$(basename "$file")
        local ligand="${filename%.npz}"
        IFS='_' read -r first second rest <<< "$ligand"
        echo "$second"
    }

    export -f extract_ligand


        echo "Using xargs for extraction..."
        printf '%s\n' "${NPZ_FILES[@]}" | \
            xargs -P "$MAX_WORKERS" -I {} bash -c 'extract_ligand "$@"' _ {} | \
            sort -u > "$TEMP_LIGANDS"


    cp "$TEMP_LIGANDS" "$REQUIRED_LIGANDS"
    UNIQUE_LIGANDS=$(wc -l < "$REQUIRED_LIGANDS")
    rm -f "$TEMP_LIGANDS"

    echo "Identified $UNIQUE_LIGANDS unique ligands to lookup"
fi
echo -e "\n${BLUE}[3/5] Loading relevant mappings from CSV...${NC}"

# Use awk to filter CSV for only required ligands (much faster than bash loops)
# This loads only needed rows instead of all 30k entries
awk -F',' '
BEGIN {
    # Read required ligands file first
    while ((getline line < "'"$REQUIRED_LIGANDS"'") > 0) {
        ligands[line] = 1
    }
    close("'"$REQUIRED_LIGANDS"'")
}
NR == 1 { next }  # Skip CSV header
{
    ligand = $2
    gsub(/^[ \t\r\n"]+|[ \t\r\n"]+$/, "", ligand)  # Trim whitespace and quotes
    if (ligand in ligands) {
        folder = $3
        gsub(/^[ \t\r\n"]+|[ \t\r\n"]+$/, "", folder)  # Trim whitespace and quotes
        print ligand ":" folder
    }
}' "$MAPPING_CSV" > /tmp/ligand_mapping_xray.txt

# Load filtered mapping into associative array
declare -A LIGAND_TO_FOLDER
while IFS=':' read -r ligand folder; do
    LIGAND_TO_FOLDER["$ligand"]="$folder"
done < /tmp/ligand_mapping_xray.txt

MAPPING_COUNT=${#LIGAND_TO_FOLDER[@]}
echo "Loaded $MAPPING_COUNT ligand-to-folder mappings (filtered from CSV)"

echo -e "\n${BLUE}[4/5] Creating cluster folders...${NC}"

# Get unique folders from filtered mapping
declare -A UNIQUE_FOLDERS_MAP
for folder in "${LIGAND_TO_FOLDER[@]}"; do
    UNIQUE_FOLDERS_MAP["$folder"]=1
done

for folder in "${!UNIQUE_FOLDERS_MAP[@]}"; do
    if [[ -n "$folder" ]]; then
        # Sanitize folder name
        safe_folder=$(echo "$folder" | tr '/' '_' | tr '\\' '_')
        mkdir -p "$SOURCE_DIR/$safe_folder"
    fi
done

echo "Created ${#UNIQUE_FOLDERS_MAP[@]} cluster folders"

echo -e "\n${BLUE}[5/5] Organizing files into clusters...${NC}"

# RE-SCAN for files that still exist in root directory
echo "Re-scanning for remaining .npz files in root directory..."
mapfile -t NPZ_FILES < <(find "$SOURCE_DIR" -maxdepth 1 -type f -name "*.npz")
REMAINING_FILES=${#NPZ_FILES[@]}

if [[ $REMAINING_FILES -eq 0 ]]; then
    echo -e "${GREEN}✓ No files to organize - all files already in correct folders!${NC}"
    echo "=========================================="
    exit 0
fi

echo "Found $REMAINING_FILES files remaining to organize (down from $TOTAL_FILES)"

# Function to organize a single file
organize_file() {
    local file="$1"
    local filename=$(basename "$file")

    # Extract ligand name (second element when split by '_')
    local ligand="${filename%.npz}"
    IFS='_' read -r first second rest <<< "$ligand"
    ligand="$second"

    # Look up folder from file instead of array
    local folder=$(grep "^${ligand}:" /tmp/ligand_to_folder_map_xray.txt 2>/dev/null | cut -d':' -f2)

    if [[ -n "$folder" ]]; then
        local safe_folder=$(echo "$folder" | tr '/' '_' | tr '\\' '_')
        local dest="$SOURCE_DIR/$safe_folder/$filename"

        # Check if file already exists at destination
        if [[ -f "$dest" ]]; then
            rm "$file"
            echo "duplicate_removed:$filename"
            return
        fi

        if ln "$file" "$dest" 2>/dev/null; then
            rm "$file"
            echo "linked:$filename"
        else
            mv "$file" "$dest" 2>/dev/null && echo "moved:$filename" || echo "error:$filename"
        fi
    else
        # Create individual folder for unmatched ligand
        local safe_ligand=$(echo "$ligand" | tr '/' '_' | tr '\\' '_')
        mkdir -p "$SOURCE_DIR/$safe_ligand"
        local dest="$SOURCE_DIR/$safe_ligand/$filename"

        # Check if file already exists at destination
        if [[ -f "$dest" ]]; then
            rm "$file"
            echo "duplicate_removed:$filename:$ligand"
            return
        fi

        if ln "$file" "$dest" 2>/dev/null; then
            rm "$file"
            echo "unmatched_linked:$filename:$ligand"
        else
            mv "$file" "$dest" 2>/dev/null && echo "unmatched_moved:$filename:$ligand" || echo "error:$filename"
        fi
    fi
}

> /tmp/ligand_to_folder_map_xray.txt
for ligand in "${!LIGAND_TO_FOLDER[@]}"; do
    echo "$ligand:${LIGAND_TO_FOLDER[$ligand]}" >> /tmp/ligand_to_folder_map_xray.txt
done

export -f organize_file
export SOURCE_DIR

# Process files in parallel
SUCCESS=0
LINKED=0
MOVED=0
ERRORS=0
UNMATCHED=0
DUPLICATES=0


    # Fallback to xargs
    echo "Using xargs with $MAX_WORKERS workers..."
    echo -e "${YELLOW}Tip: Install GNU parallel for progress tracking: sudo apt-get install parallel${NC}"

    printf '%s\n' "${NPZ_FILES[@]}" | \
        xargs -P "$MAX_WORKERS" -I {} bash -c 'organize_file "$@"' _ {} > /tmp/organize_results_xray.txt


# Process results
> /tmp/unmatched_files_xray.log
> /tmp/organization_errors_xray.log
> /tmp/duplicate_files_xray.log

while IFS=':' read -r status filename ligand; do
    case "$status" in
        linked)
            ((SUCCESS++))
            ((LINKED++))
            ;;
        moved)
            ((SUCCESS++))
            ((MOVED++))
            ;;
        unmatched_linked)
            ((SUCCESS++))
            ((UNMATCHED++))
            echo "$filename (ligand: $ligand) -> created folder: $ligand/" >> /tmp/unmatched_files_xray.log
            ;;
        unmatched_moved)
            ((SUCCESS++))
            ((UNMATCHED++))
            echo "$filename (ligand: $ligand) -> created folder: $ligand/" >> /tmp/unmatched_files_xray.log
            ;;
        duplicate_removed)
            ((SUCCESS++))
            ((DUPLICATES++))
            echo "$filename (already existed at destination)" >> /tmp/duplicate_files_xray.log
            ;;
        error)
            ((ERRORS++))
            echo "$filename" >> /tmp/organization_errors_xray.log
            ;;
    esac
done < /tmp/organize_results_xray.txt

echo -e "\n=========================================="
echo -e "${GREEN}✓ Successfully organized $SUCCESS files${NC}"
echo "  - Matched and linked: $LINKED"
echo "  - Matched and moved: $MOVED"
echo "  - Unmatched (individual folders): $UNMATCHED"
echo "  - Duplicates removed: $DUPLICATES"

if [[ $ERRORS -gt 0 ]]; then
    echo -e "\n${YELLOW}⚠ Encountered $ERRORS errors${NC}"
    echo "  Error details saved to: /tmp/organization_errors_xray.log"
fi

if [[ $DUPLICATES -gt 0 ]]; then
    echo -e "\n${BLUE}ℹ $DUPLICATES duplicate files removed${NC}"
    echo "  Duplicate files log saved to: /tmp/duplicate_files_xray.log"
fi

if [[ $UNMATCHED -gt 0 ]]; then
    echo -e "\n${BLUE}ℹ $UNMATCHED files placed in individual ligand folders${NC}"
    echo "  Unmatched files log saved to: /tmp/unmatched_files_xray.log"
fi

echo -e "\n${GREEN}✓ Organization complete!${NC}"
echo "  Source directory: $(realpath "$SOURCE_DIR")"
echo "  Memory optimization: Loaded only $MAPPING_COUNT CSV entries"
echo "=========================================="

# Cleanup
rm -f /tmp/organize_results_xray.txt /tmp/ligand_mapping_xray.txt /tmp/ligand_to_folder_map_xray.txt