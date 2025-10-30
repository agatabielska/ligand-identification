#!/bin/bash

# Ligand File Organizer by Cluster (Optimized)
# Organizes .npz files into folders based on ligand_mapping.csv

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/"

# Configuration
MAPPING_CSV="${SCRIPT_DIR}../../data/ligand_mapping.csv"
SOURCE_DIR="${SCRIPT_DIR}../../data/cryoem_blobs"
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

# Extract unique ligands from filenames
declare -A REQUIRED_LIGANDS
for file in "${NPZ_FILES[@]}"; do
    filename=$(basename "$file")
    ligand="${filename%.npz}"
    ligand="${ligand##*_}"
    REQUIRED_LIGANDS["$ligand"]=1
done

UNIQUE_LIGANDS=${#REQUIRED_LIGANDS[@]}
echo "Identified $UNIQUE_LIGANDS unique ligands to lookup"

# Save required ligands to temp file for awk filtering
> /tmp/required_ligands.txt
for ligand in "${!REQUIRED_LIGANDS[@]}"; do
    echo "$ligand" >> /tmp/required_ligands.txt
done

echo -e "\n${BLUE}[3/5] Loading relevant mappings from CSV...${NC}"

# Use awk to filter CSV for only required ligands (much faster than bash loops)
# This loads only needed rows instead of all 30k entries
awk -F',' 'NR==1 {next}  # Skip header
FNR==NR {ligands[$1]=1; next}  # Build lookup table from required_ligands.txt
{
    ligand=$2
    gsub(/^[ \t\r\n"]+|[ \t\r\n"]+$/, "", ligand)  # Trim whitespace and quotes
    if (ligand in ligands) {
        folder=$3
        gsub(/^[ \t\r\n"]+|[ \t\r\n"]+$/, "", folder)  # Trim whitespace and quotes
        print ligand ":" folder
    }
}' /tmp/required_ligands.txt "$MAPPING_CSV" > /tmp/ligand_mapping.txt

# Load filtered mapping into associative array
declare -A LIGAND_TO_FOLDER
while IFS=':' read -r ligand folder; do
    LIGAND_TO_FOLDER["$ligand"]="$folder"
done < /tmp/ligand_mapping.txt

MAPPING_COUNT=${#LIGAND_TO_FOLDER[@]}
echo "Loaded $MAPPING_COUNT ligand-to-folder mappings (filtered from CSV)"

if [[ $MAPPING_COUNT -eq 0 ]]; then
    echo -e "${RED}❌ Error: No matching ligands found in CSV${NC}"
    exit 1
fi

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

# Function to organize a single file
organize_file() {
    local file="$1"
    local filename=$(basename "$file")
    
    # Extract ligand name (last 3 characters before .npz)
    # Format: 3J9L_C_1301_DTP.npz -> DTP
    local ligand="${filename%.npz}"
    ligand="${ligand##*_}"
    
    # Check if ligand exists in mapping
    if [[ -n "${LIGAND_TO_FOLDER[$ligand]}" ]]; then
        local folder="${LIGAND_TO_FOLDER[$ligand]}"
        local safe_folder=$(echo "$folder" | tr '/' '_' | tr '\\' '_')
        local dest="$SOURCE_DIR/$safe_folder/$filename"
        
        # Try hard link first (instant, no space), fallback to move
        if ln "$file" "$dest" 2>/dev/null; then
            rm "$file"  # Remove original after hard link
            echo "linked:$filename"
        else
            mv "$file" "$dest" 2>/dev/null && echo "moved:$filename" || echo "error:$filename"
        fi
    else
        echo "unmatched:$filename:$ligand"
    fi
}

export -f organize_file
export SOURCE_DIR
export -A LIGAND_TO_FOLDER

# Process files in parallel
SUCCESS=0
LINKED=0
MOVED=0
ERRORS=0
UNMATCHED=0

if command -v parallel &> /dev/null; then
    # Use GNU parallel if available (faster with progress bar)
    echo "Using GNU parallel with $MAX_WORKERS workers..."
    
    printf '%s\n' "${NPZ_FILES[@]}" | \
        parallel -j "$MAX_WORKERS" --bar organize_file {} > /tmp/organize_results.txt
else
    # Fallback to xargs
    echo "Using xargs with $MAX_WORKERS workers..."
    echo -e "${YELLOW}Tip: Install GNU parallel for progress tracking: sudo apt-get install parallel${NC}"
    
    printf '%s\n' "${NPZ_FILES[@]}" | \
        xargs -P "$MAX_WORKERS" -I {} bash -c 'organize_file "$@"' _ {} > /tmp/organize_results.txt
fi

# Process results
> /tmp/unmatched_files.log
> /tmp/organization_errors.log

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
        error)
            ((ERRORS++))
            echo "$filename" >> /tmp/organization_errors.log
            ;;
        unmatched)
            ((UNMATCHED++))
            echo "$filename (extracted ligand: $ligand)" >> /tmp/unmatched_files.log
            ;;
    esac
done < /tmp/organize_results.txt

echo -e "\n=========================================="
echo -e "${GREEN}✓ Successfully organized $SUCCESS files${NC}"
echo "  - Hard linked: $LINKED"
echo "  - Moved: $MOVED"

if [[ $ERRORS -gt 0 ]]; then
    echo -e "\n${YELLOW}⚠ Encountered $ERRORS errors${NC}"
    echo "  Error details saved to: /tmp/organization_errors.log"
fi

if [[ $UNMATCHED -gt 0 ]]; then
    echo -e "\n${YELLOW}⚠ $UNMATCHED unmatched files${NC}"
    echo "  Unmatched files saved to: /tmp/unmatched_files.log"
fi

echo -e "\n${GREEN}✓ Organization complete!${NC}"
echo "  Source directory: $(realpath "$SOURCE_DIR")"
echo "  Memory optimization: Loaded only $MAPPING_COUNT/$((30000)) CSV entries"
echo "=========================================="

# Cleanup
rm -f /tmp/organize_results.txt /tmp/ligand_mapping.txt /tmp/required_ligands.txt