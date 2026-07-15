#!/usr/bin/bash
# X-ray Ligand File Organizer
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/"

# Paths
DATA_DIR="${SCRIPT_DIR}../../data"
TRAIN_CSV="${DATA_DIR}/xray_train.csv"
HOLDOUT_CSV="${DATA_DIR}/xray_holdout.csv"
SOURCE_DIR="${DATA_DIR}/xray_blobs"
TRAIN_DIR="${SOURCE_DIR}/xray_train"
HOLDOUT_DIR="${SOURCE_DIR}/xray_holdout"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "X-ray Ligand Organizer"
echo "======================"

# Validate
if [[ ! -d "$SOURCE_DIR" ]]; then
    echo -e "${RED}Error: Source directory not found: $SOURCE_DIR${NC}"
    exit 1
fi

if [[ ! -f "$TRAIN_CSV" ]]; then
    echo -e "${RED}Error: Training CSV not found: $TRAIN_CSV${NC}"
    exit 1
fi

if [[ ! -f "$HOLDOUT_CSV" ]]; then
    echo -e "${RED}Error: Holdout CSV not found: $HOLDOUT_CSV${NC}"
    exit 1
fi

# Create directories
mkdir -p "$TRAIN_DIR" "$HOLDOUT_DIR"

# Process training set
echo -e "\n${BLUE}Processing xray_train.csv -> xray_train/${NC}"

tail -n +2 "$TRAIN_CSV" | xargs -P 16 -I {} bash -c '
    IFS="," read -r filename ligand_group <<< "{}"
    
    # Clean up fields
    filename=$(echo "$filename" | sed "s/^[ \t\r\n\"]*//;s/[ \t\r\n\"]*$//")
    ligand_group=$(echo "$ligand_group" | sed "s/^[ \t\r\n\"]*//;s/[ \t\r\n\"]*$//")
    
    # Skip empty lines
    [[ -z "$filename" || -z "$ligand_group" ]] && exit 0
    
    # Full path to source file
    SOURCE_FILE="'"$SOURCE_DIR"'/${filename}"
    
    # Check if file exists
    [[ ! -f "$SOURCE_FILE" ]] && exit 0
    
    # Sanitize group name for folder
    safe_group=$(echo "$ligand_group" | tr "/" "_" | tr "\\\\" "_")
    
    # Create destination folder if needed
    GROUP_DIR="'"$TRAIN_DIR"'/${safe_group}"
    mkdir -p "$GROUP_DIR"
    
    # Move file
    DEST_FILE="${GROUP_DIR}/${filename}"
    mv "$SOURCE_FILE" "$DEST_FILE" 2>/dev/null
'

echo -e "${GREEN}✓ Training set complete${NC}"

# Process holdout set
echo -e "\n${BLUE}Processing xray_holdout.csv -> xray_holdout/${NC}"

tail -n +2 "$HOLDOUT_CSV" | xargs -P 16 -I {} bash -c '
    IFS="," read -r filename ligand_group <<< "{}"
    
    # Clean up fields
    filename=$(echo "$filename" | sed "s/^[ \t\r\n\"]*//;s/[ \t\r\n\"]*$//")
    ligand_group=$(echo "$ligand_group" | sed "s/^[ \t\r\n\"]*//;s/[ \t\r\n\"]*$//")
    
    # Skip empty lines
    [[ -z "$filename" || -z "$ligand_group" ]] && exit 0
    
    # Full path to source file
    SOURCE_FILE="'"$SOURCE_DIR"'/${filename}"
    
    # Check if file exists
    [[ ! -f "$SOURCE_FILE" ]] && exit 0
    
    # Sanitize group name for folder
    safe_group=$(echo "$ligand_group" | tr "/" "_" | tr "\\\\" "_")
    
    # Create destination folder if needed
    GROUP_DIR="'"$HOLDOUT_DIR"'/${safe_group}"
    mkdir -p "$GROUP_DIR"
    
    # Move file
    DEST_FILE="${GROUP_DIR}/${filename}"
    mv "$SOURCE_FILE" "$DEST_FILE" 2>/dev/null
'

echo -e "${GREEN}✓ Holdout set complete${NC}"

echo -e "\n${GREEN}✓ Complete${NC}"
echo "======================"
exit 0