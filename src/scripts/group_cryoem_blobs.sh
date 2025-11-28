#!/usr/bin/bash
# CryoEM Ligand File Organizer by Fold
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/"

# Here define which folds go to training and which to holdout
TRAIN_FOLDS=(0 1)
HOLDOUT_FOLDS=(2)

# Paths
DATA_DIR="${SCRIPT_DIR}../../data"
SOURCE_DIR="${SCRIPT_DIR}../../data/cryoem_blobs"
TRAIN_DIR="${SOURCE_DIR}/train"
HOLDOUT_DIR="${SOURCE_DIR}/holdout"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "CryoEM Ligand Organizer"
echo "======================="

# Validate
if [[ ! -d "$SOURCE_DIR" ]]; then
    echo -e "${RED}Error: Source directory not found: $SOURCE_DIR${NC}"
    exit 1
fi

for fold in 0 1 2; do
    if [[ ! -f "${DATA_DIR}/fold${fold}.csv" ]]; then
        echo -e "${RED}Error: CSV not found: ${DATA_DIR}/fold${fold}.csv${NC}"
        exit 1
    fi
done

# Create directories
mkdir -p "$TRAIN_DIR" "$HOLDOUT_DIR"

# Process each fold
for fold in 0 1 2; do
    CSV_FILE="${DATA_DIR}/fold${fold}.csv"
    
    # Determine destination
    if [[ " ${TRAIN_FOLDS[@]} " =~ " ${fold} " ]]; then
        DEST_DIR="$TRAIN_DIR"
        DEST_TYPE="train"
    elif [[ " ${HOLDOUT_FOLDS[@]} " =~ " ${fold} " ]]; then
        DEST_DIR="$HOLDOUT_DIR"
        DEST_TYPE="holdout"
    else
        echo -e "${RED}Error: Fold ${fold} not assigned${NC}"
        exit 1
    fi
    
    echo -e "\n${BLUE}Processing fold${fold}.csv -> ${DEST_TYPE}/${NC}"
    
    # Process in parallel with xargs
    tail -n +2 "$CSV_FILE" | xargs -P 16 -I {} bash -c '
        IFS="," read -r ligand_group filename <<< "{}"
        
        # Clean up fields
        ligand_group=$(echo "$ligand_group" | sed "s/^[ \t\r\n\"]*//;s/[ \t\r\n\"]*$//")
        filename=$(echo "$filename" | sed "s/^[ \t\r\n\"]*//;s/[ \t\r\n\"]*$//")
        
        # Skip empty lines
        [[ -z "$filename" || -z "$ligand_group" ]] && exit 0
        
        # Full path to source file
        SOURCE_FILE="'"$SOURCE_DIR"'/${filename}"
        
        # Check if file exists
        [[ ! -f "$SOURCE_FILE" ]] && exit 0
        
        # Sanitize group name for folder
        safe_group=$(echo "$ligand_group" | tr "/" "_" | tr "\\\\" "_")
        
        # Create destination folder if needed
        GROUP_DIR="'"$DEST_DIR"'/${safe_group}"
        mkdir -p "$GROUP_DIR"
        
        # Move file
        DEST_FILE="${GROUP_DIR}/${filename}"
        mv "$SOURCE_FILE" "$DEST_FILE" 2>/dev/null
    '
    
    echo -e "${GREEN}✓ Fold ${fold} complete${NC}"
done

echo -e "\n${GREEN}✓ Complete${NC}"
echo "======================="
exit 0