#!/usr/bin/bash

# Filter CryoEM Classes Script
# Hardlinks frequent ligand folders into filtered_cryoem_classes/
# Merges all rare ligands into filtered_cryoem_classes/rare/

# Get script directory
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/"

# Configuration
CRYOEM_DIR="${SCRIPT_DIR}../../data/cryoem_blobs"
LIGAND_GROUPS="${SCRIPT_DIR}../../data/ligand_groups.txt"
OUTPUT_DIR="${SCRIPT_DIR}../../data/filtered_cryoem_classes"
RARE_DIR="${OUTPUT_DIR}/rare"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================="
echo "Filter CryoEM Classes Script"
echo "=========================================="
echo "Source:        $CRYOEM_DIR"
echo "Ligand groups: $LIGAND_GROUPS"
echo "Output:        $OUTPUT_DIR"
echo "=========================================="

# Validate
if [[ ! -d "$CRYOEM_DIR" ]]; then
    echo -e "${RED}❌ Error: CryoEM directory not found: $CRYOEM_DIR${NC}"
    exit 1
fi
if [[ ! -f "$LIGAND_GROUPS" ]]; then
    echo -e "${RED}❌ Error: ligand_groups.txt not found: $LIGAND_GROUPS${NC}"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$RARE_DIR"

read FREQUENT_COUNT RARE_LIGAND_COUNT RARE_FILE_COUNT FREQUENT_FILE_COUNT < <(python3 - "$CRYOEM_DIR" "$OUTPUT_DIR" "$RARE_DIR" "$LIGAND_GROUPS" << 'EOF'
import os, sys
from pathlib import Path

cryoem_dir    = Path(sys.argv[1])
output_dir    = Path(sys.argv[2])
rare_dir      = Path(sys.argv[3])
ligand_groups = Path(sys.argv[4])

# Load the frequent ligand names into a set
frequent_ligands = set(ligand_groups.read_text().splitlines())
frequent_ligands.discard("")  # remove any blank lines

frequent_dirs  = 0
rare_ligands   = 0
rare_files     = 0
frequent_files = 0

for ligand_dir in sorted(cryoem_dir.iterdir()):
    if not ligand_dir.is_dir():
        continue

    npz_files = list(ligand_dir.glob("*.npz"))
    count = len(npz_files)

    if count == 0:
        continue

    if ligand_dir.name in frequent_ligands:
        dest = output_dir / ligand_dir.name
        dest.mkdir(exist_ok=True)
        for f in npz_files:
            dest_f = dest / f.name
            if dest_f.exists():
                dest_f.unlink()
            os.link(f, dest_f)
        frequent_dirs  += 1
        frequent_files += count
    else:
        for f in npz_files:
            dest_f = rare_dir / f.name
            if dest_f.exists():
                dest_f.unlink()
            os.link(f, dest_f)
        rare_ligands += 1
        rare_files   += count

print(frequent_dirs, rare_ligands, rare_files, frequent_files)
EOF
)

echo ""
echo -e "${GREEN}✓ Done!${NC}"
echo ""
echo "Statistics:"
echo "  Frequent ligands: $FREQUENT_COUNT  ($FREQUENT_FILE_COUNT files)"
echo "  Rare ligands:     $RARE_LIGAND_COUNT  ($RARE_FILE_COUNT files → merged into rare/)"
echo ""
echo "Output:"
echo "  $OUTPUT_DIR/"
echo "    ├── <ligand>/   × $FREQUENT_COUNT frequent classes"
echo "    └── rare/       ($RARE_FILE_COUNT files from $RARE_LIGAND_COUNT ligands)"
echo "=========================================="