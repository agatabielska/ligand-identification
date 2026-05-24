from pathlib import Path
import sys


thresholded = Path("data/cryoem_blobs")
unthresholded = Path("data/NoThresholdBlobs")

if not thresholded.exists():
	print(f"Missing directory: {thresholded}")
	sys.exit(1)
if not unthresholded.exists():
	print(f"Missing directory: {unthresholded}")
	sys.exit(1)

thresholded_files = sorted(thresholded.glob("**/*.npz"))
unthresholded_files = sorted(unthresholded.glob("**/*.npz"))

th_set = {p.stem for p in thresholded_files}
unth_set = {p.stem for p in unthresholded_files}

print(f"thresholded: {len(th_set)} files")
print(f"unthresholded: {len(unth_set)} files")

missing = th_set - unth_set
if not missing:
    print("All thresholded files are present in unthresholded (by stem).")
    # Remove from unthresholded
    to_remove = unth_set - th_set
    for file in unthresholded_files:
        if file.stem in to_remove:
            file.unlink()
else:
	print(f"Missing {len(missing)} files in unthresholded:")
	for name in sorted(missing):
		print(name)
  
# check number of files in each directory
thresholded_files = sorted(thresholded.glob("**/*.npz"))
unthresholded_files = sorted(unthresholded.glob("**/*.npz"))
print(f"After filtering, thresholded: {len(thresholded_files)} files")
print(f"After filtering, unthresholded: {len(unthresholded_files)} files")