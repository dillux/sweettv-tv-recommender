"""Quick look at the SWEET.TV files: headers, first rows, value ranges."""
import csv, sys, os, itertools

BASE = sys.argv[1] if len(sys.argv) > 1 else "."
FILES = [
    "dataset11-30.csv",
    "export_arh_11-20-final.csv",
    "export_arh_21-30-final.csv",
    "export_arh_31-42-final.csv",
    "submission.csv",
]

for name in FILES:
    path = os.path.join(BASE, name)
    if not os.path.exists(path):
        print(f"\n=== {name}: NOT FOUND at {path}")
        continue
    size = os.path.getsize(path) / 1e6
    print(f"\n=== {name}  ({size:.0f} MB)")
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        r = csv.reader(f)
        try:
            header = next(r)
        except StopIteration:
            print("  empty"); continue
        print("  header:", header)
        for i, row in enumerate(itertools.islice(r, 3)):
            print(f"  row{i}:", row[:14])
print("\ndone")
