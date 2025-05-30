#!/usr/bin/env python3
import sys
from pathlib import Path
import argparse
import pandas as pd

def find_theme_dirs(base_dir: Path, prefix: str):
    return [p for p in base_dir.iterdir()
            if p.is_dir() and p.name.startswith(prefix)]

def merge_csvs(theme_dirs, prefix, itn_name, grades_name, results_name):
    itn_list, grades_list, results_list = [], [], []
    for d in theme_dirs:
        itn     = pd.read_csv(d / itn_name)
        grades  = pd.read_csv(d / grades_name)
        results = pd.read_csv(d / results_name)

        tema_letter = d.name.replace(prefix, "").upper()
        results["tema"] = tema_letter

        itn_list.append(itn)
        grades_list.append(grades)
        results_list.append(results)

    itn_all     = pd.concat(itn_list,    ignore_index=True)
    grades_all  = pd.concat(grades_list, ignore_index=True)
    results_all = pd.concat(results_list,ignore_index=True)
    return itn_all, grades_all, results_all

def sanity_checks(itn_all, grades_all):
    grades_all["image"] = grades_all["file"].str.replace(r"\.png$", "", regex=True)

    n_itn, n_itn_u = len(itn_all), itn_all["image"].nunique()
    n_gr,  n_gr_u  = len(grades_all), grades_all["image"].nunique()
    dup_itn = itn_all[itn_all.duplicated(subset=["image"], keep=False)]["image"].unique()
    dup_gr  = grades_all[grades_all.duplicated(subset=["image"], keep=False)]["image"].unique()
    set_itn = set(itn_all["image"])
    set_gr  = set(grades_all["image"])
    only_in_grades = sorted(set_gr - set_itn)
    only_in_itn    = sorted(set_itn - set_gr)

    print("=== Sanity checks ===")
    print(f"image-to-name_all.csv: {n_itn} rows ({n_itn_u} unique images)")
    if dup_itn.size:
        print(f"  ↳ DUPLICATES in image-to-name_all.csv: {list(dup_itn)}")
    print(f"grades_all.csv      : {n_gr} rows ({n_gr_u} unique images)")
    if dup_gr.size:
        print(f"  ↳ DUPLICATES in grades_all.csv: {list(dup_gr)}")
    print(f"Matched keys        : {len(set_itn & set_gr)}")
    if only_in_grades:
        print(f"  ↳ Only in grades : {only_in_grades}")
    if only_in_itn:
        print(f"  ↳ Only in names  : {only_in_itn}")
    print("=====================\n")

def main():
    parser = argparse.ArgumentParser(
        description="Merge per-tema CSVs into unified tables and run sanity checks."
    )
    parser.add_argument("base_dir", nargs="?", default=".",
                        help="Base directory containing tema* subdirectories")
    parser.add_argument("--prefix", default="tema",
                        help="Directory name prefix for themes (default: tema)")
    parser.add_argument("--itn", default="image-to-name.csv",
                        help="Filename for image-to-name CSV (default: image-to-name.csv)")
    parser.add_argument("--grades", default="grades.csv",
                        help="Filename for grades CSV (default: grades.csv)")
    parser.add_argument("--results", default="results.csv",
                        help="Filename for results CSV (default: results.csv)")
    parser.add_argument("--out-prefix", default="",
                        help="Prefix for output filenames (e.g. 'all_').")
    parser.add_argument("--out-dir", default=".",
                        help="Output directory (default: current directory)")
    args = parser.parse_args()

    base = Path(args.base_dir)
    theme_dirs = find_theme_dirs(base, args.prefix)
    if not theme_dirs:
        print(f"No directories starting with '{args.prefix}' found in {base}", 
              file=sys.stderr)
        sys.exit(1)

    itn_all, grades_all, results_all = merge_csvs(
        theme_dirs, args.prefix, args.itn, args.grades, args.results
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.out_prefix
    itn_all.to_csv(out_dir / f"{prefix}image-to-name_all.csv", index=False)
    grades_all.to_csv(out_dir / f"{prefix}grades_all.csv",       index=False)
    results_all.to_csv(out_dir / f"{prefix}results_all.csv",     index=False)
    
    sanity_checks(itn_all, grades_all)

    merge_df = pd.merge(
        itn_all,
        grades_all[["image", "grade"]],
        on="image",
        how="left",
        validate="one_to_one"
    )
    merge_df.to_csv(out_dir / f"{prefix}grades_with_names.csv", index=False)

    print("✔ Created in", out_dir.resolve())
    print(f"  • {prefix}image-to-name_all.csv")
    print(f"  • {prefix}grades_all.csv")
    print(f"  • {prefix}results_all.csv (with 'tema')")
    print(f"  • {prefix}grades_with_names.csv")

if __name__ == "__main__":
    main()

