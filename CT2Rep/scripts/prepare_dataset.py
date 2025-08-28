import os
import sys
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
import pdfplumber


def read_dicom_series_to_volume(dicom_dir: Path) -> np.ndarray:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in {dicom_dir}")
    # use the first series id by default
    series_files = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
    reader.SetFileNames(series_files)
    image = reader.Execute()
    volume = sitk.GetArrayFromImage(image)  # shape: [D, H, W]
    # convert to [H, W, D]
    volume = np.transpose(volume, (1, 2, 0)).astype(np.float32)
    return volume


def extract_text_from_pdf(pdf_path: Path) -> str:
    text_parts = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            text_parts.append(txt)
    text = "\n".join(text_parts)
    # fallback: if empty, try loading raw
    if not text.strip():
        try:
            with open(pdf_path, 'rb') as f:
                raw = f.read().decode(errors='ignore')
                text = raw
        except Exception:
            pass
    return text


def normalize_and_save_npz(volume: np.ndarray, out_path: Path):
    # scale to HU-like, if already HU, this is idempotent enough
    volume = volume * 1000.0
    volume = np.clip(volume, -1000, 200)
    volume = ((volume + 400.0) / 600.0).astype(np.float32)
    np.savez_compressed(out_path, volume)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', type=str, required=True, help='Root folder containing 50 scans')
    parser.add_argument('--output_root', type=str, required=True, help='Output dataset root to build (e.g., /workspace/datasets/CT2Rep)')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Train split ratio (rest used for valid)')
    parser.add_argument('--accession_regex', type=str, default=None, help='Optional regex to extract AccessionNo from folder name')
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    train_dir = output_root / 'train'
    valid_dir = output_root / 'valid'
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    # Expect structure: for each scan folder: contains dicom folder and report.pdf (and possibly two STL files we will ignore)
    # We'll use the scan folder name as AccessionNo unless a regex is provided.
    scan_folders = [p for p in input_root.iterdir() if p.is_dir()]
    scan_folders.sort()

    records = []
    for scan_path in scan_folders:
        accession_no = scan_path.name
        dicom_dir = None
        pdf_path = None
        # find dicom folder and pdf under scan_path
        for root, dirs, files in os.walk(scan_path):
            for d in dirs:
                if d.lower().startswith('dicom') or d.lower().endswith('dicom'):
                    dicom_dir = Path(root) / d
            for f in files:
                if f.lower().endswith('.pdf'):
                    pdf_path = Path(root) / f

        if dicom_dir is None:
            # try another heuristic: folder named 'DICOMDIR' etc., otherwise skip
            for d in ['dicom', 'DICOM', 'DICOMDIR']:
                candidate = scan_path / d
                if candidate.exists():
                    dicom_dir = candidate
                    break

        if dicom_dir is None:
            print(f"[WARN] Skipping {scan_path} - no DICOM folder found")
            continue

        try:
            volume = read_dicom_series_to_volume(dicom_dir)
        except Exception as e:
            print(f"[WARN] Failed reading DICOM at {dicom_dir}: {e}")
            continue

        # split by index
        idx = scan_folders.index(scan_path)
        is_train = idx < int(len(scan_folders) * args.split_ratio)
        out_dir = train_dir if is_train else valid_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_npz = out_dir / f"{accession_no}.npz"
        normalize_and_save_npz(volume, out_npz)

        findings_text = ''
        if pdf_path and pdf_path.exists():
            findings_text = extract_text_from_pdf(pdf_path)
        if not findings_text.strip():
            findings_text = 'Not given.'

        records.append({'AccessionNo': accession_no, 'Findings_EN': findings_text})

    # build xlsx covering both splits as expected by code (we can save one xlsx and pass it to both)
    df = pd.DataFrame.from_records(records)
    df.to_excel(str(output_root / 'data_reports.xlsx'), index=False)

    print(f"Prepared dataset at: {output_root}")
    print(f"Train npz count: {len(list(train_dir.glob('*.npz')))} | Valid npz count: {len(list(valid_dir.glob('*.npz')))}")


if __name__ == '__main__':
    main()

