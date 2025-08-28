import argparse
import os
from pathlib import Path
import numpy as np
import torch
import SimpleITK as sitk

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.tokenizers import Tokenizer
from models.ct2rep import CT2RepModel


def read_dicom_series_to_volume(dicom_dir: Path) -> np.ndarray:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in {dicom_dir}")
    series_files = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
    reader.SetFileNames(series_files)
    image = reader.Execute()
    volume = sitk.GetArrayFromImage(image)  # [D, H, W]
    volume = np.transpose(volume, (1, 2, 0)).astype(np.float32)  # [H, W, D]
    return volume


def preprocess_volume(volume: np.ndarray) -> torch.Tensor:
    volume = volume * 1000.0
    volume = np.clip(volume, -1000, 200)
    volume = ((volume + 400.0) / 600.0).astype(np.float32)
    tensor = torch.tensor(volume)

    target_shape = (480, 480, 240)
    dh, dw, dd = target_shape
    h, w, d = tensor.shape

    h_start = max((h - dh) // 2, 0)
    h_end = min(h_start + dh, h)
    w_start = max((w - dw) // 2, 0)
    w_end = min(w_start + dw, w)
    d_start = max((d - dd) // 2, 0)
    d_end = min(d_start + dd, d)

    tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

    pad_h_before = (dh - tensor.size(0)) // 2
    pad_h_after = dh - tensor.size(0) - pad_h_before
    pad_w_before = (dw - tensor.size(1)) // 2
    pad_w_after = dw - tensor.size(1) - pad_w_before
    pad_d_before = (dd - tensor.size(2)) // 2
    pad_d_after = dd - tensor.size(2) - pad_d_before

    tensor = torch.nn.functional.pad(
        tensor,
        (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after),
        value=-1,
    )

    tensor = tensor.permute(2, 0, 1)  # [D, H, W]
    tensor = tensor.unsqueeze(0)  # [1, D, H, W]
    return tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dicom_dir', type=str, required=True, help='Folder containing a DICOM series')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained checkpoint .pth')
    parser.add_argument('--xlsxfile', type=str, required=True, help='Path to Excel used for tokenizer vocab')
    parser.add_argument('--threshold', type=int, default=10)
    parser.add_argument('--max_seq_length', type=int, default=300)
    args = parser.parse_args()

    # Build tokenizer consistent with training
    class ArgsShim:
        pass
    targs = ArgsShim()
    targs.threshold = args.threshold
    targs.xlsxfile = args.xlsxfile
    targs.max_seq_length = args.max_seq_length
    tokenizer = Tokenizer(targs)

    # Build model
    # Reuse minimal args required by model/visual extractor
    margs = ArgsShim()
    margs.d_model = 512
    margs.d_ff = 512
    margs.d_vf = 512
    margs.num_heads = 8
    margs.num_layers = 3
    margs.dropout = 0.1
    margs.logit_layers = 1
    margs.bos_idx = 0
    margs.eos_idx = 0
    margs.pad_idx = 0
    margs.use_bn = 0
    margs.drop_prob_lm = 0.5
    margs.rm_num_slots = 3
    margs.rm_num_heads = 8
    margs.rm_d_model = 512
    margs.sample_method = 'greedy'
    margs.beam_size = 3
    margs.temperature = 1.0
    margs.sample_n = 1
    margs.group_size = 1
    margs.output_logsoftmax = 1
    margs.decoding_constraint = 0
    margs.block_trigrams = 1
    margs.n_gpu = 0

    model = CT2RepModel(margs, tokenizer)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    # Read dicom and preprocess
    volume = read_dicom_series_to_volume(Path(args.dicom_dir))
    tensor = preprocess_volume(volume).unsqueeze(0)  # [1, 1, D, H, W]
    tensor = tensor.to(device)

    with torch.no_grad():
        output = model(tensor, mode='sample')
        report = tokenizer.decode(output.squeeze(0).cpu().numpy())
    print("Generated report:")
    print(report)


if __name__ == '__main__':
    main()

