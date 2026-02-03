import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Union, Optional, BinaryIO
from sonar.models.blaser.loader import load_blaser_model
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline

args = argparse.ArgumentParser()
args.add_argument("-i", "--input", type=str, required=True)
args.add_argument("-o", "--output", type=str, required=True)
args = args.parse_args()

os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.input, "r") as f:
  data = [json.loads(line) for line in f.readlines()]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blaser = load_blaser_model("blaser_2_0_qe", device=device).eval()
text_embedder = TextToEmbeddingModelPipeline(
    encoder="text_sonar_basic_encoder",
    tokenizer="text_sonar_basic_encoder",
    device=device)
speech_embedder = SpeechToEmbeddingModelPipeline(
    encoder='sonar_speech_encoder_eng', device=device)
langcode = {"en": "eng_Latn", "de": "deu_Latn", "zh": "zho_Hans"}


def get_waveform(
    path_or_fp: Union[str, BinaryIO],
    normalization=True,
    mono=True,
    frames=-1,
    start=0,
    always_2d=False,
    output_sample_rate=16000,
) -> Tuple[np.ndarray, int]:
  meta = path_or_fp.split(":")
  if len(meta) == 3 and (meta[0].endswith(".wav") or meta[0].endswith(".flac")):
    path_or_fp = meta[0]
    start = int(meta[1])
    frames = int(meta[2])
  else:
    path_or_fp = path_or_fp

  if isinstance(path_or_fp, str):
    ext = Path(path_or_fp).suffix
    if ext in [".wav", ".flac", ".ogg", ".mp3"]:
      pass
    else:
      raise ValueError(f"Unsupported audio format: {ext}")

  try:
    import soundfile as sf
  except ImportError:
    raise ImportError(
        "Please install soundfile to load WAV/FLACC/OGG/MP3 audios")
  waveform, sample_rate = sf.read(
      path_or_fp, dtype="float32", always_2d=True, frames=frames, start=start)
  waveform = waveform.T

  waveform, sample_rate = convert_waveform(
      waveform, sample_rate, to_mono=mono, to_sample_rate=output_sample_rate)
  if not normalization:
    waveform *= 2**15
  if not always_2d:
    waveform = waveform.squeeze(axis=0)
  return waveform


def convert_waveform(
    waveform: Union[np.ndarray, torch.Tensor],
    sample_rate: int,
    normalize_volume: bool = False,
    to_mono: bool = False,
    to_sample_rate: Optional[int] = None,
) -> Tuple[Union[np.ndarray, torch.Tensor], int]:
  """convert a waveform:
    - to a target sample rate
    - from multi-channel to mono channel
    - volume normalization
    Args:
        waveform (numpy.ndarray or torch.Tensor): 2D original waveform
            (channels x length)
        sample_rate (int): original sample rate
        normalize_volume (bool): perform volume normalization
        to_mono (bool): convert to mono channel if having multiple channels
        to_sample_rate (Optional[int]): target sample rate
    Returns:
        waveform (numpy.ndarray): converted 2D waveform (channels x length)
        sample_rate (float): target sample rate
    """
  try:
    import torchaudio.sox_effects as ta_sox
  except ImportError:
    raise ImportError("Please install torchaudio: pip install torchaudio")

  effects = []
  if normalize_volume:
    effects.append(["gain", "-n"])
  if to_sample_rate is not None and to_sample_rate != sample_rate:
    effects.append(["rate", f"{to_sample_rate}"])
  if to_mono and waveform.shape[0] > 1:
    effects.append(["channels", "1"])
  if len(effects) > 0:
    is_np_input = isinstance(waveform, np.ndarray)
    _waveform = torch.from_numpy(waveform) if is_np_input else waveform
    converted, converted_sample_rate = ta_sox.apply_effects_tensor(
        _waveform, sample_rate, effects)
    if is_np_input:
      converted = converted.numpy()
    return converted, converted_sample_rate
  return waveform, sample_rate


scores = []
for line in tqdm(data, total=len(data)):
  path = line["audio_filename"]
  tgt_lang = langcode[line["tgt_lang"]]
  mt = line["tgt_text"]
  wvf = get_waveform(path, output_sample_rate=16000)
  wvf = torch.tensor(wvf).unsqueeze(0).to(device)
  src_embs = speech_embedder.predict([wvf])
  mt_embs = text_embedder.predict([mt], source_lang=tgt_lang)
  score = blaser(src=src_embs, mt=mt_embs).item()
  scores.append(score)

with open(args.output, "w") as f:
  for score in scores:
    f.write(f"{score}\n")
