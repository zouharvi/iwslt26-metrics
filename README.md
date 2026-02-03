# IWSLT26 metrics shared task

This repository contains data and all the wrangling, baseline and evaluation scripts for [IWSLT26 Metrics Shared Task](https://iwslt.org/2026/metrics).

```bash
pip3 install -r requirements
```


## Meta-evaluation

To evaluate the output of automated metrics, pass a file with a score per each line in `-m` to the following script.
The `-i` points to the JSONL file with `"score"` value in the dictionary.

```bash
python3 scripts/04-hf_to_jsonl.py # generate data/iwslt26/dev.jsonl
python3 evaluation -i data/iwslt26/dev.jsonl -m data/output/iwslt26dev_*.jsonl
```

The results for textual human transcript baselines (IWSLT26 dev) are:
```
SEGMENT-LEVEL
      iwslt26dev_comet_partial: 11.6% | ende:11.3% enzh:12.0%
              iwslt26dev_comet: 34.6% | ende:32.6% enzh:36.5%
           iwslt26dev_speechqe: 29.2% | ende:26.6% enzh:31.8%
  iwslt26dev_blaser_2_0_qe_s2t: 24.4% | ende:22.0% enzh:26.8%


SYSTEM-LEVEL
      iwslt26dev_comet_partial: 56.6% | ende:44.4% enzh:68.7%
              iwslt26dev_comet: 89.4% | ende:86.2% enzh:92.6%
           iwslt26dev_speechqe: 86.0% | ende:78.6% enzh:93.4%
  iwslt26dev_blaser_2_0_qe_s2t: 76.9% | ende:86.0% enzh:67.7%
```

## Baselines

We include several baselines. The format for a submission is a list of numbers with the same length as the input JSONL file.
```bash
mkdir -p data/output/ data/iwslt26/
python3 scripts/04-hf_to_jsonl.py
python3 baselines -i data/iwslt26/dev.jsonl -m asr_comet -o data/output/iwslt26dev_comet.jsonl
python3 baselines -i data/iwslt26/dev.jsonl -m asr_comet_partial -o data/output/iwslt26dev_comet_partial.jsonl
```

### End-to-End SpeechQE
Preprocess: it takes `dev.jsonl` as input and outputs `dev.tsv`
```bash
python baselines/speeechqe_preprocess.py 
```
Run E2E SpeechQE model.
Please note that enzh is zero-shot (The `SpeechQE-TowerInstruct-7B-en2de` model is only trained with QE.ende task).
```bash
git clone https://github.com/h-j-han/SpeechQE.git
conda activate speechqe # setup details are in the repo
cd SpeechQE
python speechqe/score_speechqe.py \
    --dataroot=data/iwslt26 \
    --manifest_files=dev.tsv \
    --speechqe_model=h-j-han/SpeechQE-TowerInstruct-7B-en2de \
```
```
python3 evaluation -i data/iwslt26/dev.jsonl -m data/output/speechqe.iwslt26dev.jsonl 
```

### Blaser
Using speech to text version of [blaser2.0qe](https://huggingface.co/facebook/blaser-2.0-qe).
```
python baselines/blaser.py -i dev.jsonl -o data/output/iwslt26dev_blaser_2_0_qe_s2t.jsonl
```
