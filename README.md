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

The results for baselines (IWSLT26 dev) are:
```
SEGMENT-LEVEL
      iwslt26dev_comet_partial: 11.6% | ende:11.3% enzh:12.0%
              iwslt26dev_comet: 34.6% | ende:32.6% enzh:36.5%


SYSTEM-LEVEL
      iwslt26dev_comet_partial: 56.6% | ende:44.4% enzh:68.7%
              iwslt26dev_comet: 89.4% | ende:86.2% enzh:92.6%
```

## Baselines

We include several baselines. The format for a submission is a list of numbers with the same length as the input JSONL file.
```bash
mkdir -p data/output/ data/iwslt26/
python3 scripts/04-hf_to_jsonl.py
python3 baselines -i data/iwslt26/dev.jsonl -m asr_comet -o data/output/iwslt26dev_comet.jsonl
python3 baselines -i data/iwslt26/dev.jsonl -m asr_comet_partial -o data/output/iwslt26dev_comet_partial.jsonl
```