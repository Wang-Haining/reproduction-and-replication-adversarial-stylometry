# reproduction-and-replication-adversarial-stylometry

This repository contains scripts for the paper *Reproduction and Replication of an Adversarial Stylometry Experiment*.

This reproducibility bundle is archived on Zenodo: https://doi.org/10.5281/zenodo.18729526

## Setting up environments

```bash
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data layout

- **RJ (replication data)** is included in this bundle under:
  `resource/defending-against-authorship-attribution-corpus/`

- **EBG (reproduction data)** is third-party data and is not redistributed here.
  Obtain it from the PSAL Anonymouth repository:
  https://github.com/psal/anonymouth/tree/master/jsan_resources/corpora/amt

  Place the downloaded `amt` corpus at:
  `resource/Drexel-AMT-Corpus/`

## Reproducing from the command line

### SVM (Writeprints-static) and logistic regression (Koppel512)

`train.py` reproduces the SVM and logistic regression results. It takes three arguments:

- `-c` / `--corpus`: `rj` or `ebg`
- `-t` / `--task`:
  - for both corpora: `imitation`, `obfuscation`, `cross_validation`
  - for RJ only: `control`, `backtranslation_de`, `backtranslation_ja`, `backtranslation_de_ja`
- `-m` / `--model`: `svm` or `logistic_regression`

Example: RJ control results with logistic regression:

```bash
python train.py -c rj -t control -m logistic_regression
```

Outputs are printed to stdout and saved under `./results/` with informative filenames, for example:

- `./results/rj_control_logistic_regression.json`

To reproduce all SVM + logistic regression JSON outputs used in the paper:

```bash
# EBG
for task in obfuscation imitation cross_validation; do
  for model in svm logistic_regression; do
    python train.py -c ebg -t "$task" -m "$model"
  done
done

# RJ
for task in control obfuscation imitation cross_validation backtranslation_de backtranslation_ja backtranslation_de_ja; do
  for model in svm logistic_regression; do
    python train.py -c rj -t "$task" -m "$model"
  done
done
```

### RoBERTa (optional, compute-intensive)

`train_roberta.py` reproduces the RoBERTa results. This requires additional dependencies (e.g., PyTorch,
Transformers, Weights \& Biases) and typically a GPU. See `train_roberta.py --help` for arguments.

## Licenses

- Code: ISC (see `LICENSE`)
- RJ corpus, `metadata.csv`, and MTurk study materials: CC0 1.0 (see `DATA_LICENSE`)
- EBG: third-party data obtained from the original source above

## Citation

```tex
@misc{wang2026rr,
  title={Reproduction and Replication of an Adversarial Stylometry Experiment},
  author={Haining Wang and Patrick Juola and Allen Riddell},
  year={2026},
  eprint={2208.07395},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2208.07395}
}
