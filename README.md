# reproduction-and-replication-adversarial-stylometry

A repo hosts scripts for paper *Reproduction and Replication of an Adversarial Stylometry Experiment*.

## Setting up Environments

```bash
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Reproducing with Command Line

The results can be easily reproduced with `train.py` module.
The module takes three optional arguments: `-c` (`--corpus`), `-t` (`--task`), and `-m` (`--model`).
For example, the RJ corpus, to produce results under the control group using logistic regression (with the Koppel512
featureset), runs the following.

```bash
python train.py -c rj -t control -m logistic_regression
```
The results will be printed out and a .json file will be saved to './results' with informative file names (in this case,
'./results/rj_control_logistic_regression.json'). Running one experiment can take several minutes.

Argument `-c` can be specified as either 'rj' (the Riddell-Juola corpus) or 'ebg' (the Extended-Brennan-Greenstadt
corpus).
For both RJ and EBG corpus, 'imitation', 'obfuscation', and 'cross_validation' can be specified to `-t`. RJ can take in
additionally values: 'control', 'translation_ja', 'translation_de', and 'translation_de_ja'.
Argument `-m` takes either 'svm' or 'logistic_regression'. When specified as 'svm', the "writeprints-static" featureset
will be used; otherwise the Kopppel512 featureset will be used.

## License
ISC

## Citation
```tex
@misc{wang2026reproductionreplicationadversarialstylometry,
      title={Reproduction and Replication of an Adversarial Stylometry Experiment},
      author={Haining Wang and Patrick Juola and Allen Riddell},
      year={2026},
      eprint={2208.07395},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2208.07395},
}
```
