"""
This module is used to reproduce roberta-base results in *Reproduction and Replication of an Adversarial
Stylometry Experiment*.

Author: Reproduction and Replication of an Adversarial Stylometry Experiment Authors
Version: 1.0.0
License: ISC
"""

__author__ = (
    "Reproduction and Replication of an Adversarial Stylometry Experiment Authors"
)
__version__ = "1.0.0"
__license__ = "ISC"

import os
import json
import time
import torch
import wandb
import shutil
import argparse
import numpy as np
from copy import deepcopy
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments, TrainerCallback
from utilities import get_data_from_rj, get_data_from_ebg, TASKS, SEED
from transformers import RobertaForSequenceClassification, RobertaTokenizer, EarlyStoppingCallback


def main(corpus,
          task,
          runs,
          batch_size,
          wandb_project_name):
    """
    Reproducing the results reported in Reproduction and Replication of an Adversarial Stylometry Experiment.
    Args:
        corpus: str, should be in ['rj', 'ebg']
        task: str, EBG and RJ can have ['obfuscation', 'imitation', 'cross_validation'], RJ additionally can be
            specified with ['control', 'backtranslation_ja', 'backtranslation_de', 'backtranslation_de_ja']
        runs: int, how many runs on a certain candidate size should be performed
        batch_size: int, batch size for all GPUs
        wandb_project_name: str, wandb project name
    Returns:
         None.
    """

    assert corpus in ["rj", "ebg"], "Expected `corpus`: rj and ebg."
    if corpus == "rj":
        assert task in TASKS, f"Expected `task` for {corpus} `corpus`: {TASKS}"
    else:
        assert task in TASKS[:3], f"Expected `task` for {corpus} `corpus`: {TASKS[:3]}"

    print("*" * 89)
    print(
        "Reproduce results in Reproduction and Replication of an Adversarial Stylometry Experiment"
    )
    # reads data
    if corpus == "rj":
        train_text, train_label, val_text, val_label, test_text, test_label = get_data_from_rj(task=task, dev=True)
    else:
        train_text, train_label, val_text, val_label, test_text, test_label = get_data_from_ebg(task=task, dev=True)
    model_name = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    cand_sizes = (
        list(range(len(set(train_label)))[::5])[1:]
        if len(set(train_label)) <= 40
        else list(range(5, 45, 5))
    )
    accs = {f"{cand_size}_candidates": [] for cand_size in cand_sizes}
    rng = np.random.default_rng(SEED)

    for cand_size in cand_sizes:

        for run in range(runs):
            # sample `cand_size` authors
            val_text_, val_label_ = zip(
                *rng.choice(list(zip(val_text, val_label)), size=cand_size, replace=False).tolist())
            train_text_, train_label_ = zip(*[tpl for tpl in zip(train_text, train_label) if tpl[1] in val_label_])

            if task != 'cross_validation':
                # address certain attack
                test_text_, test_label_ = zip(*[tpl for tpl in zip(test_text, test_label) if tpl[1] in val_label_])
            else:
                # cross-validation like split
                train_text_, train_label_, test_text_, test_label_ = train_test_split(list(train_text_),
                                                                                      list(train_label_),
                                                                                      test_size=cand_size,
                                                                                      random_state=SEED,
                                                                                      stratify=list(train_label_))
            # encode labels
            le = LabelEncoder()
            train_label_ = le.fit_transform(train_label_)
            test_label_ = le.transform(test_label_)
            val_label_ = le.transform(val_label_)
            # encode data
            train_encodings = tokenizer(train_text_, truncation=True, padding=True, max_length=512)
            val_encodings = tokenizer(val_text_, truncation=True, padding=True, max_length=512)
            test_encodings = tokenizer(test_text_, truncation=True, padding=True, max_length=512)
            train_dataset = CommonDataset(train_encodings, train_label_)
            val_dataset = CommonDataset(val_encodings, val_label_)
            test_dataset = CommonDataset(test_encodings, test_label_)

            # init logger each time
            wandb.init(project=wandb_project_name,
                       group=f'{corpus}-{task}-cand_size={str(cand_size)}')
            run_name = f'{corpus}-{task}-cand_size={str(cand_size)}-run={str(run)}'
            training_args = TrainingArguments(run_name=f'{corpus}-{task}-{cand_size}-{str(run)}',
                                              output_dir=os.path.join('script/saved_models/roberta_ckpts', run_name),
                                              seed=SEED,
                                              do_eval=True,
                                              learning_rate=3e-5,
                                              adam_beta1=0.99,
                                              adam_beta2=0.9999,
                                              adam_epsilon=1e-08,
                                              per_device_train_batch_size=batch_size,
                                              per_device_eval_batch_size=batch_size,
                                              warmup_ratio=.05,
                                              load_best_model_at_end=True,
                                              metric_for_best_model='eval_loss',
                                              greater_is_better=False,
                                              dataloader_num_workers=4,
                                              dataloader_pin_memory=True,
                                              logging_strategy='epoch',
                                              save_strategy='epoch',
                                              evaluation_strategy="epoch",
                                              # housekeeping
                                              fp16=False,
                                              overwrite_output_dir=True,
                                              num_train_epochs=200,
                                              save_total_limit=20,
                                              report_to='wandb')
            model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=len(set(train_label_)))
            # .to(
            #     torch.device(device))
            # train the model
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=50)]
            )
            trainer.add_callback(TrainAccuracyCallback(trainer))
            trainer.train()

            # make prediction
            prediction = trainer.predict(test_dataset)
            acc = prediction.metrics['test_accuracy']
            wandb.log({"test_accuracy": acc})
            accs[f"{cand_size}_candidates"].append(acc)

            wandb.finish()
            # after training, housecleaning
            del training_args, trainer, model, train_dataset, val_dataset
            # free GPU memory for the next run
            torch.cuda.empty_cache()
            # remove model folders
            shutil.rmtree(os.path.join('script/saved_models/roberta_ckpts', run_name))
            time.sleep(10)
    if not os.path.isdir("results"):
        os.mkdir("results")
    json.dump(accs, open(f"results/{corpus}_{task}_{model_name}.json", "w"))

    for cand_size in cand_sizes:
        print(
            f"{str(cand_size)} authors: mean accuracy {np.round(np.mean(accs[f'{cand_size}_candidates'])* 100, 2)}"
            f"% (std. {np.round(np.std(accs[f'{cand_size}_candidates'])* 100, 2)}%)"
        )

    print(f"\nResults have been saved to 'results/{corpus}_{task}_{model_name}.json'")
    print("*" * 89)


class CommonDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }


class TrainAccuracyCallback(TrainerCallback):

    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reproduce results in the paper Reproduction and Replication of an Adversarial Stylometry "
                    "Experiment"
    )
    parser.add_argument(
        "-c",
        "--corpus",
        default="rj",
        help="corpus to use, allows 'rj' (Riddell-Juola corpus with machine transltion samples) and 'ebg' ("
             "Extended-Brennan-Greenstadt corpus)",
    )
    parser.add_argument(
        "-t",
        "--task",
        default="obfuscation",
        help="task for a specific corpus, rj and ebg both allow 'imitation', 'obfuscation', and 'cross_validation'; "
             "rj allows additional 'control', 'translation_de', 'translation_ja', and 'translation_de_ja' ",
    )
    parser.add_argument(
        "-r",
        "--runs",
        default=10,
        type=int,
        help="how many runs on a certain candidate size should be performed, default 10",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=8,
        type=int,
        help="batch size for all GPUs, default 8",
    )
    parser.add_argument(
        "-w",
        "--wandb_project_name",
        default="reproduction_and_replication_adversarial_stylometry-test",
        help="wandb project name for monitoring",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="%(prog)s (version {version})".format(version=__version__),
    )
    args = parser.parse_args()

    # run exps
    main(args.corpus, args.task, args.runs, args.batch_size, args.wandb_project_name)
