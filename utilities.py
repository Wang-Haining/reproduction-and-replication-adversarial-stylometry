"""
This module contains useful functions used to reproduce results in *Reproduction and Replication of an Adversarial
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
import re
import json
import wandb
import random
import chardet
import numpy as np
from copy import deepcopy
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from writeprints_static import WriteprintsStatic
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, Normalizer
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import RobertaForSequenceClassification, RobertaTokenizer, EarlyStoppingCallback

# possible tasks
TASKS = [
    "imitation",
    "obfuscation",
    "cross_validation",
    "control",
    "backtranslation_de",
    "backtranslation_ja",
    "backtranslation_de_ja",
]
SEED = 42

def get_data_from_rj(
        task,
        corpus_dir="resource/defending-against-authorship-attribution-corpus/corpus",
        dev=False
):
    """
    Reads in texts and labels from the Riddell-Juola (RJ) corpus. The RJ version contains a control group, attacks of
    imitation and obfuscation, and three round-trip translation attacks (['translation_de', 'translation_ja',
    'translation_de_ja']). The translation attacks have testing samples are translated with Google Translate and
    share the same training examples with the control group.
    Args:
        task: a str, should be in ['control', 'imitation', 'obfuscation', 'backtranslation_de', 'backtranslation_ja',
            'backtranslation_de_ja', 'cross_validation']; when specified as 'cross_validation', all the training samples
             of ['control', 'imitation', 'obfuscation'] and no test samples will be returned
        corpus_dir: a str, path to RJ corpus
        dev: a bool, whether the first samples of each author is used as a dev sample, used in deep learning scenarios
    Returns:
        if not dev, four lists, text/label of train/test sets
        else, six lists, text/label of train/dev/test sets
    """

    def get_task_specific_data(task, dev):
        train_text_, train_label_, dev_text_, dev_label_ = [], [], [], []
        authors = [f.name.split(".")[0] for f in os.scandir(os.path.join(corpus_dir, "attacks_" + task)) if not f.name.startswith(".")]
        for dir_ in [os.path.join(corpus_dir, author) for author in authors if author != "cfeec8"]:  # cfeec8 does not have training data
            for raw in os.scandir(dir_):
                if dev:
                    if not raw.name.endswith('_01.txt'):
                        train_text_.append(open(raw.path, 'r', encoding='utf8').read())
                        train_label_.append(raw.name.split('_')[0])
                    else:
                        dev_text_.append(open(raw.path, 'r', encoding='utf8').read())
                        dev_label_.append(raw.name.split('_')[0])
                else:
                    train_text_.append(open(raw.path, "r", encoding="utf8").read())
                    train_label_.append(raw.name.split("_")[0])
        # read in testing
        test_text_, test_label_ = zip(
            *[(open(f.path, "r", encoding=chardet.detect(open(f.path, "rb").read())["encoding"]).read(), f.name.split(".")[0])
                for f in os.scandir(os.path.join(corpus_dir, "attacks_" + task))
                if ".txt" in f.name])
        if not dev:
            return train_text_, train_label_, list(test_text_), list(test_label_)
        else:
            return train_text_, train_label_, dev_text_, dev_label_, list(test_text_), list(test_label_)

    train_text, train_label, dev_text, dev_label, test_text, test_label = [], [], [], [], [], []
    if task != "cross_validation":
        return get_task_specific_data(task, dev)
    else:
        for _task in ["imitation", "obfuscation", "control"]:
            if dev:
                train_text_, train_label_, dev_text_, dev_label_, _, _ = get_task_specific_data(_task, dev)
                train_text.extend(train_text_)
                train_label.extend(train_label_)
                dev_text.extend(dev_text_)
                dev_label.extend(dev_label_)
            else:
                train_text_, train_label_, _, _ = get_task_specific_data(_task, dev)
                train_text.extend(train_text_)
                train_label.extend(train_label_)
        if dev:
            return train_text, train_label, dev_text, dev_label, test_text, test_label
        else:
            return train_text, train_label, test_text, test_label


def get_data_from_ebg(task, corpus_dir="resource/Drexel-AMT-Corpus", dev=False):
    """
    Reads in texts and labels from the Extended Brennan-Greenstadt (EBG) corpus.
    Args:
        task: a str, should be in ['obfuscation', 'imitation', 'cross_validation']
        corpus_dir: a str, path to EBG corpus
        dev: a bool, whether the first samples of each author is used as a dev sample, used in deep learning scenarios
    Returns:
        if not dev, four lists, text/label of train/test sets
        else, six lists, text/label of train/dev/test sets
    """
    train_text, train_label, dev_text, dev_label, test_text, test_label = [], [], [], [], [], []

    for author in os.scandir(corpus_dir):
        if not author.name.startswith("."):
            if dev:
                train_text_, dev_text_ = [], []
                for f in os.scandir(author.path):
                    if re.match(r'[a-z]+_[0-9]{2}_*', f.name):
                        if f.name.endswith('_01_1.txt'):
                            dev_text_.append(open(f.path, 'r', encoding=chardet.detect(open(f.path, 'rb').read())[
                                'encoding']).read())
                        else:
                            train_text_.append(open(f.path, 'r', encoding=chardet.detect(open(f.path, 'rb').read())[
                                'encoding']).read())
                train_text.extend(train_text_)
                dev_text.extend(dev_text_)
                train_label.extend([author.name] * len(train_text_))
                dev_label.extend([author.name] * len(dev_text_))
            else:
                train_text.extend(
                    [open(f.path, "r", encoding=chardet.detect(open(f.path, "rb").read())["encoding"],).read()
                     for f in os.scandir(author.path) if re.match(r"[a-z]+_[0-9]{2}_*", f.name)])
                train_label.extend([author.name] * len([f.name for f in os.scandir(author.path)
                                                        if re.match(r"[a-z]+_[0-9]{2}_*", f.name)]))
            # read in testing
            if task == "imitation":
                test_text.extend(
                    [open(f.path, "r", encoding=chardet.detect(open(f.path, "rb").read())["encoding"]).read()
                     for f in os.scandir(author.path) if re.match(r"[a-z]+_imitation_01.txt", f.name)])
                test_label.append(author.name)
            elif task == "obfuscation":
                test_text.extend(
                    [open(f.path, "r", encoding=chardet.detect(open(f.path, "rb").read())["encoding"]).read()
                     for f in os.scandir(author.path) if re.match(r"[a-z]+_obfuscation.txt", f.name)])
                test_label.append(author.name)
            else:
                # when task_name == 'cross_validation'
                pass
    if dev:
        return train_text, train_label, dev_text, dev_label, test_text, test_label
    else:
        return train_text, train_label, test_text, test_label


def vectorize_writeprints_static(raws):
    """
    Extracts `writeprints-static` features from a list of texts. Done by PyPI library `writeprints-static` v0.0.2.
    Args:
        raws: a list of str.
    Returns:
         a np.array of writeprints-static numeric.
    """
    vec = WriteprintsStatic()
    features = vec.transform(raws)

    return features.toarray()


def vectorize_koppel512(raws):
    """
    Extracts `Koppel512` function words count.
    Args:
        raws: a list of str.
    Returns:
         a np.array of Koppel512 numeric.
    """
    function_words = open("resource/koppel_function_words.txt", "r").read().splitlines()

    return np.array(
        [
            [
                len(re.findall(r"\b" + function_word + r"\b", raw.lower()))
                for function_word in function_words
            ]
            for raw in raws
        ]
    )


class TrainAccuracyCallback(TrainerCallback):

    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy


def get_huggingface_model_accuracy(run_name,
                                   train_text, train_label, val_text, val_label,
                                   runs,
                                   cand_sizes,
                                   cv_folds,
                                   # save_dir,
                                   model_name=MODEL_NAME,
                                   max_length=MAX_LENGTH,
                                   test_text=None, test_label=None,
                                   device="cuda",
                                   learning_rate=LEARNING_RATE,
                                   batch_size=BATCH_SIZE,
                                   project_name='reproducing-adversarial-stylometry-distilbert-base-uncased'):
    """

    Args:
        project_name: str, wandb project name
        batch_size: int, batch size
        learning_rate: float, learning rate
        run_name: str, a group name used on wandb
        train_text: list
        train_label: list
        val_text: list
        val_label: list
        runs: int, how many rounds should be performed on a certain candidate size
        cand_sizes: list of int, candidate sizes
        cv_folds: bool, True if a cross-validation-like setting; otherwise False and specify `test_text` and `test_label`
        model_name: str, 'roberta-base'
        max_length: int, 512
        test_text: list
        test_label: list
        device: str, e.g., 'cuda:0'

    Returns:

    """
    # """
    # Fits 10 roberta-base, predict 100 randomly selected samples for each cand_size in cand_sizes.
    #
    # # :param raw_train: a list of str, from an Attack instance
    # # :param label_train: a np.ndarray, an Attack instance
    # # :param raw_test: a list of str
    # # :param label_test: a np.ndarray
    # :param rounds: an int, how many random sampling is desired, rounds in BAG2012 equals to 1000
    # :param prediction_per_model: an int, how many random sampling is desired,
    #         prediction_per_model * rounds == 1000 for each cand_size in cand_sizes.
    # :param cand_sizes: a list, how many candidates are desired in each round
    # :param save_dir: a str, the dir to save
    # :param save_dir: a str, the dir to save
    # :param attack_mode: a str, indicates the current attack mode [ 'obfuscation','imitation', 'special_english', 'cv'],
    #         'special_english' is to be implemented [TODO]
    # :param label_to_frame: a list, for framing CM use
    # :param raw_to_frame: a list, for framing CM use
    # :return: a dict of lists, keyed by `cand_sizes`, valued by `rounds` accuracy of calculation
    # """
    rng = np.random.default_rng(SEED)
    if model_name == 'roberta-base':
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
    else:
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    for cand_size in cand_sizes:

        for i in range(runs):
            # sample `cand_size` authors
            val_text_, val_label_ = zip(
                *rng.choice(list(zip(val_text, val_label)), size=cand_size, replace=False).tolist())
            train_text_, train_label_ = zip(*[tpl for tpl in zip(train_text, train_label) if tpl[1] in val_label_])

            if (test_text and test_label):
                # address certain attack
                test_text_, test_label_ = zip(*[tpl for tpl in zip(test_text, test_label) if tpl[1] in val_label_])
            elif cv_folds:
                # cross-validation like split
                train_text_, train_label_, test_text_, test_label_ = create_dev_from_train(list(train_text_),
                                                                                           list(train_label_))
            else:
                print(
                    "Set `test_text` and `test_label` for a certain attack; otherwise leave as None and specify `cv_folds` as True.")
            # return train_text_, train_label_, val_text_, val_label_, test_text_, test_label_
            # init a new logger within a group
            wandb.init(project=project_name,
                       group=run_name + "-" + str(cand_size))

            # encode labels
            le = LabelEncoder()
            train_label_ = le.fit_transform(train_label_)
            test_label_ = le.transform(test_label_)
            val_label_ = le.transform(val_label_)

            # encode data
            train_encodings = tokenizer(train_text_, truncation=True, padding=True, max_length=max_length)
            val_encodings = tokenizer(val_text_, truncation=True, padding=True, max_length=max_length)
            test_encodings = tokenizer(test_text_, truncation=True, padding=True, max_length=max_length)
            # convert our tokenized data into a torch Dataset
            train_dataset = AAADataset(train_encodings, train_label_)
            val_dataset = AAADataset(val_encodings, val_label_)
            test_dataset = AAADataset(test_encodings, test_label_)

            training_args = TrainingArguments(run_name=run_name + "-" + str(round),
                                              output_dir='script/saved_models/roberta_ckpts/' + run_name,
                                              seed=SEED,
                                              do_eval=True,
                                              learning_rate=learning_rate,
                                              adam_beta1=0.99,
                                              adam_beta2=0.9999,
                                              adam_epsilon=1e-08,
                                              # per_device_train_batch_size=min(len(train_label_), batch_size),
                                              # per_device_eval_batch_size=min(len(val_label_), batch_size),
                                              per_device_train_batch_size=batch_size,
                                              per_device_eval_batch_size=batch_size,
                                              warmup_ratio=.05,
                                              # weight_decay=1e-4,  # strength of weight decay
                                              load_best_model_at_end=True,
                                              metric_for_best_model='eval_loss',
                                              greater_is_better=False,
                                              # metric_for_best_model='eval_accuracy',
                                              # greater_is_better=True,
                                              dataloader_num_workers=4,
                                              dataloader_pin_memory=True,
                                              logging_strategy='epoch',
                                              # logging_steps=1,
                                              # eval_steps=1,
                                              save_strategy='epoch',
                                              evaluation_strategy="epoch",
                                              # housekeeping
                                              fp16=False,
                                              overwrite_output_dir=True,
                                              num_train_epochs=MAX_EPOCHS_ROBERTA,
                                              save_total_limit=20,
                                              report_to='wandb')
            if model_name == 'roberta-base':
                model = RobertaForSequenceClassification.from_pretrained(model_name,
                                                                         num_labels=len(set(train_label_))).to(
                    torch.device(device))
            else:
                model = DistilBertForSequenceClassification.from_pretrained(model_name,
                                                                            num_labels=len(set(train_label_))).to(
                    torch.device(device))
            # train the model
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                # callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE_ROBERTA,
                #                                  early_stopping_threshold=EARLY_STOPPING_THRESHOLD)]
            )
            trainer.add_callback(TrainAccuracyCallback(trainer))
            trainer.train()

            # make prediction
            prediction = trainer.predict(test_dataset)
            wandb.log({"test_accuracy": prediction.metrics['test_accuracy']})

            wandb.finish()
            # after training, housecleaning
            del training_args, trainer, model, train_dataset, val_dataset
            # free GPU memory for the next run
            torch.cuda.empty_cache()
            # remove model folders
            shutil.rmtree('script/saved_models/roberta_ckpts/' + run_name)
            time.sleep(5)
