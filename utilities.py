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
import chardet
import numpy as np
from writeprints_static import WriteprintsStatic


# possible tasks
TASKS = [
    "imitation",
    "obfuscation",
    "cross_validation",
    "control",
    "backtranslation_de",
    "backtranslation_ja",
    "backtranslation_de_ja",
    "special_english",
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
