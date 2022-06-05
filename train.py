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
import random
import chardet
import warnings
import argparse
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from writeprints_static import WriteprintsStatic
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, Normalizer


def get_data_from_rj(
    task, corpus_dir="resource/defending-against-authorship-attribution-corpus/corpus"
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
    Returns:
        four lists, text/label of train/test sets
    """

    def get_task_specific_data(task):
        train_text_, train_label_ = [], []
        authors = [
            f.name.split(".")[0]
            for f in os.scandir(os.path.join(corpus_dir, "attacks_" + task))
            if not f.name.startswith(".")
        ]
        for dir_ in [
            os.path.join(corpus_dir, author) for author in authors if author != "cfeec8"
        ]:  # cfeec8 does not have training data
            for raw in os.scandir(dir_):
                train_text_.append(open(raw.path, "r", encoding="utf8").read())
                train_label_.append(raw.name.split("_")[0])
        # read in testing
        test_text_, test_label_ = zip(
            *[
                (
                    open(
                        f.path,
                        "r",
                        encoding=chardet.detect(open(f.path, "rb").read())["encoding"],
                    ).read(),
                    f.name.split(".")[0],
                )
                for f in os.scandir(os.path.join(corpus_dir, "attacks_" + task))
                if ".txt" in f.name
            ]
        )
        return train_text_, train_label_, list(test_text_), list(test_label_)

    train_text, train_label, test_text, test_label = [], [], [], []
    if task != "cross_validation":
        return get_task_specific_data(task)
    else:
        for _task in ["imitation", "obfuscation", "control"]:
            train_text_, train_label_, _, _ = get_task_specific_data(_task)
            train_text.extend(train_text_)
            train_label.extend(train_label_)
        return train_text, train_label, test_text, test_label


def get_data_from_ebg(task, corpus_dir="resource/Drexel-AMT-Corpus"):
    """
    Reads in texts and labels from the Extended Brennan-Greenstadt (EBG) corpus.
    Args:
        task: a str, should be in ['obfuscation', 'imitation', 'cross_validation']
        corpus_dir: a str, path to EBG corpus
    Returns:
        four lists, text/label of train/test sets
    """
    train_text, train_label, test_text, test_label = [], [], [], []

    for author in os.scandir(corpus_dir):
        if not author.name.startswith("."):
            train_text.extend(
                [
                    open(
                        f.path,
                        "r",
                        encoding=chardet.detect(open(f.path, "rb").read())["encoding"],
                    ).read()
                    for f in os.scandir(author.path)
                    if re.match(r"[a-z]+_[0-9]{2}_*", f.name)
                ]
            )
            train_label.extend(
                [author.name]
                * len(
                    [
                        f.name
                        for f in os.scandir(author.path)
                        if re.match(r"[a-z]+_[0-9]{2}_*", f.name)
                    ]
                )
            )
            # read in testing
            if task == "imitation":
                test_text.extend(
                    [
                        open(
                            f.path,
                            "r",
                            encoding=chardet.detect(open(f.path, "rb").read())[
                                "encoding"
                            ],
                        ).read()
                        for f in os.scandir(author.path)
                        if re.match(r"[a-z]+_imitation_01.txt", f.name)
                    ]
                )
                test_label.append(author.name)
            elif task == "obfuscation":
                test_text.extend(
                    [
                        open(
                            f.path,
                            "r",
                            encoding=chardet.detect(open(f.path, "rb").read())[
                                "encoding"
                            ],
                        ).read()
                        for f in os.scandir(author.path)
                        if re.match(r"[a-z]+_obfuscation.txt", f.name)
                    ]
                )
                test_label.append(author.name)
            else:
                # when task_name == 'cross_validation'
                pass
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


def train(corpus, task, model):
    """
    The heavy-lifting function.
    Args:
        corpus: a str, should be in ['rj', 'ebg']
        task: a str, EBG and RJ can have ['obfuscation', 'imitation', 'cross_validation'], RJ additionally can be
            specified with ['control', 'translation_ja', 'translation_de', 'translation_de_ja']
        model: a str, should be in ['svm', 'logistic_regression']; when specified as 'svm', writeprints-static features
            set will be used, otherwise Koppel512 will be used.
    Returns:
         None.
    """

    assert corpus in ["rj", "ebg"], "Expected `corpus`: rj and ebg."
    if corpus == "rj":
        assert task in TASKS, f"Expected `task` for {corpus} `corpus`: {TASKS}"
    else:
        assert task in TASKS[:3], f"Expected `task` for {corpus} `corpus`: {TASKS[:3]}"
    assert model in [
        "logistic_regression",
        "svm",
    ], f"Expected `algorithm`: {['logistic_regression', 'svm']}"
    print("*" * 89)
    print(
        "Reproduce results in Reproduction and Replication of an Adversarial Stylometry Experiment"
    )
    # reads data
    if corpus == "rj":
        train_text, train_label, test_text, test_label = get_data_from_rj(task)
    else:
        train_text, train_label, test_text, test_label = get_data_from_ebg(task)
    # feature engineering
    if task == "cross_validation":
        X_test = []
        if model == "logistic_regression":
            X_train = vectorize_koppel512(train_text)
        else:
            X_train = vectorize_writeprints_static(train_text)
    else:
        if model == "logistic_regression":
            X_train, X_test = map(vectorize_koppel512, (train_text, test_text))
        else:
            X_train, X_test = map(vectorize_writeprints_static, (train_text, test_text))

    runs = RUNS if task != "cross_validation" else round(0.1 * RUNS)
    learner = (
        (
            "logistic regression",
            LogisticRegression(
                C=1, solver="lbfgs", max_iter=1e9, multi_class="multinomial"
            ),
        )
        if model == "logistic_regression"
        else (
            "polynomial SVM",
            SVC(C=0.01, coef0=100, degree=3, gamma=0.0001, kernel="poly", max_iter=-1),
        )
    )

    cand_sizes = (
        list(range(len(set(train_label)))[::5])[1:]
        if len(set(train_label)) <= 40
        else list(range(5, 45, 5))
    )
    if task != "cross_validation":
        print(
            f"Calculating {corpus}-{task} by randomly choosing {cand_sizes} authors each time for {str(runs)} times:"
        )
    else:
        print(
            f"Calculating {corpus}-{task} by randomly choosing {cand_sizes} authors each time for {str(round(runs*10))} times ({str(runs)} runs of 10-fold cross-validation):"
        )

    def calculate_accuary(
        _task, _learner, _runs, _cand_sizes, _X_train, _y_train, _X_test, _y_test
    ):
        """
        calculates accuracy based on given parameters.
        Args:
            _task: a str, same as the parent function's `task`
            _learner: a two-element tuple, a specific learner according to the parent function's `model`
            _runs: an int, 1000 for non-cross-validation settings and 100 for otherwise
            _cand_sizes: a list of int, candidate size that will be sub-sampled from a scenario in each run
            _X_train: np.array, training data numeric
            _y_train: a list of str, training labels
            _X_test: np.array, testing data numeric; [] for cross-validation
            _y_test: a list of str, testing labels; [] for cross-validation
        Returns:
            accs: a dict keyed by f'{cand_size}__candidates', valued by accuracy metrics.
        """
        accs = {f"{cand_size}_candidates": [] for cand_size in cand_sizes}

        for cand_size in _cand_sizes:
            for _ in range(runs):
                # sample `cand_size` authors
                # cands = rng.choice(list(set(_y_train)), size=cand_size, replace=False)
                cands = random.sample(list(set(_y_train)), k=cand_size)
                # get indices of X, y for `cands`
                idx_train = [
                    index for index, element in enumerate(_y_train) if element in cands
                ]
                # get X_ah_train, y_ah_train
                X_ah_train = _X_train[idx_train, :]
                y_ah_train = [_y_train[i] for i in idx_train]

                # specify pipeline
                pipeline = Pipeline(
                    [
                        ("normalizer", Normalizer(norm="l1")),
                        ("scaler", StandardScaler()),
                        learner,
                    ]
                )

                if _task == "cross_validation":
                    accs[f"{cand_size}_candidates"].extend(
                        cross_val_score(pipeline, X_ah_train, y_ah_train, cv=10)
                    )
                else:
                    # address a certain attack
                    # get ad hoc testing samples
                    idx_test = [
                        index_ for index_, cand in enumerate(_y_test) if cand in cands
                    ]
                    # get X_ah_test, y_ah_test
                    X_ah_test = _X_test[idx_test, :]
                    y_ah_test = [_y_test[i] for i in idx_test]
                    # fit and score
                    pipeline.fit(X_ah_train, y_ah_train)
                    accs[f"{cand_size}_candidates"].append(
                        pipeline.score(X_ah_test, y_ah_test)
                    )

        return accs

    # runs exps
    accs = calculate_accuary(
        task, learner, runs, cand_sizes, X_train, train_label, X_test, test_label
    )
    if not os.path.isdir("results"):
        os.mkdir("results")
    json.dump(accs, open(f"results/{corpus}_{task}_{model}.json", "w"))

    for cand_size in cand_sizes:
        print(
            f"{str(cand_size)} authors: mean accuracy {np.round(np.mean(accs[f'{cand_size}_candidates'])* 100, 2)}"
            f"% (std. {np.round(np.std(accs[f'{cand_size}_candidates'])* 100, 2)}%)"
        )

    print(f"\nResults have been saved to 'results/{corpus}_{task}_{model}.json'")
    print("*" * 89)


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
        "-m",
        "--model",
        default="svm",
        help="model for the corresponding `corpus` and `task`, allows 'svm' (svm with writeprints-static features) "
        "and 'logistic_regression' (logistic regression with Koppel512 features) ",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="%(prog)s (version {version})".format(version=__version__),
    )
    args = parser.parse_args()

    # filters out warnings when candidate_size==5 in cross-validation
    warnings.filterwarnings(action="ignore", module="sklearn")
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
    # experiment counts, for cross-validation settings, the RUNS is divided by the count of folds
    RUNS = 1000
    # fixes seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    os.environ["PYTHONHASHSEED"] = str(42)

    # run exps
    train(args.corpus, args.task, args.model)
