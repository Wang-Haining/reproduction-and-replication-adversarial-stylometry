import os
import translators as ts
from shutil import rmtree


def create_backtranslation_examples(
    via_lang="de",
    path_to_corpus="defending-against-authorship-attribution-corpus/corpus",
):
    """
    Creates backtranslation examples from the control group with intermedium language(s).
    """
    # check input
    assert via_lang in ["ja", "de", "de_ja"], print(
        f"via_lang expects {['ja', 'de', 'de_ja']}."
    )

    # mkdir
    target_dir = os.path.join(path_to_corpus, "attacks_backtranslation_" + via_lang)
    if os.path.exists(target_dir):
        rmtree(target_dir)
    os.makedirs(target_dir)

    # read in control samples
    for f_name, f_path in [
        (f.name, f.path)
        for f in os.scandir(os.path.join(path_to_corpus, "attacks_control"))
        if not f.name.startswith(".")
    ]:
        raw = open(f_path, "r").read()
        if via_lang == "de_ja":
            trans = ts.google(
                ts.google(ts.google(raw, "en", "de"), "de", "ja"), "ja", "en"
            )
        else:
            trans = ts.google(ts.google(raw, "en", via_lang), via_lang, "en")
        open(os.path.join(target_dir, f_name), "w").write(trans)
    print(f"Directory {target_dir} has been populated with backtranslation examples.")


if __name__ == "__main__":
    print('Star populating backtranslation samples from the control group...')
    create_backtranslation_examples("ja")
    create_backtranslation_examples("de")
    create_backtranslation_examples("de_ja")
    print('All done! 🎉✨ ')
