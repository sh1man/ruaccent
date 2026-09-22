"""process_all / process_yo against stand-ins for the models: no downloads, no onnxruntime
sessions. The stand-ins split punctuation the way the real tokenizers do ("»," is two
words to them, one to split_by_words), which is what the position matching is for."""
import re

import pytest

from ruaccent import RUAccent

PUNCTUATION = r"[^\w\s]"
VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"


class Tagger:
    """A token-classification model: `label` for the words in `marked`, `other` otherwise."""

    def __init__(self, label, other, *marked):
        self.label, self.other, self.marked = label, other, set(marked)
        self.read = []

    def predict(self, text):
        self.read.append(text)
        return [
            {
                "entity": self.label if m.group().lower() in self.marked else self.other,
                "word": m.group(),
                "start": m.start(),
                "end": m.end(),
            }
            for m in re.finditer(r"\w+|[^\w\s]", text)
        ]

    predict_yo_homographs = predict_stress_usage = predict


class FirstVowel:
    def put_accent(self, word):
        return re.sub(f"([{VOWELS}])", r"+\1", word, count=1)


def accentizer(yo=(), unstressed=()):
    ru = RUAccent()
    ru.tiny_mode = False
    ru.omographs, ru.accents = {}, {}
    ru.yo_words = {"еще": "ещё", "левой": "лёвой"}
    ru.yo_homographs = {"все": "всё", "чем": "чём", "самое": "самоё"}
    ru.yo_homograph_model = Tagger("YO", "NO_YO", *yo)
    ru.stress_usage_predictor = Tagger("NO_STRESS", "STRESS", *unstressed)
    ru.accent_model = FirstVowel()
    return ru


def unmarked(text):
    return text.replace("+", "").replace("ё", "е").replace("Ё", "Е")


def test_with_skip_regex_the_models_still_read_whole_sentences():
    ru = accentizer()
    ru.process_all("Раньше, чем случилась беда. Потом, увы, все.", skip_regex=PUNCTUATION)
    assert ru.yo_homograph_model.read == ["раньше, чем случилась беда.", " потом, увы, все."]
    assert ru.stress_usage_predictor.read == ["Раньше, чем случилась беда.", " Потом, увы, все."]


def test_the_resolver_reads_text_without_yo():
    ru = accentizer(yo=["все"])
    assert ru.process_all("Ещё не все, ёлки!", skip_regex=PUNCTUATION) == "+Ещё не всё, +ёлки!"
    assert ru.yo_homograph_model.read == ["еще не все, елки!"]


@pytest.mark.parametrize(
    "text",
    [
        "О, «Гамлет» — это классика!",
        "Сказки: «Солдат и смерть», «Огниво».",
        "знать, что мы помогаем, — бесценно.",
        "Скажи 'привет' (тихо), ладно?",
        "два  пробела ,  и запятая",
        "  пробелы по краям  ",
        "кто-то — где-то… и № 5 ☺",
        "",
    ],
)
def test_with_skip_regex_the_text_only_gains_marks(text):
    out = accentizer(yo=["все"]).process_all(text, skip_regex=PUNCTUATION)
    assert unmarked(out) == unmarked(text)


def test_with_skip_regex_the_marks_are_those_of_plain_processing():
    text = "«Все», чем жили, — все еще тут. А все пришли?"
    marks = lambda out: re.findall(r"[\w+]+", out)  # noqa: E731
    protected = accentizer(yo=["все"], unstressed=["а"]).process_all(text, skip_regex=PUNCTUATION)
    plain = accentizer(yo=["все"], unstressed=["а"]).process_all(text)
    assert marks(protected) == marks(plain)
    assert protected == "«Всё», чем ж+или, — всё +ещё тут. А всё пр+ишли?"


def test_a_label_belongs_to_the_word_at_its_position():
    # "»," is one word here and two predictions: by index, "чем" would get the comma's label
    # and "жили" the one of "чем".
    ru = accentizer(yo=["чем"], unstressed=["жили"])
    assert ru.process_all("«Да», чем жили потом") == "«Да», чём жили п+отом"


def test_a_match_of_skip_regex_comes_back_untouched_and_is_still_context():
    ru = accentizer(yo=["все"])
    assert ru.process_all("Все или все", skip_regex=r"^\w+|или") == "Все или всё"
    assert ru.yo_homograph_model.read == ["все или все"]


def test_words_excluded_from_yo_keep_their_e():
    ru = accentizer(yo=["самое", "все"])
    ru._exclude_yo(["самое", "левой"])
    assert ru.process_all("Самое все левой") == "С+амое всё л+евой"


def test_process_yo_changes_nothing_but_the_yo():
    ru = accentizer(yo=["все"])
    assert ru.process_yo("Все хорошо.  И все тут!\n") == "Всё хорошо.  И всё тут!\n"


def test_koziev_is_looked_for_in_the_package_and_then_in_workdir(tmp_path):
    ru = RUAccent()
    ru.module_path, ru.workdir = str(tmp_path / "package"), str(tmp_path / "workdir")
    assert ru._koziev_root() == ru.workdir
    (tmp_path / "package" / "koziev").mkdir(parents=True)
    assert ru._koziev_root() == ru.module_path


def test_koziev_in_workdir_is_importable_as_part_of_the_package(tmp_path):
    probe = tmp_path / "koziev" / "probe"
    probe.mkdir(parents=True)
    (probe / "marker.py").write_text("WHERE = 'workdir'\n", encoding="utf-8")
    RUAccent()._add_to_package_path(str(tmp_path))

    from ruaccent.koziev.probe import marker

    assert marker.WHERE == "workdir"


def test_mark_polysyllables_overrules_the_predictor_except_for_one_syllable():
    ru = accentizer(unstressed=["золотой", "да", "или"])
    assert ru.process_all("Да, золотой или нет") == "Да, золотой или нет"
    ru.mark_polysyllables = True
    assert ru.process_all("Да, золотой или нет", skip_regex=PUNCTUATION) == "Да, з+олотой +или нет"


def test_a_yo_from_the_dictionary_survives_a_resolver_that_says_yo_on_it():
    ru = accentizer(yo=["еще", "все"])  # "еще" is in yo_words, not a homograph
    assert ru.process_all("еще и все") == "+ещё и всё"  # the fake accent model marks the first vowel
