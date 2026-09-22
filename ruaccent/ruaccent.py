import json
import pathlib
import sys
from huggingface_hub import HfFileSystem, hf_hub_download
import os
import gzip
from os.path import join as join_path
from .omograph_model import OmographModel
from .accent_model import AccentModel
from .stress_usage_model import StressUsagePredictorModel
from .yo_homograph_model import YoHomographModel
from .text_preprocessor import TextPreprocessor
from .text_postprocessor import fix_capital
import re


# Words that keep their е although the dictionaries know a ё twin: an archaism ("самоё",
# "далёко"), a rarity ("нёбо", "лёт", "перёд", "маркёр", "вселённой"), a name ("Лёвой",
# "Королёва") - or a present tense the resolver always prefers to the far commoner future
# ("узна́ем"). The resolver picks the twin even with the whole sentence to go by ("самое
# время" -> "самоё" 63% of the time, "небо" 50%, "узнаем" 100%), and in 50 hours of
# transcribed speech none of these was said with a ё but "узнаём", 5 times in 22.
# load(yo_exclude=...) replaces the list.
DEFAULT_YO_EXCLUDE = (
    "самое", "далеко", "недалеко",
    "небо", "неба", "небу", "небом", "небе",
    "лет", "лета", "лету", "летом", "лете",
    "узнаем", "узнаете",
    "шлем", "перед", "лень", "маркер", "берег", "отсек", "левой", "чел",
    "королева", "королеву", "королеве", "королевой",
    "вселенной", "нулевых",
)


class RUAccent:
    def __init__(self):
        self.omograph_model = OmographModel()
        self.accent_model = AccentModel()
        self.stress_usage_predictor = StressUsagePredictorModel()
        self.yo_homograph_model = YoHomographModel()
        self.fs = HfFileSystem()
        self.normalize = re.compile(r"[^a-zA-Z0-9\sа-яА-ЯёЁ—.,!?:;""''(){}\[\]«»„“”-]")
        self.omograph_models_paths = {'big_poetry': '/nn/nn_omograph/big_poetry', 
                                      'medium_poetry': '/nn/nn_omograph/medium_poetry', 
                                      'small_poetry': '/nn/nn_omograph/small_poetry',
                                      'turbo': '/nn/nn_omograph/turbo',
                                      'turbo2': '/nn/nn_omograph/turbo2',
                                      'turbo3': '/nn/nn_omograph/turbo3',
                                      'turbo3.1': '/nn/nn_omograph/turbo3.1',
                                      'tiny': '/nn/nn_omograph/tiny',
                                      'tiny2': '/nn/nn_omograph/tiny2',
                                      'tiny2.1': '/nn/nn_omograph/tiny2.1',

                                      }
    
        self.accentuator_paths = ['/nn/nn_accent', '/nn/nn_stress_usage_predictor','/nn/nn_yo_homograph_resolver', '/dictionary', '/dictionary/rule_engine']
        self.letters_accent = {'о': '+о', 'О': '+О'}
        self.koziev_paths = ["/koziev/rulemma", "/koziev/rupostagger", "/koziev/rupostagger/database"]
        self.tiny_mode = False
        self.mark_polysyllables = False
        
    def load(
        self,
        omograph_model_size="turbo2",
        use_dictionary=False,
        custom_dict={},
        custom_homographs={},
        device="CPU",
        repo="ruaccent/accentuator",
        workdir=None,
        tiny_mode=False,
        yo_exclude=None,
        mark_polysyllables=False
        ):
        self.tiny_mode = tiny_mode
        self.mark_polysyllables = mark_polysyllables
        if workdir:
            self.workdir = workdir
        else:
            self.workdir = str(pathlib.Path(__file__).resolve().parent)
        self.module_path = str(pathlib.Path(__file__).resolve().parent)
        self.custom_dict = custom_dict
        self.accents = {}

        if not os.path.exists(
            join_path(self.workdir, "dictionary")
        ):
            for path in self.accentuator_paths:
                files = self.fs.ls(repo + path)
                for file in files:
                    if file["type"] == "file":
                        hf_hub_download(repo_id=repo, local_dir_use_symlinks=False, local_dir=self.workdir, filename=file['name'].replace(repo+'/', ''))
    
        if not os.path.exists(join_path(self.workdir, "nn")):
            os.mkdir(join_path(self.workdir, "nn"))
        
        if not os.path.exists(join_path(self.workdir, "nn", "nn_omograph", omograph_model_size)):
            model_path = self.omograph_models_paths.get(omograph_model_size, None)
            if model_path:
                files = self.fs.ls(repo + model_path)
                for file in files:
                    if file["type"] == "file":
                        hf_hub_download(repo_id=repo, local_dir_use_symlinks=False, local_dir=self.workdir, filename=file['name'].replace(repo+'/', ''))
        if not self.tiny_mode:
            koziev_root = self._koziev_root()
            if not os.path.exists(join_path(koziev_root, "koziev")):
                for path in self.koziev_paths:
                    files = self.fs.ls(repo + path)
                    for file in files:
                        if file["type"] == "file":
                            hf_hub_download(repo_id=repo, local_dir_use_symlinks=False, local_dir=koziev_root, filename=file['name'].replace(repo+'/', ''))
            self._add_to_package_path(koziev_root)

        self.omographs = json.load(
            gzip.open(join_path(self.workdir, "dictionary","omographs.json.gz"))
        )
        self.omographs.update({"коса": ["к+оса", "кос+а"]})
        self.omographs.update(custom_homographs)
        self.omograph_model.load(join_path(self.workdir, self.omograph_models_paths[omograph_model_size][1:]), device=device)

        self.yo_words = json.load(
            gzip.open(join_path(self.workdir, "dictionary","yo_words.json.gz"))
        ) 
        self.accent_model.load(join_path(self.workdir, "nn","nn_accent/"), device=device)
        self.yo_homographs = json.load(
                gzip.open(join_path(self.workdir, "dictionary","yo_homographs.json.gz"))
            ) 
        self.yo_homograph_model.load(join_path(self.workdir, "nn","nn_yo_homograph_resolver"), device=device)
        self._exclude_yo(DEFAULT_YO_EXCLUDE if yo_exclude is None else yo_exclude)

        if self.tiny_mode or not use_dictionary:
            self.accents.update(json.load(
                gzip.open(join_path(self.workdir, "dictionary","accents_nn.json.gz"))
            ))
        else:
            self.accents.update(json.load(
                gzip.open(join_path(self.workdir, "dictionary","accents.json.gz"))
            ))


        self.accents.update(self.custom_dict)
        self.accents.update(self.letters_accent)


        if not self.tiny_mode:
            from .rule_accent_engine import RuleEngine
            self.rule_accent = RuleEngine()

            self.stress_usage_predictor.load(join_path(self.workdir, "nn","nn_stress_usage_predictor/"), device=device)
            self.rule_accent.load(join_path(self.workdir, "dictionary","rule_engine"))


    def _koziev_root(self):
        """The directory whose `koziev` folder holds Koziev's tagger and lemmatizer.

        190 MB of code and data that come from the hub, not from the wheel. They used to
        go into the installed package, always: every environment downloaded its own copy,
        and the package directory had to be writable (not so in a container that runs as
        another user than the one that installed it). Now only a copy that is already in
        the package is used from there; otherwise they live in `workdir` with the models
        and the dictionaries, shared by every environment that names the same `workdir`.
        """
        if os.path.exists(join_path(self.module_path, "koziev")):
            return self.module_path
        return self.workdir

    def _add_to_package_path(self, directory):
        # rule_accent_engine imports `.koziev...`: a directory on the package's __path__
        # is searched for subpackages like the package's own.
        package_path = sys.modules[__package__].__path__
        if directory not in package_path:
            package_path.append(directory)

    def _exclude_yo(self, words):
        for word in words:
            self.yo_words.pop(word, None)
            self.yo_homographs.pop(word, None)

    def count_vowels(self, text):
        vowels = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
        return sum(1 for char in text if char in vowels)

    def has_punctuation(self, text):
        for char in text:
            if char in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~":
                return True
        return False

    def delete_spaces_before_punc(self, text):
        punc = "!\"#$%&'()*,./:;<=>?@[\\]^_`{|}-"
        for char in punc:
            if char == '-':
                text = text.replace(" " + char, char).replace(char + " ", char)
            text = text.replace(" " + char, char)
        return text.replace('~', '-')

    def extract_entities(self, data):
        entities = []
        for item in data:
            entity = item['entity']
            entities.append(entity)
        return entities

    def _labels(self, predictions, words, spans, default):
        """A model's label for each word, found by the word's position in the sentence.

        The n-th prediction is not the n-th word: the models' tokenizers make two words of
        "»," where split_by_words makes one, and every word after it got its neighbour's
        label (2.5% of the sentences of a speech corpus). `default` is for a word the
        model has nothing on.
        """
        if spans is None or len(spans) != len(words):
            labels = self.extract_entities(predictions)[:len(words)]
            return labels + [default] * (len(words) - len(labels))
        by_start = {p.get("start"): p["entity"] for p in predictions}
        return [by_start.get(start, default) for start, _ in spans]

    def _process_yo(self, words, sentence, spans=None):
        if spans is None:
            spans = TextPreprocessor.word_spans(sentence)
        # The resolver was trained on text without ё and is less sure of itself on text
        # that has some: 14.7% of the е it should have left alone became ё, 11.7% without.
        lower_sentence = sentence.lower().replace('ё', 'е')
        if len(lower_sentence) != len(sentence):
            spans = None

        yo_predictions = None
        if 'е' in lower_sentence:
            yo_predictions = self._labels(self.yo_homograph_model.predict_yo_homographs(lower_sentence), words, spans, "NO_YO")
        
        for i, word in enumerate(words):
            lower_word = word.lower()
            words[i] = fix_capital(word, self.yo_words.get(lower_word, word))
            # Only a homograph is the resolver's to decide: a YO on any other word used to
            # put the word back as written, undoing the ё the dictionary had just given it.
            if yo_predictions and yo_predictions[i] == "YO" and lower_word in self.yo_homographs:
                words[i] = fix_capital(word, self.yo_homographs[lower_word])
        return words


    def _process_omographs(self, text):
        splitted_text = text
    
        founded_omographs = []
        texts = []
        hypotheses = []
    
        for i, word in enumerate(splitted_text):
            variants = self.omographs.get(word)
            if variants:
                founded_omographs.append(
                    {"word": word, "variants": variants, "position": i}
                )
                texts.append(splitted_text)
                hypotheses.append(variants)
    
        if len(founded_omographs) > 0:
            texts_batch = []
            hypotheses_batch = [val for sublist in hypotheses for val in sublist]
            num_hypotheses = [len(i) for i in hypotheses]
            
            for o, t in zip(founded_omographs, texts):
                position = o["position"]
                t_back = t[position]
                t[position] = ' <w>' + t[position] + '</w> '
                for _ in range(len(o["variants"])):
                    texts_batch.append(self.delete_spaces_before_punc(" ".join(t.copy())))
                t[position] = t_back
            cls_batch = self.omograph_model.classify(texts_batch, hypotheses_batch, num_hypotheses)
            for cls_index, omograph in enumerate(founded_omographs):
                position = omograph["position"]
                splitted_text[position] = cls_batch[cls_index]
        return splitted_text


    
    def _process_accent(self, text, stress_usages):
        splitted_text = text
        for i, word in enumerate(splitted_text):
            if '+' in word:
                continue
            if stress_usages[i] == "STRESS":
                lower_word = word.lower()
                stressed_word = self.accents.get(lower_word, lower_word)
                if stressed_word == lower_word and not self.has_punctuation(lower_word) and self.count_vowels(lower_word) > 1:
                    splitted_text[i] = self.accent_model.put_accent(word)
                else:
                    match = re.finditer(r'\+', stressed_word)
                    word_fixed = list(word)
                    for j, e in enumerate(list(match)):
                        word_fixed = word_fixed[:e.start() + j] + ["+"] + list(word)[e.end() - 1:]
                    splitted_text[i] = "".join(word_fixed)
        return splitted_text

        
    def process_yo(self, text):
        sentences = TextPreprocessor.split_by_sentences(text)
        outputs = []
        for sentence in sentences:
            words, remaining_text = TextPreprocessor.split_by_words(sentence)
            processed_words = self._process_yo(words, sentence)
            processed_text = "".join([l+r for l,r in zip(remaining_text, processed_words)] + [remaining_text[-1]])
            processed_text = self.delete_spaces_before_punc(processed_text)
            outputs.append(processed_text)
        return "".join(outputs)
    

    def _process_words(self, sentence):
        """split_by_words(sentence), the words' positions and the words with ё and stress."""
        words, remaining_text = TextPreprocessor.split_by_words(sentence)
        if len(words) == 0:
            return words, remaining_text, [], []
        spans = TextPreprocessor.word_spans(sentence)
        if self.tiny_mode:
            stress_usages = ["STRESS"] * len(words)
        else:
            stress_usages = self._labels(self.stress_usage_predictor.predict_stress_usage(sentence), words, spans, "STRESS")
            if self.mark_polysyllables:
                # The predictor answers "would a reader need the mark here", and leaves some
                # content words without one ("'Золотой ключик'"). A speech synthesizer needs
                # it on every word that has a choice: without, "Золотой" came out stressed
                # elsewhere in 4 generations of 8. Words of one syllable stay the predictor's.
                stress_usages = ["STRESS" if self.count_vowels(word) > 1 else usage for word, usage in zip(words, stress_usages)]
        processed_words = self._process_yo(list(words), sentence, spans)
        processed_words = self._process_omographs(processed_words)
        processed_words = self._process_accent(processed_words, stress_usages)
        return words, remaining_text, spans, processed_words

    def process_all_internal(self, text):
        text = re.sub(self.normalize, "", text)
        sentences = TextPreprocessor.split_by_sentences(text)
        outputs = []
        for sentence in sentences:
            words, remaining_text, _, processed_words = self._process_words(sentence)
            if len(words) == 0:
                outputs.append("".join(remaining_text))
                continue
            processed_sentence = "".join([l+r for l,r in zip(remaining_text, processed_words)] + [remaining_text[-1]])
            processed_sentence = self.delete_spaces_before_punc(processed_sentence)
            
            outputs.append(processed_sentence)
        return "".join(outputs)

    def process_all(self, text, skip_regex=None):
        """Put ё and stress marks into `text`.

        With `skip_regex`, what it matches comes back untouched - and so does everything
        that is not a word: the text keeps its punctuation and spacing, and only gains
        "+" and ё. The models still read whole sentences, matches included. (They used to
        get the stretches between the matches one at a time: with punctuation protected,
        "раньше, чем случилась трагедия" was "раньше" and "чем случилась трагедия" to
        them, and the second came back as "чём". A stretch of nothing but whitespace
        between two matches was lost, too.)
        """
        if not skip_regex:
            return self.process_all_internal(text)
        skipped = [(m.start(), m.end()) for m in re.finditer(skip_regex, text) if m.end() > m.start()]
        if len(skipped) == 0:
            return self.process_all_internal(text)

        sentences = TextPreprocessor.split_by_sentences(text)
        if "".join(sentences) != text:
            sentences = [text]
        outputs = []
        offset = 0
        next_skipped = 0
        for sentence in sentences:
            # What process_all_internal deletes is blanked out instead, for the models
            # only: positions in `readable` are positions in `sentence`.
            readable = re.sub(self.normalize, " ", sentence)
            words, _, spans, processed_words = self._process_words(readable)
            if len(readable) != len(sentence) or len(spans) != len(words):
                words, spans, processed_words = [], [], []
            position = 0
            for word, (start, end), processed_word in zip(words, spans, processed_words):
                while next_skipped < len(skipped) and skipped[next_skipped][1] <= offset + start:
                    next_skipped += 1
                protected = next_skipped < len(skipped) and skipped[next_skipped][0] < offset + end
                outputs.append(sentence[position:start])
                if protected or not any(ch.isalpha() for ch in word):
                    outputs.append(sentence[start:end])
                else:
                    outputs.append(processed_word)
                position = end
            outputs.append(sentence[position:])
            offset += len(sentence)
        return "".join(outputs)


