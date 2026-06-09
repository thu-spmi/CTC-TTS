"""
Implementation of tokenizer
"""
import io
import os
import pickle
import re
import uuid
from collections import OrderedDict
from typing import Dict, Iterable, List, Literal, Optional, Union

import sentencepiece as sp


def randstr() -> str:
    return str(uuid.uuid4())


def gen_cache_path() -> str:
    return os.path.join("/tmp", randstr())


def file2bin(file_path: str) -> bytes:
    assert os.path.isfile(file_path), f"no such file: '{file_path}'"
    with open(file_path, "rb") as file_obj:
        return file_obj.read()


def bin2file(binary_data: bytes, target_path: Optional[str] = None) -> str:
    if target_path is None:
        target_path = gen_cache_path()
    with open(target_path, "wb") as file_obj:
        file_obj.write(binary_data)
    return target_path


class AbsTokenizer:
    def encode(self, strings: Union[str, Iterable[str]]) -> Union[List[int], List[List[int]]]:
        if isinstance(strings, str):
            return self._enc(strings)
        try:
            iterator = iter(strings)
        except TypeError as exc:
            raise RuntimeError(
                f"{self.__class__.__name__}.encode: input is neither str nor iterable."
            ) from exc
        return [self._enc(text) for text in iterator]

    def decode(self, indices: Union[Iterable[int], Iterable[Iterable[int]]]) -> Union[str, List[str]]:
        try:
            items = list(indices)
        except TypeError as exc:
            raise RuntimeError(
                f"{self.__class__.__name__}.decode(): input is not iterable."
            ) from exc
        if not items:
            return ""
        if isinstance(items[0], int):
            return self._dec(items)
        return [self._dec(item) for item in items]

    def _enc(self, text: str) -> List[int]:
        raise NotImplementedError

    def _dec(self, indices: Iterable[int]) -> str:
        raise NotImplementedError

    @staticmethod
    def train(*args, **kwargs):
        return

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def dump_vocab(self, fileio: Optional[Union[str, io.TextIOWrapper]] = None) -> Optional[Dict[int, str]]:
        vocab = self._vocab_to_dict()
        if fileio is None:
            return vocab
        if isinstance(fileio, str):
            with open(fileio, "w", encoding="utf-8") as file_obj:
                for idx, token in vocab.items():
                    file_obj.write(f"{token} {idx}\n")
            return None
        if isinstance(fileio, io.TextIOWrapper):
            for idx, token in vocab.items():
                fileio.write(f"{token} {idx}\n")
            return None
        raise ValueError(
            f"Unsupported file object type: {type(fileio)}, expected str or TextIOWrapper."
        )

    def _vocab_to_dict(self) -> Dict[int, str]:
        raise NotImplementedError

    def state_dict(self) -> OrderedDict:
        raise NotImplementedError

    def load_state_dict(self, state_dict: OrderedDict):
        raise NotImplementedError

    def __getstate__(self):
        return self.state_dict()

    def __setstate__(self, state: OrderedDict):
        self.load_state_dict(state)


class SimpleTokenizer(AbsTokenizer):
    """Passing a file, a list of words, or a word-to-index mapping to build a simple tokenizer.

    When passed a file, assume one word per line, 
    if `read_index_from_file=False`, use the first one as word, and the lineid+1 as token id;
    if `read_index_from_file=True`, take first column as words, second column as token id.
    """

    def __init__(self, dmap: Union[str, List[str], Dict[str, int]], read_index_from_file: bool = False) -> None:
        super().__init__()
        self._r_vocab = OrderedDict([('<s>', 0), ('<unk>', 1)])
        if isinstance(dmap, str) and os.path.isfile(dmap):
            if read_index_from_file:
                consturctd = {}
                with open(dmap, 'r') as fi:
                    for line in fi:
                        line = line.strip()
                        if line == '':
                            continue
                        w, i = line.split(maxsplit=2)[:2]
                        consturctd[w] = i
            else:
                consturctd = []
                with open(dmap, 'r') as fi:
                    for line in fi:
                        line = line.strip()
                        if line == '':
                            continue
                        consturctd.append(line.split(maxsplit=1)[0])
            dmap = consturctd

        if isinstance(dmap, list):
            offset = 2
            for c in dmap:
                if c in self._r_vocab:
                    continue
                self._r_vocab[c] = offset
                offset += 1
        elif isinstance(dmap, dict):
            if '<s>' in dmap:
                del dmap['<s>']
            if '<unk>' in dmap:
                del dmap['<unk>']
            self._r_vocab.update(dmap)
        else:
            raise ValueError(str(type(dmap)))
        self._i_vocab = [c for c in self._r_vocab.keys()]

    @property
    def vocab_size(self) -> int:
        return len(self._r_vocab)

    def _enc(self, s: str) -> List[int]:
        return [self._r_vocab.get(c, 1) for c in s.split()]

    def _dec(self, indices: Iterable[int]) -> str:
        return ' '.join(self._i_vocab[i] for i in indices)

    def _vocab_to_dict(self) -> Dict[int, str]:
        return {i: c for i, c in enumerate(self._i_vocab)}

    def state_dict(self) -> OrderedDict:
        return self._r_vocab

    def load_state_dict(self, state_dict: OrderedDict):
        self._r_vocab = state_dict
        self._i_vocab = [c for c in self._r_vocab.keys()]


class LexiconTokenizer(AbsTokenizer):
    def __init__(
        self,
        lexicon: Optional[str] = None,
        add_special_token: bool = True,
        bos_interface: str = "<s>",
        unk_interface: str = "<unk>",
    ) -> None:
        self._bos_token = bos_interface
        self._unk_token = unk_interface
        self._w2pid: OrderedDict[str, tuple] = OrderedDict()
        self._units: OrderedDict[str, int] = OrderedDict()
        self._id2unit: Dict[int, str] = {}
        if lexicon is not None:
            assert os.path.isfile(lexicon), f"given w2p_map='{lexicon}' not exist."
            self.init_lexicon(lexicon, add_special_token)

    def init_lexicon(self, mapping_path: str, add_special_token: bool):
        lexicon = OrderedDict()
        with open(mapping_path, "r", encoding="utf-8") as file_obj:
            for line in file_obj:
                utt = re.sub(r"\s+", " ", line).strip().split()
                if not utt:
                    continue
                if utt[0] not in lexicon:
                    lexicon[utt[0]] = utt[1:]

        units = OrderedDict()
        word2unitid = OrderedDict()
        if add_special_token:
            units[self._bos_token] = 0
            units[self._unk_token] = 1
            word2unitid[self._bos_token] = (0,)
            word2unitid[self._unk_token] = (1,)

        raw_units = []
        seen_units = set(units)
        for unit_list in lexicon.values():
            for unit in unit_list:
                if unit not in seen_units:
                    seen_units.add(unit)
                    raw_units.append(unit)

        start_index = len(units)
        for offset, unit in enumerate(raw_units):
            units[unit] = start_index + offset

        for word, unit_list in lexicon.items():
            word2unitid[word] = tuple(units[unit] for unit in unit_list)

        self._w2pid = word2unitid
        if not add_special_token and (
            self._bos_token not in self._w2pid or self._unk_token not in self._w2pid
        ):
            raise RuntimeError(
                f"{self._bos_token} and(or) {self._unk_token} are not found in '{mapping_path}'."
            )

        self._units = units
        self._id2unit = {idx: unit for unit, idx in self._units.items()}

    @property
    def vocab_size(self) -> int:
        return len(self._units)

    def _vocab_to_dict(self) -> Dict[int, str]:
        return dict(self._id2unit)

    def state_dict(self) -> OrderedDict:
        return OrderedDict(
            [
                ("w2pid", self._w2pid),
                ("units", self._units),
                ("bos", self._bos_token),
                ("unk", self._unk_token),
            ]
        )

    def load_state_dict(self, state_dict: OrderedDict):
        # Support two serialization formats:
        # 1. Current format：{"w2pid", "units", "bos", "unk"}
        # 2. Legacy pickle format that stores instance attributes directly：{"_w2pid", "_units", "_bos_token", "_unk_token"}
        w2pid = state_dict.get("w2pid", state_dict.get("_w2pid"))
        units = state_dict.get("units", state_dict.get("_units"))
        bos_token = state_dict.get("bos", state_dict.get("_bos_token"))
        unk_token = state_dict.get("unk", state_dict.get("_unk_token"))

        if w2pid is None or units is None or bos_token is None or unk_token is None:
            raise KeyError(
                "LexiconTokenizer state_dict is missing required keys. "
                "Expected current keys {'w2pid','units','bos','unk'} or legacy keys "
                "{'_w2pid','_units','_bos_token','_unk_token'}."
            )

        self._w2pid = OrderedDict(w2pid)
        self._units = OrderedDict(units)
        self._bos_token = bos_token
        self._unk_token = unk_token
        self._id2unit = {idx: unit for unit, idx in self._units.items()}

    def update_from_lexicon(
        self,
        mapping_path: str,
        overwrite_existing_words: bool = True,
    ) -> Dict[str, int]:
        """
        Incrementally update the current LexiconTokenizer from a pronunciation lexicon.

        Args:
            mapping_path:
                Path to the pronunciation lexicon. Each line should be formatted as
                `word phoneme1 phoneme2 ...` or `word<TAB>phoneme1 phoneme2 ...`.
            overwrite_existing_words:
                Whether to overwrite existing word mappings with pronunciations from the lexicon.

        Returns:
            A dictionary containing update statistics:
            - `added_units`: Number of newly added phonemes
            - `added_words`: Number of newly added words
            - `updated_words`: Number of existing words overwritten
            - `total_units`: Total number of phonemes after the update
            - `total_words`: Total number of entries after the update
        """
        if not os.path.isfile(mapping_path):
            raise FileNotFoundError(f"given w2p_map='{mapping_path}' not exist.")

        lexicon = OrderedDict()
        with open(mapping_path, "r", encoding="utf-8") as file_obj:
            for line in file_obj:
                utt = re.sub(r"\s+", " ", line).strip().split()
                if len(utt) < 2:
                    continue
                if utt[0] not in lexicon:
                    lexicon[utt[0]] = utt[1:]

        added_units = 0
        added_words = 0
        updated_words = 0

        for word, unit_list in lexicon.items():
            token_ids = []
            for unit in unit_list:
                if unit not in self._units:
                    self._units[unit] = len(self._units)
                    added_units += 1
                token_ids.append(self._units[unit])

            word_exists = word in self._w2pid
            if word_exists and not overwrite_existing_words:
                continue

            self._w2pid[word] = tuple(token_ids)
            if word_exists:
                updated_words += 1
            else:
                added_words += 1

        self._id2unit = {idx: unit for unit, idx in self._units.items()}
        return {
            "added_units": added_units,
            "added_words": added_words,
            "updated_words": updated_words,
            "total_units": len(self._units),
            "total_words": len(self._w2pid),
        }

    def _enc(self, text: str) -> List[int]:
        unk_ids = self._w2pid[self._unk_token]
        words = text.split()
        return [idx for word in words for idx in self._w2pid.get(word, unk_ids)]

    def _dec(self, indices: Iterable[int]) -> str:
        return " ".join(self._id2unit[idx] for idx in indices)


class SentencePieceTokenizer(AbsTokenizer):
    def __init__(self, model_file: Optional[str] = None) -> None:
        self._tokenizer: Optional[sp.SentencePieceProcessor] = None
        self.byte_model: Optional[bytes] = None
        if model_file is not None:
            self._load_model_file(model_file)

    def _load_model_file(self, model_file: str):
        if not os.path.isfile(model_file):
            raise RuntimeError(
                f"{self.__class__.__name__}: sentencepiece model path '{model_file}' is invalid."
            )
        self._tokenizer = sp.SentencePieceProcessor(model_file=model_file)
        self.byte_model = file2bin(model_file)

    def _ensure_loaded(self):
        if self._tokenizer is None:
            raise RuntimeError(f"{self.__class__.__name__} has not loaded a model.")

    def encode(self, strings: Union[str, Iterable[str]]) -> Union[List[int], List[List[int]]]:
        self._ensure_loaded()
        return self._tokenizer.Encode(strings)

    def decode(self, idx_tokens: Union[List[int], List[List[int]]]) -> Union[str, List[str]]:
        self._ensure_loaded()
        return self._tokenizer.Decode(idx_tokens)

    @property
    def vocab_size(self) -> int:
        self._ensure_loaded()
        return self._tokenizer.vocab_size()

    def _vocab_to_dict(self) -> Dict[int, str]:
        self._ensure_loaded()
        return {idx: self._tokenizer.IdToPiece(idx) for idx in range(self.vocab_size)}

    def state_dict(self) -> OrderedDict:
        if self.byte_model is None:
            raise RuntimeError(f"{self.__class__.__name__} has not loaded a model.")
        return OrderedDict([("model-data", self.byte_model)])

    def load_state_dict(self, state_dict: OrderedDict):
        self.byte_model = state_dict["model-data"]
        cachefile = bin2file(self.byte_model)
        try:
            self._tokenizer = sp.SentencePieceProcessor(model_file=cachefile)
        finally:
            os.remove(cachefile)

    def _enc(self, text: str) -> List[int]:
        return self.encode(text)

    def _dec(self, indices: Iterable[int]) -> str:
        return self.decode(list(indices))

    @staticmethod
    def train(
        f_text: str,
        model_prefix: str,
        vocab_size: int,
        add_dummy_prefix: bool = True,
        character_coverage: float = 0.9995,
        model_type: Literal["unigram", "bpe", "word", "char"] = "unigram",
        use_all_vocab: bool = False,
        bos_id: int = 0,
        unk_id: int = 1,
        eos_id: int = -1,
        unk_surface: str = "<unk>",
        minloglevel: int = 1,
        user_defined_symbols: Union[str, List[str]] = "",
        train_extremely_large_corpus: bool = False,
        **options,
    ):
        assert os.path.isfile(f_text), f"Given text file '{f_text}' not found."
        if isinstance(user_defined_symbols, str) and user_defined_symbols and os.path.isfile(user_defined_symbols):
            with open(user_defined_symbols, "r", encoding="utf-8") as file_obj:
                user_defined_symbols = [line.strip() for line in file_obj if line.strip()]

        output_dir = os.path.dirname(model_prefix)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        sp.SentencePieceTrainer.Train(
            input=f_text,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            add_dummy_prefix=add_dummy_prefix,
            character_coverage=character_coverage,
            model_type=model_type,
            use_all_vocab=use_all_vocab,
            bos_id=bos_id,
            unk_id=unk_id,
            eos_id=eos_id,
            unk_surface=unk_surface,
            minloglevel=minloglevel,
            user_defined_symbols=user_defined_symbols,
            train_extremely_large_corpus=train_extremely_large_corpus,
            **options,
        )


TOKENIZER_REGISTRY = {
    "SimpleTokenizer": SimpleTokenizer,
    "LexiconTokenizer": LexiconTokenizer,
    "SentencePieceTokenizer": SentencePieceTokenizer,
}

class _CompatTokenizerUnpickler(pickle.Unpickler):
    """Custom Unpickler for backward-compatible tokenizer loading"""

    # Mapping table from legacy (module, class) names to current class objects
    _CLASS_MAP = {
        ("cat.shared.tokenizer", "AbsTokenizer"): AbsTokenizer,
        ("cat.shared.tokenizer", "SimpleTokenizer"): SimpleTokenizer,
        ("cat.shared.tokenizer", "LexiconTokenizer"): LexiconTokenizer,
        ("cat.shared.tokenizer", "SentencePieceTokenizer"): SentencePieceTokenizer,
    }

    def find_class(self, module: str, name: str):
        mapped_class = self._CLASS_MAP.get((module, name))
        if mapped_class is not None:
            return mapped_class
        if module == "cat.shared.tokenizer":
            current_symbol = globals().get(name)
            if current_symbol is not None:
                return current_symbol
        return super().find_class(module, name)


def initialize(cfg: Dict) -> AbsTokenizer:
    assert "type" in cfg, "'type' is missing in tokenizer initialization."
    tokenizer_type = cfg["type"]
    if tokenizer_type not in TOKENIZER_REGISTRY:
        raise NameError(
            f"'{tokenizer_type}' is not defined. Available tokenizers: {', '.join(TOKENIZER_REGISTRY)}."
        )

    tokenizer_cls = TOKENIZER_REGISTRY[tokenizer_type]
    if "option-train" in cfg:
        tokenizer_cls.train(**cfg["option-train"])

    init_opts = dict(cfg.get("option-init", {}))
    if tokenizer_cls is SentencePieceTokenizer and "model_file" not in init_opts:
        model_prefix = cfg.get("option-train", {}).get("model_prefix")
        if model_prefix is None:
            raise ValueError(
                f"For {SentencePieceTokenizer.__name__}, set 'option-init.model_file' or 'option-train.model_prefix'."
            )
        init_opts["model_file"] = f"{model_prefix}.model"

    return tokenizer_cls(**init_opts)


def save(obj: AbsTokenizer, target: str):
    assert isinstance(obj, AbsTokenizer), "tokenizer.save: input object is not AbsTokenizer instance."
    with open(target, "wb") as file_obj:
        pickle.dump(obj, file_obj)



def load_tknz(src: str) -> AbsTokenizer:
    assert os.path.isfile(src), f"no such tokenizer file: '{src}'"
    with open(src, "rb") as file_obj:
        try:
            tokenizer = pickle.load(file_obj)
        except ModuleNotFoundError as exc:
            if exc.name != "cat":
                raise
            file_obj.seek(0)
            tokenizer = _CompatTokenizerUnpickler(file_obj).load()
    assert isinstance(tokenizer, AbsTokenizer), "tokenizer.load_tknz: loaded object is not AbsTokenizer instance."
    return tokenizer


def load(src: str) -> AbsTokenizer:
    return load_tknz(src)
