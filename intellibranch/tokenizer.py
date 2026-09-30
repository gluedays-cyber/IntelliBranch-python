from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass
class MergeRule:
    token1: int
    token2: int
    target: int


class BPETokenizer:
    """BPETokenizer implements subword segmentation via iterative byte-pair merge rules."""

    def __init__(self, vocab: Sequence[str], rules: Sequence[MergeRule]) -> None:
        self.vocab: List[str] = list(vocab)
        self.vocab_map: Dict[str, int] = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.merge_rules: List[MergeRule] = list(rules)
        self.rule_lookup: Dict[Tuple[int, int], int] = {
            (r.token1, r.token2): r.target for r in self.merge_rules
        }

    @classmethod
    def train(cls, corpus: Sequence[str], target_vocab_size: int = 200) -> "BPETokenizer":
        """Learns a BPE vocabulary and merge rules from a raw text corpus."""
        if target_vocab_size < 10:
            target_vocab_size = 10

        vocab: List[str] = ["[PAD]", "[UNK]"]
        vocab_map: Dict[str, int] = {tok: idx for idx, tok in enumerate(vocab)}

        # 1. Collect initial unique characters
        for text in corpus:
            cleaned = text.strip().lower()
            for ch in cleaned:
                if ch not in vocab_map:
                    vocab_map[ch] = len(vocab)
                    vocab.append(ch)

        # 2. Tokenize corpus into character token ID sequences
        tokenized_corpus: List[List[int]] = []
        for text in corpus:
            cleaned = text.strip().lower()
            if not cleaned:
                continue
            seq = [vocab_map[ch] for ch in cleaned if ch in vocab_map]
            if seq:
                tokenized_corpus.append(seq)

        merge_rules: List[MergeRule] = []
        rule_lookup: Dict[Tuple[int, int], int] = {}

        # 3. Iteratively merge most frequent adjacent pairs
        while len(vocab) < target_vocab_size:
            pair_counts: Dict[Tuple[int, int], int] = {}
            for seq in tokenized_corpus:
                for i in range(len(seq) - 1):
                    pair = (seq[i], seq[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            best_pair: Tuple[int, int] | None = None
            max_freq: int = -1
            for pair, freq in pair_counts.items():
                if freq > max_freq:
                    max_freq = freq
                    best_pair = pair

            if best_pair is None or (max_freq < 2 and len(vocab) >= target_vocab_size // 2):
                break

            t1, t2 = best_pair
            str1 = vocab[t1]
            str2 = vocab[t2]
            merged_str = str1 + str2

            new_id = len(vocab)
            vocab_map[merged_str] = new_id
            vocab.append(merged_str)

            rule = MergeRule(token1=t1, token2=t2, target=new_id)
            merge_rules.append(rule)
            rule_lookup[best_pair] = new_id

            # Apply merge in-place across tokenized corpus
            new_corpus: List[List[int]] = []
            for seq in tokenized_corpus:
                new_seq: List[int] = []
                i = 0
                while i < len(seq):
                    if i < len(seq) - 1 and seq[i] == t1 and seq[i + 1] == t2:
                        new_seq.append(new_id)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                new_corpus.append(new_seq)
            tokenized_corpus = new_corpus

        return cls(vocab=vocab, rules=merge_rules)

    def encode(self, text: str) -> List[int]:
        """Converts input text into subword token IDs using the learned merge rules."""
        cleaned = text.strip().lower()
        if not cleaned:
            return []

        unk_id = self.vocab_map.get("[UNK]", 1)
        tokens: List[int] = [self.vocab_map.get(ch, unk_id) for ch in cleaned]

        if len(tokens) <= 1:
            return tokens

        while True:
            merged = False
            next_tokens: List[int] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1:
                    pair = (tokens[i], tokens[i + 1])
                    if pair in self.rule_lookup:
                        next_tokens.append(self.rule_lookup[pair])
                        i += 2
                        merged = True
                        continue
                next_tokens.append(tokens[i])
                i += 1
            tokens = next_tokens
            if not merged:
                break

        return tokens

    def decode(self, tokens: Sequence[int]) -> str:
        """Transforms a sequence of token IDs back into readable text."""
        parts: List[str] = []
        for tok in tokens:
            if 0 <= tok < len(self.vocab):
                val = self.vocab[tok]
                if val not in ("[PAD]", "[UNK]"):
                    parts.append(val)
        return "".join(parts)

    def vocab_size(self) -> int:
        """Returns the count of registered subwords."""
        return len(self.vocab)


def is_valid_utf8(s: str) -> bool:
    """Checks whether the string is valid UTF-8."""
    try:
        s.encode("utf-8").decode("utf-8")
        return True
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False
