import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set
import json

class ByteLevelBPETokenizer:
    def __init__(self):
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        self.bpe_merges = []
        self.bpe_ranks = {}
        self.vocab = {}
        self.decoder = {}

    def _bytes_to_unicode(self) -> Dict[int, str]:
        bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
        cs = bs[:]
        n = 0

        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1

        return dict(zip(bs, [chr(c) for c in cs]))

    def _get_pairs(self, word: Tuple[str, ...]) -> Set[Tuple[str, str]]:
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def _basic_clean(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def train(self, texts: List[str], vocab_size: int = 50000, min_frequency: int = 2):
        word_freqs = Counter()

        for text in texts:
            text = self._basic_clean(text)
            encoded_text = ''.join(self.byte_encoder[b] for b in text.encode('utf-8'))

            words = re.findall(r'\S+|\s+', encoded_text)

            for word in words:
                word_tuple = tuple(word + '')
                word_freqs[word_tuple] += 1

        print(f"Found {len(word_freqs)} unique words")

        vocab = set()
        for word in word_freqs.keys():
            vocab.update(word)

        merges = []

        while len(vocab) < vocab_size:
            pairs = defaultdict(int)
            for word, freq in word_freqs.items():
                word_pairs = self._get_pairs(word)
                for pair in word_pairs:
                    pairs[pair] += freq

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)

            if pairs[best_pair] < min_frequency:
                break

            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word = self._merge_word(word, best_pair)
                new_word_freqs[new_word] = freq

            word_freqs = new_word_freqs
            merges.append(best_pair)
            vocab.add(''.join(best_pair))

            if len(merges) % 1000 == 0:
                print(f"Learned {len(merges)} merges, vocab size: {len(vocab)}")

        self.bpe_merges = merges
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}

        vocab_list = sorted(list(vocab))
        self.vocab = {token: i for i, token in enumerate(vocab_list)}
        self.decoder = {i: token for token, i in self.vocab.items()}

        print(f"{len(merges)} merges, final vocab size: {len(self.vocab)}")

    def _merge_word(self, word: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(pair[0], i)
                new_word.extend(word[i:j])
                i = j
            except ValueError:
                new_word.extend(word[i:])
                break

            if i < len(word) - 1 and word[i + 1] == pair[1]:
                new_word.append(pair[0] + pair[1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        return tuple(new_word)

    def _bpe(self, token: str) -> List[str]:
        if len(token) <= 1:
            return [token]

        word = tuple(token)
        pairs = self._get_pairs(word)

        if not pairs:
            return [token]

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))

            if bigram not in self.bpe_ranks:
                break

            word = self._merge_word(word, bigram)
            if len(word) == 1:
                break
            else:
                pairs = self._get_pairs(word)

        return list(word)

    def encode(self, text: str) -> List[int]:
        if not self.vocab:
            raise ValueError("Tokenizer not trained. Call train() first.")

        text = self._basic_clean(text)
        encoded_text = ''.join(self.byte_encoder[b] for b in text.encode('utf-8'))

        words = re.findall(r'\S+|\s+', encoded_text)

        bpe_tokens = []
        for word in words:
            word += ''
            word_tokens = self._bpe(word)
            bpe_tokens.extend(word_tokens)

        token_ids = []
        for token in bpe_tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                for char in token:
                    if char in self.vocab:
                        token_ids.append(self.vocab[char])
                    else:
                        # This shouldn't happen if training was done properly
                        print(f"Warning: Unknown character {repr(char)}")

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        if not self.decoder:
            raise ValueError("Tokenizer not trained. Call train() first.")

        tokens = []
        for token_id in token_ids:
            if token_id in self.decoder:
                tokens.append(self.decoder[token_id])
            else:
                print(f"Warning: Unknown token ID {token_id}")

        text = ''.join(tokens).replace('', ' ')

        try:
            byte_sequence = bytes([self.byte_decoder[c] for c in text])
            return byte_sequence.decode('utf-8')
        except (KeyError, UnicodeDecodeError) as e:
            print(f"Warning: Decoding error {e}")
            return text

    def save(self, filepath: str):
        data = {
            'bpe_merges': self.bpe_merges,
            'vocab': self.vocab,
            'byte_encoder': {str(k): v for k, v in self.byte_encoder.items()}
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.bpe_merges = [tuple(pair) for pair in data['bpe_merges']]
        self.bpe_ranks = {tuple(pair): i for i, pair in enumerate(self.bpe_merges)}
        self.vocab = data['vocab']
        self.decoder = {v: k for k, v in self.vocab.items()}
        self.byte_encoder = {int(k): v for k, v in data['byte_encoder'].items()}
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
     
