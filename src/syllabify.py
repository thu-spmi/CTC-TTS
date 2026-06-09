# -*- coding: utf-8 -*-
"""
A simple consonant/vowel-based syllabification approach:
- Consecutive vowels are merged into one nucleus (supports diphthongs such as `a ʊ`)
- For consonant clusters between two nuclei, assign the last consonant to the next onset and the rest to the previous coda
- Word-initial consonant clusters go to the onset of the first syllable; word-final consonant clusters go to the coda of the last syllable
"""

from typing import List

# Classify using the provided phoneme inventory
VOWELS = {
    "a", "e", "i", "o", "u",
    "æ", "ɑ", "ɔ", "ə", "ɛ", "ɜ", "ɪ", "ʊ", "ʌ",
}

# Treat as consonants, including affricates as single consonants
CONSONANTS = {
    "b", "d", "f", "h", "j", "k", "l", "m", "n", "p", "s", "t", "v", "w", "z",
    "ð", "ŋ", "ɡ", "ɹ", "ɾ", "ʃ", "ʒ", "t͡ʃ", "d͡ʒ", "ɵ",
}

def is_vowel(ph: str) -> bool:
    return ph in VOWELS

def is_consonant(ph: str) -> bool:
    return ph in CONSONANTS

def is_legal_onset(onset: List[str]) -> bool:
    """
    Determine whether the given onset consonant cluster is a valid onset.
    Uses heuristic common English onset patterns only and does not depend on a finer-grained syllable-structure lexicon.
    """
    L = len(onset)
    if L == 0:
        return False
    if L == 1:
        return onset[0] in CONSONANTS

    # Allowed two-consonant onsets
    allowed_two = {
        ("p", "l"), ("p", "ɹ"), ("b", "l"), ("b", "ɹ"),
        ("t", "ɹ"), ("d", "ɹ"),
        ("k", "l"), ("k", "ɹ"), ("g", "l"), ("g", "ɹ"),
        ("f", "l"), ("f", "ɹ"),
        ("s", "l"), ("s", "m"), ("s", "n"), ("s", "p"), ("s", "t"), ("s", "k"),
        ("s", "w"),
        # Consonant + `j`, such as `pure (p j)` or `beauty (b j)`
        ("p", "j"), ("b", "j"), ("t", "j"), ("d", "j"), ("k", "j"), ("g", "j"), ("f", "j"),
        # `ʃ + ɹ`, such as `shred`
        ("ʃ", "ɹ"),
    }

    if L == 2:
        pair = (onset[0], onset[1])
        return pair in allowed_two

    # Allowed three-consonant onsets: `s + (p|t|k) + (r|l|w)`
    allowed_three = {
        ("s", "p", "ɹ"), ("s", "p", "l"),
        ("s", "t", "ɹ"),
        ("s", "k", "ɹ"), ("s", "k", "l"), ("s", "k", "w"),
    }
    if L == 3:
        triple = (onset[0], onset[1], onset[2])
        return triple in allowed_three

    # Four or more consonants are usually not treated as valid onsets
    return False

def choose_maximal_legal_onset_suffix(cons_cluster: List[str]) -> (List[str], List[str]):
    """
    Within the inter-nuclear consonant cluster `cons_cluster`, choose the largest valid onset suffix as the onset of the next syllable,
    Return `(next_onset, coda)`, where `coda` remains in the previous syllable.
    If no valid onset exists, fall back to using the last consonant as the onset.
    """
    if not cons_cluster:
        return [], []

    max_len = min(3, len(cons_cluster))
    for L in range(max_len, 0, -1):
        candidate = cons_cluster[-L:]
        if is_legal_onset(candidate):
            return candidate, cons_cluster[:-L]

    # Fallback strategy: use at least the last consonant as the onset
    return [cons_cluster[-1]], cons_cluster[:-1]

def syllabify_tokens(tokens: List[str]) -> List[List[str]]:
    """
    Split a phoneme list into syllables and return the phoneme list for each syllable.
    """
    if not tokens:
        return []

    syllables: List[List[str]] = []
    onset_buffer: List[str] = []       # Current syllable onset
    nucleus_buffer: List[str] = []     # Current syllable nucleus (may contain one or more vowels, such as a diphthong)
    inter_vowel_cons: List[str] = []   # Consonant cluster accumulated between two nuclei

    def flush_current_syllable(include_coda: List[str] = None):
        # Emit the accumulated onset + nucleus (+ coda) as one syllable
        if not nucleus_buffer and not onset_buffer and not include_coda:
            return
        syll = []
        syll.extend(onset_buffer)
        syll.extend(nucleus_buffer)
        if include_coda:
            syll.extend(include_coda)
        syllables.append(syll)

    i = 0
    n = len(tokens)

    # First collect the word-initial onset until the first vowel is encountered
    i = 0
    n = len(tokens)
    while i < n and tokens[i] in CONSONANTS:
        onset_buffer.append(tokens[i])
        i += 1

    while i < n:
        tok = tokens[i]
        if tok in VOWELS:
            nucleus_buffer.append(tok)
            i += 1
            # Merge consecutive vowels into the current nucleus (for diphthongs, etc.)
            while i < n and tokens[i] in VOWELS:
                nucleus_buffer.append(tokens[i])
                i += 1

            # Collect the consonant cluster after the current nucleus
            inter_vowel_cons.clear()
            while i < n and tokens[i] in CONSONANTS:
                inter_vowel_cons.append(tokens[i])
                i += 1

            if i < n:
                # If more vowels follow, split the consonant cluster into `(coda, next_onset)` using the largest valid onset suffix
                next_onset, coda = choose_maximal_legal_onset_suffix(inter_vowel_cons)
                # Output the current syllable: onset + nucleus + coda
                syll = []
                syll.extend(onset_buffer)
                syll.extend(nucleus_buffer)
                syll.extend(coda)
                syllables.append(syll)
                # Set the next syllable onset to `next_onset`; clear the nucleus and wait for the following vowels
                onset_buffer = next_onset
                nucleus_buffer = []
            else:
                # At the word end, any remaining consonant cluster becomes the coda of the last syllable
                coda = inter_vowel_cons
                syll = []
                syll.extend(onset_buffer)
                syll.extend(nucleus_buffer)
                syll.extend(coda)
                syllables.append(syll)
                onset_buffer = []
                nucleus_buffer = []
        else:
            # If a symbol is neither a vowel nor a consonant (or lies outside the inventory), conservatively merge it into the current nucleus to avoid losing it
            nucleus_buffer.append(tok)
            i += 1

    # After finishing, if there is still an un-emitted onset/nucleus (which should not normally happen), emit it
    if onset_buffer or nucleus_buffer:
        # This branch should not normally be reached; if it is, emit the remainder
        syll = []
        syll.extend(onset_buffer)
        syll.extend(nucleus_buffer)
        syllables.append(syll)
    return syllables

def syllabify_str(phoneme_str: str) -> List[str]:
    """
    Split a space-separated phoneme string into syllables and return each syllable as a space-joined string.
    """
    tokens = [t for t in phoneme_str.strip().split() if t]
    syllables = syllabify_tokens(tokens)
    print(f"syllables: {syllables}")
    return [" ".join(s) for s in syllables]

if __name__ == "__main__":
    examples = [
        "s k w i z ɪ n",
        "t͡ʃ æ n ʒ ə",
        "b r a ʊ n",
        "d ɪ s t ɹ ɪ b j u ʃ ə n", 
        "k æ t",
        "a ʊ t",
        "m s k m",
        "ɪ ŋ ɡ l ɪ ʃ",
        "æ k ə m i ɵ i z",
        "ɡ j u d͡ʒ ə i",
        "d ɪ s k ɹ ɪ b",
        "k ɹ i e i ɾ ɪ v",
        "k ɹ i e i ʃ ə n l l i",
    ]
    for ex in examples:
        print("Input:", ex)
        print("Syllables:", syllabify_str(ex))
        print("-" * 40)