import subprocess
import re

def process_phonemes(phonemes_text):
    """
    Post-process phonemes with sed-style rules.
    
    Args:
        phonemes_text (str): Original phoneme string
    
    Returns:
        str: Post-processed phoneme string
    """
    
    processed = phonemes_text
    
    # replacements = [
    #     ('tʃ', 't͡ʃ'),        # Affricate merge
    #     ('dʒ', 'd͡ʒ'),        # Affricate merge
    #     ('d ʒ', 'd͡ʒ'),       # Merge separated affricates
    # ]
    
    # for old, new in replacements:
    #     processed = processed.replace(old, new)
    
    return processed

def g2p_phonetize(word, model_path='/mnt/workspace/liuhw/sftp_dir/lexicon/model.fst', apply_processing=True):
    """
    Convert a word to phonemes and optionally post-process the result.
    
    Args:
        word (str): Word to convert
        model_path (str): G2P model path
        apply_processing (bool): Whether to apply phoneme post-processing
    
    Returns:
        list: Processed phoneme list. Returns an empty list on failure.
    """
    word = word.upper()
    try:
        # Build the command
        cmd = ['phonetisaurus', 'predict', '--model', model_path, word]
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Parse the output
        output = result.stdout.strip()
        if output:
            # Support two output formats:
            # 1) New format: "WORD PHONEME PHONEME ..."
            # 2) Legacy format: "WORD # PHONEME PHONEME #"
            line = next((ln for ln in output.splitlines() if ln.strip()), '')
            if line:
                if '#' in line:
                    parts = line.split('#')
                    if len(parts) >= 2:
                        phonemes_str = parts[1].strip()
                    else:
                        phonemes_str = ''
                else:
                    tokens = line.strip().split()
                    phonemes_str = ' '.join(tokens[1:]) if len(tokens) >= 2 else ''
                
                if apply_processing and phonemes_str:
                    phonemes_str = process_phonemes(phonemes_str)
                
                phonemes = [p for p in phonemes_str.split() if p]
                return phonemes
        
        return []
        
    except subprocess.CalledProcessError as e:
        print(f"Error running phonetisaurus: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

# Change the codes if you download american-english.fst from http://speechtechnology.web.illinois.edu/data/g2ps/
def g2p_phonetize_batch(words, model_path='/mnt/workspace/liuhw/sftp_dir/lexicon/model.fst', apply_processing=True):
    """
    Convert multiple words to phonemes in batch and optionally post-process the result.
    
    Args:
        words (list): List of words to convert
        model_path (str): G2P model path
        apply_processing (bool): Whether to apply phoneme post-processing
    
    Returns:
        str: Space-separated phoneme string
    """
    all_phonemes = []  # Collect all phonemes here
    words = [w.upper() for w in words]
    try:
        # Use phonetisaurus uniformly without separating single-letter and multi-letter cases
        cmd = ['phonetisaurus', 'predict', '--model', model_path] + list(words)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        lines = result.stdout.strip().split('\n')
        # Parse output lines in the same order as the input words
        for j, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            if '#' in line:
                parts = line.split('#')
                phonemes_str = parts[1].strip() if len(parts) >= 2 else ''
            else:
                tokens = line.strip().split()
                phonemes_str = ' '.join(tokens[1:]) if len(tokens) >= 2 else ''
            
            if apply_processing and phonemes_str:
                phonemes_str = process_phonemes(phonemes_str)
            
            phonemes = [p for p in phonemes_str.split() if p]
            all_phonemes.extend(phonemes)
    except subprocess.CalledProcessError as e:
        print(f"Error running phonetisaurus: {e}")
        # Keep the result empty
    except Exception as e:
        print(f"Unexpected error: {e}")
        # Keep the result empty

    return ' '.join(all_phonemes)

# Usage example
if __name__ == "__main__":
    # Single-word conversion with post-processing
    word = "word"
    phonemes = g2p_phonetize(word)
    print(f"'{word}' -> {phonemes}")
    
    # Test a single letter
    letter = "a"
    phonemes = g2p_phonetize(letter)
    print(f"'{letter}' -> {phonemes}")
    
    # Single-word conversion without post-processing for comparison
    phonemes_raw = g2p_phonetize(word, apply_processing=False)
    print(f"'{word}' (raw) -> {phonemes_raw}")
    
    # Batch conversion
    words = ["hello", "world", "a", "b", "phoneme", "beautiful"]
    results = g2p_phonetize_batch([w for w in words])
    
    print("\nBatch conversion results with post-processing:")
    print(f"'{words}' -> {results}")
    
    # Test specific replacement rules
    test_cases = [
        "so",      # Test ɝ -> ɜ
        "change",    # Test tʃ -> t͡ʃ
        "judge",     # Test dʒ -> d͡ʒ
        "time",      # Test aɪ -> a ɪ
        "house",     # Test aʊ -> a ʊ
        "x",         # Test a single letter
        "y",         # Test a single letter
    ]
    
    print("\nSpecific replacement-rule tests:")
    for word in test_cases:
        phonemes = g2p_phonetize(word)
        print(f"'{word}' -> {' '.join(phonemes)}")