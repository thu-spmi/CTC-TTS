import os
import re
from collections import Counter

def process_consecutive_punctuation(text):
    """
    Handle complex logic for repeated punctuation
    1. If repeated punctuation contains apostrophes, remove the apostrophes from that punctuation span
    2. Handle spaces between punctuation marks
    3. Keep one punctuation mark by priority: ! > ? > . > , > '
    """
    # Define punctuation priority
    punctuation_priority = {
        '!': 1,
        '?': 2, 
        '.': 3,
        ',': 4,
        "'": 5
    }
    
    # Regular expression: match only repeated punctuation (two or more punctuation marks)
    # Match: punctuation+space+punctuation or punctuation+punctuation
    pattern = r"[!?.,'](?:\s*[!?.,'])+"
    
    def replace_punct_group(match):
        group = match.group(0)
        # Extract all punctuation marks and ignore spaces
        puncts = [char for char in group if char in punctuation_priority]
        
        if not puncts or len(puncts) < 2:
            return group
        
        # If repeated punctuation contains apostrophes, remove all apostrophes first
        if "'" in puncts:
            puncts = [p for p in puncts if p != "'"]
        
        # If no punctuation remains after removing apostrophes, return an empty string
        if not puncts:
            return ""
        
        # If there is only one punctuation mark, return it directly
        if len(puncts) == 1:
            return puncts[0]
        
        # Choose the highest-priority punctuation mark
        best_punct = min(puncts, key=lambda x: punctuation_priority[x])
        return best_punct
    
    # Replace all punctuation groups
    result = re.sub(pattern, replace_punct_group, text)
    return result


def clean_text_content(text):
    """
    Clean and normalize the text content
    """
    text = text.strip().lower()
    text = text.replace('’', "'")
    text = text.replace('‘', "'")
    text = text.replace('“', ' ')
    text = text.replace('”', ' ')
    # 1. Replace English colons and semicolons with periods
    text = text.replace(':', '.')
    text = text.replace(';', '.')
    
    # 2. Remove hyphens and replace dashes with commas
    text = text.replace('-', ' ')
    text = text.replace('—', ',')
    
    # 3. Replace special symbols with spaces
    special_chars = '\"()/[]{}¯'
    for char in special_chars:
        text = text.replace(char, ' ')
    
    # 4. Replace special letters
    replacements = {
        'è': 'e', 'é': 'e', 'ê': 'e',
        'ñ': 'n',
        'ò': 'o', 'ô': 'o',
        'æ': 'ae',
        'œ': 'e'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # 5. Replace repeated spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    text = re.sub(r"([,.!?])(?=[a-z0-9])", r"\1 ", text)
    
    
    # 6. Handle repeated punctuation
    text = process_consecutive_punctuation(text)
    
    # 7. Remove apostrophes that are not inside words, keeping only apostrophes like the one in `don't`
    text = re.sub(r"(?<![a-z])'|'(?![a-z])", "", text)
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    text = re.sub(r"\s+", " ", text)

    # 8. If the text starts with one of `,.!?`, remove that punctuation mark
    text = text.lstrip(",.!? ")

    # 9. If the last character is not one of `. ! ?`, append a period
    if text and text[-1] not in '.!?':
        text += '.'

    # 10. Trim leading and trailing spaces
    text = text.strip()
    
    return text

def process_text_file(input_file, output_file):
    """
    Process a single text file
    """
    processed_lines = []
    error_count = 0
    
    print(f"Processing file: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            try:
                line = line.strip()
                if not line:
                    continue
                
                # Split the ID and text content
                parts = line.split('\t', 1)
                if len(parts) != 2:
                    print(f"Warning: line {i+1} has an invalid format: {line[:50]}...")
                    error_count += 1
                    continue
                
                unique_id, text_content = parts
                
                # Process the text content
                cleaned_text = clean_text_content(text_content)
                
                # Save the processed line
                processed_lines.append(f"{unique_id}\t{cleaned_text}")
                
            except Exception as e:
                print(f"Error while processing line {i+1}: {e}")
                error_count += 1
        
        # Write the processed file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(processed_lines))
        
        print(f"Finished processing file: {output_file}")
        print(f"Successfully processed lines: {len(processed_lines)}")
        print(f"Error lines: {error_count}")
        
        return len(processed_lines), error_count
        
    except Exception as e:
        print(f"Error while processing file {input_file}: {e}")
        return 0, 1

def analyze_characters_in_file(file_path):
    """
    Analyze character statistics in a file
    """
    all_text = ""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    text_content = parts[1]
                    all_text += text_content + " "
        
        # Count characters
        char_counter = Counter(all_text)
        unique_chars = set(all_text)
        
        return unique_chars, char_counter, len(all_text)
        
    except Exception as e:
        print(f"Error analyzing file {file_path}: {e}")
        return set(), Counter(), 0

def save_character_analysis(output_dir, all_chars, all_counters, text_lengths):
    """
    Save character analysis results
    """
    analysis_file = os.path.join(output_dir, 'norm_character_analysis.txt')
    
    with open(analysis_file, 'w', encoding='utf-8') as f:
        f.write("Character statistics for normalized text\n")
        f.write("=" * 50 + "\n\n")
        
        # Merge all characters
        all_unique_chars = set()
        combined_counter = Counter()
        total_length = 0
        
        for split_name in ['train', 'dev', 'test']:
            if split_name in all_chars:
                all_unique_chars.update(all_chars[split_name])
                combined_counter.update(all_counters[split_name])
                total_length += text_lengths[split_name]
                
                f.write(f"{split_name} split:\n")
                f.write(f"Unique character count: {len(all_chars[split_name])}\n")
                f.write(f"Total text length: {text_lengths[split_name]}\n")
                f.write(f"Characters: {''.join(sorted(all_chars[split_name]))}\n\n")
        
        f.write(f"Overall statistics:\n")
        f.write(f"Total unique character count: {len(all_unique_chars)}\n")
        f.write(f"Total text length: {total_length}\n")
        f.write(f"All unique characters: {''.join(sorted(all_unique_chars))}\n\n")
        
        f.write("Character frequency statistics:\n")
        for char, count in combined_counter.most_common():
            if char == ' ':
                f.write(f"Space: {count}\n")
            elif char == '\n':
                f.write(f"Newline: {count}\n")
            elif char == '\t':
                f.write(f"Tab: {count}\n")
            else:
                f.write(f"'{char}': {count}\n")
    
    print(f"\nCharacter analysis results have been saved to: {analysis_file}")

def main():
    # Input directory
    input_dir = "/mnt/workspace/liuhw/streaming_tts/dict"
    
    print("Start processing text files...")
    print(f"Input directory: {input_dir}")
    
    # Process each dataset split
    splits = ['train', 'dev', 'test']
    total_lines = 0
    total_errors = 0
    
    all_chars = {}
    all_counters = {}
    text_lengths = {}
    
    for split in splits:
        input_file = os.path.join(input_dir, split, 'text')
        output_file = os.path.join(input_dir, split, 'norm_text')
        
        if os.path.exists(input_file):
            # Process text files
            line_count, error_count = process_text_file(input_file, output_file)
            total_lines += line_count
            total_errors += error_count
            
            # Analyze processed characters
            chars, counter, text_length = analyze_characters_in_file(output_file)
            all_chars[split] = chars
            all_counters[split] = counter
            text_lengths[split] = text_length
            
            print(f"\n{split} split character statistics:")
            print(f"Total unique characters: {len(chars)}")
            print(f"Total text characters: {text_length}")
            print(f"All unique characters: {''.join(sorted(chars))}")
            print("-" * 50)
        else:
            print(f"Warning: input file does not exist: {input_file}")
    
    # Save character analysis results to the dict directory
    if all_chars:
        save_character_analysis(input_dir, all_chars, all_counters, text_lengths)
    
    print(f"\n=== Overall processing results ===")
    print(f"Total processed lines: {total_lines}")
    print(f"Total error lines: {total_errors}")
    print(f"Output files:")
    for split in splits:
        output_file = os.path.join(input_dir, split, 'norm_text')
        if os.path.exists(output_file):
            print(f"  {output_file}")
    print(f"  {os.path.join(input_dir, 'norm_character_analysis.txt')}")
    print("\nProcessing completed!")

if __name__ == "__main__":
    main()
