from typing import Dict, Generator
import time
import random

class FixedTextStreamer:
    """A class that simulates streaming output with fixed text"""
    
    def __init__(self, delay_range=0.0001):
        """Initialize the fixed-text streaming generator
        
        Args:
            delay_range: Delay time between text segments
        """
        self.delay_range = delay_range
    
    def generate_text_stream(self, unit_list: list[list[int]]) -> Generator[list[int], None, None]:
        """Generate streaming output for fixed text
        
        Args:
            unit_list: Fixed text units to stream out
            
        Returns:
            Generator that yields text segments one by one
        """
        
        # Generate text segments one by one
        for unit in unit_list:
            if unit is None:
                print(f"warning: unit is None!")
                continue
            yield unit

# Usage example
def main():
    streamer = FixedTextStreamer()
    unit_list = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    
    # Simulate a calling pattern similar to the original code
    for unit_piece in streamer.generate_text_stream(unit_list):
        print(unit_piece, end="", flush=True)  # Print each text segment in real time
    
    print("\nStreaming output completed")

if __name__ == "__main__":
    main()