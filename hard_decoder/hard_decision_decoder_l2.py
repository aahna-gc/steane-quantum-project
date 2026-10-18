"""
Hard-decision decoder for the level-2 concatenated Steane code [[49, 1, 9]],
i.e. the circuit from steane_code/benchmarks_l2.py:

    level2_circuit = concatenate_circuit(level_1_steane(1), 1)

Same decoding as the level-1 hard_decision_decoder.py, applied recursively:
each group of 7 measured bits is corrected with the Hamming lookup table
and reduced to one hard logical bit, level by level, then the top level is
decoded exactly like level 1. See concatenated_hard_decoder.py for details.

Usage (from the project root):
    .venv/bin/python hard_decoder/hard_decision_decoder_l2.py          # unit tests
    .venv/bin/python hard_decoder/hard_decision_decoder_l2.py bench    # Monte Carlo -> CSV
"""

from pathlib import Path

from concatenated_hard_decoder import main

LEVEL = 2
OUTPUT_CSV = Path(__file__).resolve().parent / "level2_hard_decoded_results.csv"

if __name__ == "__main__":
    main(LEVEL, OUTPUT_CSV)
