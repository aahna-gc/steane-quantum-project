"""
Hard-decision decoder for the level-3 concatenated Steane code [[343, 1, 27]],
i.e. the circuit from steane_code/benchmarks_l3.py:

    level3_circuit = concatenate_circuit(level_1_steane(1), 2)

Same decoding as the level-1 hard_decision_decoder.py, applied recursively:
each group of 7 measured bits is corrected with the Hamming lookup table
and reduced to one hard logical bit, level by level, then the top level is
decoded exactly like level 1. See concatenated_hard_decoder.py for details.

Usage (from the project root):
    .venv/bin/python hard_decoder/hard_decision_decoder_l3.py          # unit tests
    .venv/bin/python hard_decoder/hard_decision_decoder_l3.py bench    # Monte Carlo -> CSV
"""

from pathlib import Path

from concatenated_hard_decoder import main

LEVEL = 3
OUTPUT_CSV = Path(__file__).resolve().parent / "level3_hard_decoded_results.csv"

if __name__ == "__main__":
    main(LEVEL, OUTPUT_CSV)
