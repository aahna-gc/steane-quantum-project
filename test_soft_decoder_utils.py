
import sys
import unittest
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent / "soft-decoder"),
)

from soft_decoder import (
    PauliMessage,
    combine_independent_paulis,
    choose_correction,
)


class TestSoftDecoderUtilities(unittest.TestCase):

    def test_certain_pauli(self):
        message = PauliMessage.certain("X")
        self.assertEqual(message.most_likely, "X")
        self.assertEqual(message.confidence, 1.0)

    def test_normalization(self):
        message = PauliMessage.from_weights({
            "I": 6,
            "X": 2,
            "Y": 1,
            "Z": 1,
        })
        self.assertAlmostEqual(
            sum(message.probabilities.values()), 1.0
        )
        self.assertEqual(message.most_likely, "I")

    def test_identity_combination(self):
        result = combine_independent_paulis(
            PauliMessage.certain("I"),
            PauliMessage.certain("X"),
        )
        self.assertEqual(result.most_likely, "X")

    def test_x_z_combination(self):
        result = combine_independent_paulis(
            PauliMessage.certain("X"),
            PauliMessage.certain("Z"),
        )
        self.assertEqual(result.most_likely, "Y")

    def test_choose_correction(self):
        message = PauliMessage.from_weights({
            "I": 0.1,
            "X": 0.7,
            "Y": 0.1,
            "Z": 0.1,
        })
        self.assertEqual(choose_correction(message), "X")

from soft_decoder import decode_steane_syndrome, syndrome_for_qubit


# Check all seven single-qubit X, Z, and Y errors.
for q in range(7):
    syndrome = syndrome_for_qubit(q)

    x_result = decode_steane_syndrome(syndrome, (0, 0, 0))
    assert x_result["x_error_qubit"] == q
    assert x_result["correction"][q] == "X"

    z_result = decode_steane_syndrome((0, 0, 0), syndrome)
    assert z_result["z_error_qubit"] == q
    assert z_result["correction"][q] == "Z"

    y_result = decode_steane_syndrome(syndrome, syndrome)
    assert y_result["correction"][q] == "Y"

print("All 21 single-qubit syndrome tests passed.")

if __name__ == "__main__":
    unittest.main(verbosity=2)
