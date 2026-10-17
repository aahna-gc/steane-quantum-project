import sys
import unittest
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent / "soft-decoder"),
)

from soft_decoder import decode_steane_syndrome, syndrome_for_qubit


class TestSteaneSyndromeDecoder(unittest.TestCase):

    def test_no_error(self):
        result = decode_steane_syndrome(
            (0, 0, 0),
            (0, 0, 0),
        )
        self.assertEqual(result["status"], "decoded")
        self.assertIsNone(result["x_error_qubit"])
        self.assertIsNone(result["z_error_qubit"])
        self.assertEqual(result["correction"], ["I"] * 7)

    def test_each_single_x_error(self):
        for qubit in range(7):
            with self.subTest(qubit=qubit):
                result = decode_steane_syndrome(
                    syndrome_for_qubit(qubit),
                    (0, 0, 0),
                )
                self.assertEqual(result["status"], "decoded")
                self.assertEqual(result["x_error_qubit"], qubit)
                self.assertEqual(result["correction"][qubit], "X")

    def test_each_single_z_error(self):
        for qubit in range(7):
            with self.subTest(qubit=qubit):
                result = decode_steane_syndrome(
                    (0, 0, 0),
                    syndrome_for_qubit(qubit),
                )
                self.assertEqual(result["status"], "decoded")
                self.assertEqual(result["z_error_qubit"], qubit)
                self.assertEqual(result["correction"][qubit], "Z")

    def test_y_error(self):
        for qubit in range(7):
            with self.subTest(qubit=qubit):
                syndrome = syndrome_for_qubit(qubit)
                result = decode_steane_syndrome(syndrome, syndrome)
                self.assertEqual(result["status"], "decoded")
                self.assertEqual(result["correction"][qubit], "Y")

    def test_invalid_syndrome_length(self):
        with self.assertRaises(ValueError):
            decode_steane_syndrome((0, 1), (0, 0, 0))

    def test_invalid_syndrome_value(self):
        with self.assertRaises(ValueError):
            decode_steane_syndrome((0, 2, 0), (0, 0, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
