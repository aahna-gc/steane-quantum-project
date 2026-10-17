
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

PAULIS = ("I", "X", "Y", "Z")
PAULI_BITS = {
    "I": (0, 0),
    "X": (1, 0),
    "Y": (1, 1),
    "Z": (0, 1),
}

# Data-qubit indices for the three Steane stabilizers.
STABILIZERS = (
    (3, 4, 5, 6),
    (1, 2, 5, 6),
    (0, 2, 4, 6),
)


@dataclass(frozen=True)
class PauliMessage:
    probabilities: Mapping[str, float]

    def __post_init__(self):
        if set(self.probabilities) != set(PAULIS):
            raise ValueError("Provide probabilities for I, X, Y, and Z.")

        values = list(self.probabilities.values())

        if any(p < 0 or p > 1 for p in values):
            raise ValueError("Probabilities must lie between 0 and 1.")

        if abs(sum(values) - 1) > 1e-9:
            raise ValueError("Probabilities must sum to 1.")

    @classmethod
    def from_weights(cls, weights):
        if set(weights) != set(PAULIS):
            raise ValueError("Provide weights for I, X, Y, and Z.")

        if any(w < 0 for w in weights.values()):
            raise ValueError("Weights cannot be negative.")

        total = sum(weights.values())
        if total <= 0:
            raise ValueError("Weights must sum to a positive value.")

        return cls({p: weights[p] / total for p in PAULIS})

    @classmethod
    def certain(cls, pauli="I"):
        if pauli not in PAULIS:
            raise ValueError(f"Unknown Pauli: {pauli}")
        return cls({p: float(p == pauli) for p in PAULIS})

    @property
    def most_likely(self):
        return max(PAULIS, key=lambda p: self.probabilities[p])

    @property
    def confidence(self):
        return self.probabilities[self.most_likely]


def combine_independent_paulis(first, second):
    result = {p: 0.0 for p in PAULIS}

    for a in PAULIS:
        ax, az = PAULI_BITS[a]

        for b in PAULIS:
            bx, bz = PAULI_BITS[b]
            bits = (ax ^ bx, az ^ bz)
            product = next(p for p in PAULIS if PAULI_BITS[p] == bits)

            result[product] += (
                first.probabilities[a] * second.probabilities[b]
            )

    return PauliMessage(result)


def choose_correction(message):
    return message.most_likely


def syndrome_for_qubit(qubit: int) -> tuple[int, int, int]:
    """Return the three stabilizer outcomes affected by an X/Z error."""
    if not 0 <= qubit < 7:
        raise ValueError("Data-qubit index must be from 0 through 6.")

    return tuple(
        int(qubit in stabilizer)
        for stabilizer in STABILIZERS
    )


def decode_steane_syndrome(
    x_syndrome: Sequence[int],
    z_syndrome: Sequence[int],
) -> dict:
    """
    Decode a single-round Steane syndrome.

    x_syndrome: outcomes from Z stabilizers; detects X components.
    z_syndrome: outcomes from X stabilizers; detects Z components.

    Each syndrome must contain three binary values.
    """
    x_syndrome = tuple(int(v) for v in x_syndrome)
    z_syndrome = tuple(int(v) for v in z_syndrome)

    if len(x_syndrome) != 3 or len(z_syndrome) != 3:
        raise ValueError("Each syndrome must contain exactly 3 bits.")

    if any(v not in (0, 1) for v in x_syndrome + z_syndrome):
        raise ValueError("Syndrome values must be 0 or 1.")

    def locate(syndrome):
        if syndrome == (0, 0, 0):
            return None

        matches = [
            q for q in range(7)
            if syndrome_for_qubit(q) == syndrome
        ]

        if len(matches) != 1:
            return None

        return matches[0]

    x_qubit = locate(x_syndrome)
    z_qubit = locate(z_syndrome)

    correction = ["I"] * 7

    if x_syndrome != (0, 0, 0) and x_qubit is None:
        return {
            "status": "unresolved",
            "correction": None,
            "x_error_qubit": None,
            "z_error_qubit": None,
        }

    if z_syndrome != (0, 0, 0) and z_qubit is None:
        return {
            "status": "unresolved",
            "correction": None,
            "x_error_qubit": x_qubit,
            "z_error_qubit": None,
        }

    if x_qubit is not None:
        correction[x_qubit] = "X"

    if z_qubit is not None:
        if correction[z_qubit] == "X":
            correction[z_qubit] = "Y"
        else:
            correction[z_qubit] = "Z"

    return {
        "status": "decoded",
        "correction": correction,
        "x_error_qubit": x_qubit,
        "z_error_qubit": z_qubit,
    }


