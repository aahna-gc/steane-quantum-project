
"""
Soft-information decoder

provides the probability-message operations needed by
the recursive decoder
note: syndrome-conditioned decoding will be added
once the Steane code interface is connected
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

PAULIS = ("I", "X", "Y", "Z")

# represent each Pauli by its two binary components (x, z)
# ultiplication of Pauli labels, ignoring global phase, is XOR
PAULI_BITS = {
    "I": (0, 0),
    "X": (1, 0),
    "Y": (1, 1),
    "Z": (0, 1),
}


@dataclass(frozen=True)
class PauliMessage:
    """Normalized probabilities for an effective Pauli error."""

    probabilities: Mapping[str, float]

    def __post_init__(self) -> None:
        if set(self.probabilities) != set(PAULIS):
            raise ValueError("Provide probabilities for I, X, Y, and Z.")

        values = list(self.probabilities.values())

        if any(p < 0.0 or p > 1.0 for p in values):
            raise ValueError("Probabilities must lie between 0 and 1.")

        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("Pauli probabilities must sum to 1.")

    @classmethod
    def from_weights(
        cls, weights: Mapping[str, float]
    ) -> "PauliMessage":
        """Normalize nonnegative weights into a probability message."""
        if set(weights) != set(PAULIS):
            raise ValueError("Provide weights for I, X, Y, and Z.")

        if any(w < 0.0 for w in weights.values()):
            raise ValueError("Weights cannot be negative.")

        total = sum(weights.values())
        if total <= 0.0:
            raise ValueError("At least one weight must be positive.")

        return cls({p: weights[p] / total for p in PAULIS})

    @classmethod
    def certain(cls, pauli: str = "I") -> "PauliMessage":
        """Create a message assigning probability 1 to one Pauli."""
        if pauli not in PAULIS:
            raise ValueError(f"Unknown Pauli: {pauli}")

        return cls({
            p: float(p == pauli)
            for p in PAULIS
        })

    @property
    def most_likely(self) -> str:
        """Return the Pauli with the largest probability."""
        return max(PAULIS, key=lambda p: self.probabilities[p])

    @property
    def confidence(self) -> float:
        """Probability assigned to the most likely Pauli."""
        return self.probabilities[self.most_likely]


def combine_independent_paulis(
    first: PauliMessage,
    second: PauliMessage,
) -> PauliMessage:
    """
    Distribution of the product of two independent Pauli errors.

    This combines error labels only. It does not perform syndrome
    decoding or account for correlations between the errors.
    """
    result = {p: 0.0 for p in PAULIS}

    for a in PAULIS:
        ax, az = PAULI_BITS[a]

        for b in PAULIS:
            bx, bz = PAULI_BITS[b]

            product = next(
                p for p in PAULIS
                if PAULI_BITS[p] == (ax ^ bx, az ^ bz)
            )

            result[product] += (
                first.probabilities[a]
                * second.probabilities[b]
            )

    return PauliMessage(result)


def choose_correction(message: PauliMessage) -> str:
    """Return the maximum-probability Pauli hypothesis."""
    return message.most_likely

