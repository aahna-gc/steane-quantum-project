"""
    Level 1 (inner, bit-flip repetition code): for each of the 3 blocks,
        use that block's 2 Z-stabilizer syndrome bits to guess which
        single qubit (if any) suffered an X error, via majority-vote /
        syndrome lookup -- this is provably optimal for a single,
        isolated 3-qubit block (our project's "zero-gap control" logic,
        applied here 3 times in parallel).

    Level 2 (outer, phase-flip repetition code): the 2 X-stabilizer
        syndrome bits behave EXACTLY like a second 3-bit repetition code,
        except the "bits" being voted on are each BLOCK's overall Z-error
        parity rather than individual qubits. The identical majority-vote
        lookup table applies -- guess which block (if any) has odd Z
        parity, and correct it.

Because both levels use literally the same 3-symbol repetition-code
syndrome-to-correction lookup table, `_REP3_LOOKUP` is defined once and
reused for both. This mirrors the literature notes from Aug 24
(day3_majority_vote_notes.md): each level only ever sees a HARD 2-bit
syndrome and commits to a single guess, with no confidence/likelihood
carried between levels or between blocks.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from shor9_code import BLOCKS, HZ_PAIRS, HX_GROUPS, NUM_QUBITS


from shor_code.circuit_level_code import circuit_level_shor, circuit_round


# Syndrome (bit0, bit1) -> index of the flipped symbol within its group of 3,
# or None if the syndrome says "no error". Derived directly from HZ_PAIRS'
# convention (bit0 = parity(sym0, sym1), bit1 = parity(sym1, sym2)):
#   no error      -> (0, 0) -> None
#   symbol 0 flip -> (1, 0) -> 0
#   symbol 1 flip -> (1, 1) -> 1
#   symbol 2 flip -> (0, 1) -> 2
# (A "symbol" is a qubit at level 1, or a block at level 2.)
_REP3_LOOKUP = {
    (0, 0): None,
    (1, 0): 0,
    (1, 1): 1,
    (0, 1): 2,
}


def decode_bitflip_level1(synd_z: Sequence[int]) -> List[int]:
    """Level 1: 6-bit Z-stabilizer syndrome -> length-9 X-correction vector.
    Each block's 2 syndrome bits are decoded independently."""
    correction = [0] * NUM_QUBITS
    for block_idx, block in enumerate(BLOCKS):
        bit0 = synd_z[2 * block_idx]
        bit1 = synd_z[2 * block_idx + 1]
        flipped_symbol = _REP3_LOOKUP[(bit0, bit1)]
        if flipped_symbol is not None:
            correction[block[flipped_symbol]] = 1
    return correction


def decode_phaseflip_level2(synd_x: Sequence[int]) -> List[int]:
    """Level 2: 2-bit X-stabilizer syndrome -> length-9 Z-correction vector.
    Treats the 3 blocks as the 3 "symbols" of an outer repetition code and
    applies the SAME lookup table as level 1. If a block is flagged, the Z
    correction is applied to that block's first qubit -- flipping any one
    qubit in the block cancels the block's Z-parity, and which qubit within
    the block is chosen doesn't affect the logical outcome (they differ
    only by a stabilizer element)."""
    correction = [0] * NUM_QUBITS
    flipped_block = _REP3_LOOKUP[(synd_x[0], synd_x[1])]
    if flipped_block is not None:
        correction[BLOCKS[flipped_block][0]] = 1
    return correction


def hard_decision_decode(
    synd_z: Sequence[int], synd_x: Sequence[int], p: float = None
) -> Tuple[List[int], List[int]]:
    """Full two-level hard-decision decoder. `p` is accepted (and ignored)
    so this function has the same call signature as the ML decoder in
    ml_decoder_stub.py -- see monte_carlo.py for why that matters."""
    return decode_bitflip_level1(synd_z), decode_phaseflip_level2(synd_x)


# ---------------------------------------------------------------------------
# Hand-checkable unit tests.
# ---------------------------------------------------------------------------

def _xor(a: Sequence[int], b: Sequence[int]) -> List[int]:
    return [x ^ y for x, y in zip(a, b)]


def _run_unit_tests() -> None:
    from shor9_code import syndrome_z, syndrome_x, logical_z_value, logical_x_value

    def check_bitflip_case(name, ex, expect_logical_flip):
        sz = syndrome_z(ex)
        cx = decode_bitflip_level1(sz)
        residual = _xor(ex, cx)
        got = logical_z_value(residual)
        assert got == expect_logical_flip, (
            f"[{name}] expected logical_z_flip={expect_logical_flip}, got {got} "
            f"(ex={ex}, syndrome={sz}, correction={cx}, residual={residual})"
        )
        print(f"  PASS: {name} -> logical_z_flip={got}")

    def check_phaseflip_case(name, ez, expect_logical_flip):
        sx = syndrome_x(ez)
        cz = decode_phaseflip_level2(sx)
        residual = _xor(ez, cz)
        got = logical_x_value(residual)
        assert got == expect_logical_flip, (
            f"[{name}] expected logical_x_flip={expect_logical_flip}, got {got} "
            f"(ez={ez}, syndrome={sx}, correction={cz}, residual={residual})"
        )
        print(f"  PASS: {name} -> logical_x_flip={got}")

    print("Level 1 (bit-flip / intra-block majority vote):")
    check_bitflip_case("no errors", [0] * 9, 0)
    check_bitflip_case("single X in block 0 (qubit 0)", [1, 0, 0, 0, 0, 0, 0, 0, 0], 0)
    check_bitflip_case("single X in block 0 (qubit 1)", [0, 1, 0, 0, 0, 0, 0, 0, 0], 0)
    check_bitflip_case("single X in block 1 (qubit 4)", [0, 0, 0, 0, 1, 0, 0, 0, 0], 0)
    check_bitflip_case(
        "double X in block 0 (qubits 0,1) -- majority vote FAILS",
        [1, 1, 0, 0, 0, 0, 0, 0, 0], 1,
    )
    check_bitflip_case(
        "triple X in block 0 (all of block 0) -- exact logical X_bar",
        [1, 1, 1, 0, 0, 0, 0, 0, 0], 1,
    )
    check_bitflip_case(
        "double-X failures in block 0 AND block 1 -- cancel (even # of failures)",
        [1, 1, 0, 1, 1, 0, 0, 0, 0], 0,
    )
    check_bitflip_case(
        "double-X failures in all 3 blocks -- odd # of failures (3), still fails",
        [1, 1, 0, 1, 1, 0, 1, 1, 0], 1,
    )

    print("\nLevel 2 (phase-flip / inter-block majority vote):")
    check_phaseflip_case("no errors", [0] * 9, 0)
    check_phaseflip_case("single Z in block 0", [1, 0, 0, 0, 0, 0, 0, 0, 0], 0)
    check_phaseflip_case(
        "single Z in each of blocks 0 and 1 (2 blocks bad) -- indistinguishable "
        "from 'only block 2 bad', so majority vote guesses the wrong (lower-weight) "
        "explanation -- level-2 majority vote FAILS, same failure mode as level 1's "
        "double-X case",
        [1, 0, 0, 1, 0, 0, 0, 0, 0], 1,
    )
    check_phaseflip_case(
        "single Z in each of ALL 3 blocks (weight-3, = exactly Z_bar) -- "
        "indistinguishable from 'no blocks bad', decoder does nothing, "
        "residual is a genuine (undetectable) logical Z error",
        [1, 0, 0, 1, 0, 0, 1, 0, 0], 1,
    )

    print("\nALL UNIT TESTS PASSED.")


if __name__ == "__main__":
    _run_unit_tests()