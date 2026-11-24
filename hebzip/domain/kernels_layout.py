"""Slot numbers of the small state arrays the kernels share with the domain classes.

Why a separate module: domain classes must not import a kernel module (that would define the
kernels before the engine is chosen - ``kernels/jit.py``), but they and the kernels must agree on
which cell holds what. One table, imported by both sides.
"""

from __future__ import annotations

from typing import Final

# number formats (pure math of the one coder and the one cm kernel - written once here, never
# again as bare literals in coder.py, rc.py, arith.py, cm.py or template_tables.py). The template
# precision (probability bits, fixed-point bits) is not here - it is two config rows
# (``[modes.<m>.template].prob_bits`` / ``fix_bits``) that reach the kernel through ``ip`` (below).
REG_BITS: Final = 64
"""The range coder's register width."""
FULL_INT: Final = (1 << REG_BITS) - 1
HALF_INT: Final = 1 << (REG_BITS - 1)
QTR_INT: Final = 1 << (REG_BITS - 2)
M32_INT: Final = (1 << 32) - 1
"""Hash values are 32-bit."""

# coder ``ci`` (int64): pending opposite bits, bits written, bits read, body bits, phantom budget, status
CI_SIZE: Final = 6
PEND: Final = 0
NOUT: Final = 1
POS: Final = 2
NBITS: Final = 3
PMAX: Final = 4
STATUS: Final = 5
ST_PHANTOM: Final = 1
ST_INTERVAL: Final = 2
ST_WEIGHT: Final = 3
STATUS_TEXT: Final = {
    ST_PHANTOM: "המפענח קרא מעבר לסוף הגוף יותר מהמותר (phantom_bits_max)",
    ST_INTERVAL: "מרווח המפענח נשבר",
    ST_WEIGHT: "משקל של מודל התבנית יצא מהטווח",
}

# template ``hp`` (uint64 hash constants)
H_MUL: Final = 0
C0_MUL: Final = 1
SEED_MUL: Final = 2
SEED_ADD: Final = 3
FIN_MUL: Final = 4
FIN_SHIFT1: Final = 5
FIN_SHIFT2: Final = 6
HP_SIZE: Final = 7

# template ``ip`` (int64 learning constants)
LR: Final = 0
LIMIT: Final = 1
AD_START: Final = 2
AD_MIN: Final = 3
AD_MAX: Final = 4
BIAS: Final = 5
WCAP: Final = 6
WSEED: Final = 7
WMAX: Final = 8
LEARN_SIZE: Final = 9
"""Slots 0..8 come from the learning rows; the rest from the precision rows (``Precision.kernel_consts``)."""
PB: Final = 9
"""``prob_bits``: template-bit probabilities are ``prob_bits``-bit integers (1 .. 2**prob_bits - 1)."""
FB: Final = 10
"""``fix_bits``: mixer weights and adaptive probabilities are fixed point with ``fix_bits`` fraction bits."""
P_ONE: Final = 11
"""2**prob_bits (derived)."""
P_SPAN: Final = 12
"""The stretch (ln-odds) domain is -P_SPAN..P_SPAN = 2**(prob_bits-1) - 1 (derived)."""
TO_PROB: Final = 13
"""fix_bits - prob_bits: an adaptive probability -> its index into the stretch table (derived)."""
HALF_FIX: Final = 14
"""2**(fix_bits-1): rounding of the fixed-point weight step (derived)."""
IP_SIZE: Final = 15

# adaptive table ``ameta`` (int64): entries in use, status
AD_COUNT: Final = 0
AD_STATUS: Final = 1

# adaptive table key layout: ``finish << AD_INPUT_BITS | input`` - the input number gets 4 bits
AD_INPUT_BITS: Final = 4
AD_INPUTS_MAX: Final = 1 << AD_INPUT_BITS
"""At most 16 adaptive inputs: a 17th would share an input number - and a key - with
another, so the limit is checked at start-up (``services/select.py::check_fits``). The layout is NOT part
of the format (keys are only compared for equality, the values do not depend on slot positions;
``AD_INPUT_BITS = 8`` gave the same 102/102 golden packages), so the limit could be raised up to 256
(``MARK`` = bit 40 stays above ``finish << 8``) without changing any package. Kept at 16: pending
decision."""
