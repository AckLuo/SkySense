"""M2 信道建模模块：AWGN、多径、瑞利衰落、调制解调、BER."""

from .awgn import add_awgn, add_awgn_snr
from .modulation import modulate, demodulate, get_constellation
from .multipath import MultipathChannel
from .metrics import ber, evm

__all__ = [
    "add_awgn",
    "add_awgn_snr",
    "modulate",
    "demodulate",
    "get_constellation",
    "MultipathChannel",
    "ber",
    "evm",
]
