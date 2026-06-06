"""synthbrain — a from-scratch spiking neural network that learns without backprop.

Public API:
    LIFGroup   - leaky integrate-and-fire neuron layer (optional adaptive threshold)
    Synapses   - weighted connectivity with exponential synaptic current
    STDP       - pair-based spike-timing-dependent plasticity
    Network    - input -> excitatory SNN with lateral inhibition (Diehl & Cook style)

Encoding helpers live in synthbrain.encoding.
"""

from __future__ import annotations

from .lif import LIFGroup
from .synapses import Synapses
from .stdp import STDP
from .network import Network

__all__ = ["LIFGroup", "Synapses", "STDP", "Network"]
