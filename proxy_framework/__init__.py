"""
Proxy training & evaluation framework for Parameter Golf.

Enables cheap local screening of model ideas on a consumer GPU (RTX 3080 12GB)
while preserving ranking signal relative to the full 8xH100 competition setting.

Design informed by:
- DataDecide (arXiv:2504.11393): small-scale rankings can be predictive
- DoReMi (arXiv:2305.10429): proxy-scale data decisions can transfer upward
- PreSelect (arXiv:2503.00808): some documents are more predictive than others
- SparseEval (arXiv:2602.07909): representative eval items can preserve ranking signal
"""

__version__ = "0.1.0"
