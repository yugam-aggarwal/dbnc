"""DBNC ablation/variant parameterisation.

``dbnc_variant_parameters`` maps a variant name (e.g. ``"full"``,
``"conditional_only"``, ``"fixed_tan"``, ``"mlp_cpd"``, the ``loss_w_*`` sweep)
to the keyword arguments that configure a :class:`dbnc.model.DBNC` instance.
"""
from dbnc._pipeline import dbnc_variant_parameters  # noqa: F401

__all__ = ["dbnc_variant_parameters"]
