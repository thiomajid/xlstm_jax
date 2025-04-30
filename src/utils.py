# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano
import math
from abc import ABC
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from flax import nnx


@dataclass
class UpProjConfigMixin:
    proj_factor: float = None  # will be overridden by subclasses
    round_proj_up_dim_up: bool = True
    round_proj_up_to_multiple_of: int = 64

    # internal
    _proj_up_dim: int = None  # will be computed from embedding_dim and proj_factor

    def _set_proj_up_dim(self, embedding_dim: int) -> None:
        if self.proj_factor is not None and embedding_dim is not None:
            proj_up_dim = self.proj_factor * embedding_dim
            multiple_of_multiplier = proj_up_dim / self.round_proj_up_to_multiple_of
            if self.round_proj_up_dim_up:
                multiple_of_multiplier = math.ceil(multiple_of_multiplier)
            else:
                multiple_of_multiplier = math.floor(multiple_of_multiplier)

            self._proj_up_dim = int(
                multiple_of_multiplier * self.round_proj_up_to_multiple_of
            )


class WeightDecayOptimGroupMixin(nnx.Module, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_weight_decay_optim_groups(
        self,
        **kwargs,
    ) -> Tuple[Sequence[nnx.Param], Sequence[nnx.Param]]:
        """Return a tuple of two sequences, one for parameters with weight decay and one for parameters without weight decay.
        Performs checks to ensure that each parameter is only in one of the two sequences.
        """
        weight_decay, no_weight_decay = self._create_weight_decay_optim_groups(**kwargs)

        # Check that parameters have been assigned correctly.
        # Each parameter can only be in one optim group.
        intersection_params = weight_decay.intersection(no_weight_decay)
        assert len(intersection_params) == 0, (
            f"parameters {['.'.join(pn) for pn, p in self.iter_modules() if p in intersection_params]} made it into both decay/no_decay sets!"
        )

        union_params = weight_decay.union(no_weight_decay)
        param_dict = {pn: p for pn, p in self.iter_modules()}
        unassigned_params = set(param_dict.values()) - union_params

        # We have parameters that were not assigned to either weight decay or no weight decay.
        # Filter out parameters that don't require gradients
        assert len(unassigned_params) == 0, (
            f"Parameters {['.'.join(pn) for pn, p in self.iter_modules() if all([p is not q for q in unassigned_params])]} were not separated into either decay/no_decay set!"
        )

        return weight_decay, no_weight_decay

    def get_weight_decay_optim_group_param_names(
        self,
        **kwargs,
    ) -> Tuple[List[str], List[str]]:
        """Return a tuple of two lists, one for parameter names with weight decay and one for parameter names without weight decay.
        Performs checks to ensure that each parameter is only in one of the two sequences.
        """

        def _is_in_sequence(param: nnx.Param, sequence: Sequence[nnx.Param]) -> bool:
            return any(param is p for p in sequence)

        weight_decay, no_weight_decay = self.get_weight_decay_optim_groups(**kwargs)

        names_weight_decay = [
            ".".join(name)
            for name, param in self.iter_modules()
            if _is_in_sequence(param, weight_decay)
        ]

        names_no_weight_decay = [
            ".".join(name)
            for name, param in self.iter_modules()
            if _is_in_sequence(param, no_weight_decay)
        ]

        return names_weight_decay, names_no_weight_decay

    def _create_weight_decay_optim_groups(
        self,
    ) -> tuple[set[nnx.Param], set[nnx.Param]]:
        """Return a tuple of two lists, one for parameters with weight decay and one for parameters without weight decay.
        Default separation:
        - weight decay: all parameters which have 'weight' in name and not 'norm'
        - no weight decay: all other parameters
        """
        decay = set()
        no_decay = set()

        # Iterate over the flattened list of (path, variable) pairs
        for path, node in self.iter_modules():
            # Only consider trainable parameters (instances of nnx.Param)
            if isinstance(node, nnx.Param):
                # Construct the full parameter name from the path (e.g., 'submodule.weight')
                name = ".".join([str(key) for key in path])
                # Apply the naming heuristic: weight decay for 'weight' params excluding 'norm'
                if "weight" in name and "norm" not in name:
                    decay.add(node)
                else:
                    no_decay.add(node)

        return decay, no_decay

    def _get_weight_decay_optim_groups_for_modules(
        self,
        modules: List["WeightDecayOptimGroupMixin"],
        **kwargs,
    ) -> Tuple[tuple[nnx.Param], tuple[nnx.Param]]:
        weight_decay = ()
        no_weight_decay = ()

        for module in modules:
            wd, nwd = module.get_weight_decay_optim_groups(**kwargs)
            weight_decay += wd
            no_weight_decay += nwd

        return weight_decay, no_weight_decay
