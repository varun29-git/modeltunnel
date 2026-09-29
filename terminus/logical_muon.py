"""Pinned Muon update with logical row slices for fused projection weights."""

import torch
from muon import SingleDeviceMuonWithAuxAdam, adam_update, muon_update


class LogicalMuonWithAuxAdam(SingleDeviceMuonWithAuxAdam):
    def __init__(self, param_groups, logical_rows):
        if len(param_groups) != len(logical_rows):
            raise ValueError("logical Muon metadata must match optimizer groups")
        # The pinned upstream constructor checks exact group keys. Add the
        # slice metadata after it has established the ordinary Muon groups.
        super().__init__(param_groups)
        for group, rows in zip(self.param_groups, logical_rows):
            if rows is None:
                continue
            if not group["use_muon"] or len(group["params"]) != 1:
                raise ValueError("logical row slices require one Muon matrix")
            parameter = group["params"][0]
            if parameter.ndim != 2 or sum(rows) != parameter.shape[0] or any(row <= 0 for row in rows):
                raise ValueError("logical row slices must partition a 2D matrix")
            group["logical_rows"] = tuple(rows)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                for parameter in group["params"]:
                    if parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter)
                    state = self.state[parameter]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(parameter)
                    momentum = state["momentum_buffer"]
                    start = 0
                    for rows in group.get("logical_rows", (parameter.shape[0],)):
                        stop = start + rows
                        update = muon_update(parameter.grad[start:stop], momentum[start:stop], beta=group["momentum"])
                        logical_weight = parameter[start:stop]
                        logical_weight.mul_(1 - group["lr"] * group["weight_decay"])
                        logical_weight.add_(update.reshape(logical_weight.shape), alpha=-group["lr"])
                        start = stop
            else:
                for parameter in group["params"]:
                    if parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter)
                    state = self.state[parameter]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(parameter)
                        state["exp_avg_sq"] = torch.zeros_like(parameter)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(parameter.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
                    parameter.add_(update, alpha=-group["lr"])
        return loss
