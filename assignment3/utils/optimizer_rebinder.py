import torch

def _flat_params(groups):
    for g in groups:
        for p in g["params"]:
            if p is not None:
                yield p

def optimizer_is_synced(model: torch.nn.Module, opt: torch.optim.Optimizer) -> bool:
    """Returns True if every param in the optimizer is exactly the same tensor object
    as the model's current parameters."""
    ids_model = [id(p) for p in model.parameters()]
    ids_opt   = [id(p) for p in _flat_params(opt.param_groups)]
    return len(ids_model) == len(ids_opt) and all(a == b for a, b in zip(ids_model, ids_opt))

def rebuild_optimizer_like(old_opt_cls, new_param_groups, old_opt):
    """
    Construct a new optimizer of the same class with identical hyperparameters
    (including per-group options like lr, weight_decay, betas, etc.).
    """
    # Clone group options but swap in new params with the same group sizes/order
    new_groups = []
    flat_new = list(_flat_params(new_param_groups))
    flat_idx = 0
    for g in old_opt.param_groups:
        count = len(g["params"])
        slice_params = flat_new[flat_idx:flat_idx + count]
        flat_idx += count

        new_g = {k: v for k, v in g.items() if k != "params"}
        new_g["params"] = slice_params
        new_groups.append(new_g)

    # Many optimizers accept a list of param group dicts directly
    defaults = dict(old_opt.defaults)
    # (Optional) For DirectML backends, ensure foreach=False if present
    if "foreach" in defaults:
        defaults["foreach"] = False

    new_opt = type(old_opt)(new_groups, **{k: v for k, v in defaults.items() if k != "params"})
    return new_opt

def transfer_optimizer_state(old_opt, new_opt):
    """
    Transfer optimizer state from old params to new params by **position**.
    Works if the model structure & parameter order are unchanged.
    """
    old_flat = list(_flat_params(old_opt.param_groups))
    new_flat = list(_flat_params(new_opt.param_groups))

    for p_old, p_new in zip(old_flat, new_flat):
        if p_old in old_opt.state:
            state = old_opt.state[p_old]
            # Move any tensor state to the param's device
            new_state = {}
            for k, v in state.items():
                if torch.is_tensor(v):
                    new_state[k] = v.detach().to(p_new.device)
                else:
                    new_state[k] = v
            new_opt.state[p_new] = new_state

def ensure_optimizer_matches_model(model: torch.nn.Module,
                                   opt: torch.optim.Optimizer,
                                   preserve_state: bool = True) -> torch.optim.Optimizer:
    """
    If the optimizer isn't pointing at the model's current tensors, rebuild it so it is.
    Optionally carry over state (Adam moments, steps) by param order.
    """
    # Quick device sanity: all model params should share device
    devices = {p.device.type for p in model.parameters()}
    if len(devices) != 1:
        print(f"[warn] Model has mixed devices: {devices}")

    if optimizer_is_synced(model, opt):
        return opt

    # Build a "mirror" optimizer that points at the model's current params
    new_opt = rebuild_optimizer_like(type(opt), model.param_groups if hasattr(model, "param_groups") else [{"params": list(model.parameters())}], opt)

    # If the above feels too magical, use this simpler constructor instead:
    # new_opt = type(opt)(model.parameters(), **{k:v for k,v in opt.defaults.items() if k != 'params'})

    if preserve_state:
        try:
            transfer_optimizer_state(opt, new_opt)
        except Exception as e:
            print(f"[warn] Failed to transfer optimizer state: {e}. Starting fresh state.")

    return new_opt
