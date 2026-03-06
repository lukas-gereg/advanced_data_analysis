import torch
from transformers import TrainingArguments


def privateuse_backend_name() -> str:
    return getattr(torch._C, "_get_privateuse1_backend_name", lambda: "")() or ""

def can_allocate_on(dev: torch.device | str) -> bool:
    try:
        torch.empty(0, device=dev)
        return True
    except Exception:
        return False

def _maybe_select_privateuse_device(dev: torch.device) -> None:
    """If privateuseone and an alias is registered, call its set_device once."""
    if dev.type != "privateuseone":
        return
    alias = privateuse_backend_name()
    if not alias:
        return
    mod = getattr(torch, alias, None)
    setter = getattr(mod, "set_device", None) if mod is not None else None
    if callable(setter):
        try:
            setter(dev)
        except Exception:
            pass

def resolve_preferred_device(spec) -> tuple[torch.device, bool, bool, str]:
    """
    Turn spec into a torch.device, try to activate private backend once,
    and classify it for Trainer behavior.
    Returns: (device, is_cpuish, is_usable, label)
    """
    dev = spec if isinstance(spec, torch.device) else torch.device(spec)
    label = dev.type

    if dev.type == "privateuseone":
        _maybe_select_privateuse_device(dev)
        alias = privateuse_backend_name()
        label = (alias or "privateuseone").lower()

    usable = can_allocate_on(dev)
    is_cpuish = (dev.type == "cpu") or (dev.type == "privateuseone")

    return dev, is_cpuish, usable, label


def _as_device(dev):
    if isinstance(dev, torch.device):
        return dev
    return torch.device(str(dev))  # accepts "privateuseone:0", "cuda:0", "ocl:0", etc.

class DeviceAwareAccelerator:
    """
    Thin proxy around a real Accelerate Accelerator that:
      • overrides .device to a user-supplied torch.device (e.g., privateuseone:0)
      • mirrors .state.device if possible (so .prepare uses the right device)
      • delegates EVERYTHING else (prepare, wrap, unwrap, backward, log, …) to the real accelerator,
        so behavior stays identical to upstream Accelerate.
    """

    def __init__(self, real_accelerator, force_device):
        self._real = real_accelerator
        self._forced_device = _as_device(force_device)

        # Best-effort: reflect the new device into the underlying state,
        # so internals that read state.device see the same thing.
        try:
            self._real.state.device = self._forced_device
        except Exception:
            pass

    @property
    def device(self):
        return self._forced_device

    def __getattr__(self, name):
        return getattr(self._real, name)

    def __repr__(self):
        return f"<DeviceAwareAccelerator device={self._forced_device} real={self._real!r}>"

def attach_device(accelerator, device):
    """Wrap an existing Accelerator so it reports/uses `device`."""
    return DeviceAwareAccelerator(accelerator, device)


class AnyDeviceTrainingArguments(TrainingArguments):
    """
    Accept any torch.device (string or torch.device).
    - Built-ins (CUDA/XPU/…): let Accelerate do its thing.
    - CPU or private/custom backends (privateuseone: OpenCL/DirectML…):
      keep HF in CPU semantics via `no_cuda=True`, but expose `args.device`
      as the requested device; we also call the backend’s set_device once.
    """
    def __init__(self, device, *args, **kwargs):
        dev, is_cpuish, usable, _ = resolve_preferred_device(device)
        self._specified_device = dev
        self._dev_is_usable = usable
        self._is_cpuish = is_cpuish

        if is_cpuish or not usable:
            # Prefer this over use_cpu: keeps semantics without toggling multiple knobs.
            kwargs.setdefault("use_cpu", True)
            # Optional but sensible on CPU/custom
            kwargs.setdefault("dataloader_pin_memory", False)

        super().__init__(*args, **kwargs)

    @property
    def device(self) -> torch.device:
        _ = super()._setup_devices

        if self._is_cpuish:
            return self._specified_device
        if self._dev_is_usable:
            return self._specified_device
        return super().device