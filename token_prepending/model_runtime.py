import torch


DEFAULT_MAC_7B_MODEL = "Qwen/Qwen2.5-7B"
DEFAULT_CACHE_DIR = ".cache/huggingface"


def resolve_torch_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    device = torch.device(device_name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is False.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def build_loading_kwargs(
    device_type: str,
    cache_dir=None,
    output_hidden_states: bool = True,
    trust_remote_code: bool = True,
    torch_dtype="auto",
):
    kwargs = {
        "output_hidden_states": output_hidden_states,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if torch_dtype:
        kwargs["torch_dtype"] = torch_dtype

    if device_type == "mps":
        kwargs["device_map"] = "mps"
    elif device_type == "cuda":
        kwargs["device_map"] = "auto"

    return kwargs


def move_model_to_device_if_needed(model, device: torch.device):
    if device.type == "cuda":
        return model
    if hasattr(model, "hf_device_map"):
        return model
    return model.to(device)
