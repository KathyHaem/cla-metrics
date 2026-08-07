def get_attn_implementation() -> str:
    """Use flash-attention if the package is installed, else fall back to PyTorch's SDPA."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"
