def main() -> None:
    """Entry point for the `minimax-h3-image` CLI."""
    # Lazy import to avoid circular import warning when run as `python -m`
    from minimax_h3_image.generate import main as _main

    _main()
