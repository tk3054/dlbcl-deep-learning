"""Export helpers for formatted downstream image sets."""

__all__ = ["build_formatted_channels"]


def __getattr__(name: str):
    if name == "build_formatted_channels":
        from dlbcl_pipeline.export.formatted_channels import build_formatted_channels

        return build_formatted_channels
    raise AttributeError(name)
