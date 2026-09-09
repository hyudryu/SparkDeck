"""Codex model metadata alongside the standard OpenAI model catalog.

The wire contract is ModelInfo in openai/codex's
codex-rs/protocol/src/openai_models.rs. OpenAI model objects alone do not
deserialize as Codex models. Advertise only capabilities the gateway knows.
"""

from typing import Any


_INSTRUCTIONS = (
    "You are a coding assistant. Help the user complete their task using the "
    "available tools. Follow the user's instructions, inspect relevant code "
    "before editing, and verify your changes."
)


def codex_model(model_id: str) -> dict[str, Any]:
    """Describe one routable model without guessing runtime capabilities."""
    return {
        "slug": model_id,
        "display_name": model_id,
        "description": "Model served through SparkDeck",
        "default_reasoning_level": None,
        "supported_reasoning_levels": [],
        "shell_type": "unified_exec",
        "visibility": "list",
        "supported_in_api": True,
        "priority": 0,
        "availability_nux": None,
        "upgrade": None,
        # Older Codex versions require base_instructions; newer versions use
        # model_messages. Keep both until older clients are no longer supported.
        "base_instructions": _INSTRUCTIONS,
        "model_messages": {"instructions_template": _INSTRUCTIONS},
        "supports_reasoning_summaries": False,
        "supports_reasoning_summary_parameter": False,
        "support_verbosity": False,
        "default_verbosity": None,
        "apply_patch_tool_type": None,
        "truncation_policy": {"mode": "bytes", "limit": 10000},
        "context_window": None,
        "experimental_supported_tools": [],
        "input_modalities": ["text"],
        "supports_parallel_tool_calls": False,
    }
