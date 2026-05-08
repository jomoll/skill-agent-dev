from .fastchat_client import FastChatAgent
from .http_agent import HTTPAgent
from .vertex_agent import VertexAgent
from .skill_aware_agent import SkillAwareAgent
from .post_verify_agent import PostVerifyingAgent

__all__ = [
    "FastChatAgent",
    "HTTPAgent",
    "VertexAgent",
    "SkillAwareAgent",
    "PostVerifyingAgent",
]


def __getattr__(name):
    if name == "LiteLLMAgent":
        from .litellm_agent import LiteLLMAgent

        return LiteLLMAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
