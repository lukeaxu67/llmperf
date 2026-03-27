from llmperf.providers.base import create_provider
from llmperf.providers.openai_chat import (
    DeepSeekChatProvider,
    HuoshanChatProvider,
    HunyuanChatProvider,
    MoonshotChatProvider,
    OpenAIProvider,
    QianwenChatProvider,
    SparkChatProvider,
    ZhipuChatProvider,
)


def test_named_providers_use_dedicated_classes():
    expected_types = {
        "openai": OpenAIProvider,
        "qianwen": QianwenChatProvider,
        "zhipu": ZhipuChatProvider,
        "deepseek": DeepSeekChatProvider,
        "spark": SparkChatProvider,
        "hunyuan": HunyuanChatProvider,
        "huoshan": HuoshanChatProvider,
        "moonshot": MoonshotChatProvider,
    }

    for provider_name, provider_type in expected_types.items():
        provider = create_provider(provider_name, "default", provider_name)
        assert isinstance(provider, provider_type)
