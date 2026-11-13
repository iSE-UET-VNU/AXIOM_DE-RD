import os

from ..bench import BENCH, work
from ..qa import ANSWER_PROMPT, GENERATOR, GENERATOR_MAX_TOKENS, GENERATOR_PROVIDER, GENERATOR_REASONING, generate, \
    generator_key
from .llm import EndpointChat, OpenRouterChat
from .qa import cached_or_refuse
from .registry import register

QWEN_VL = os.getenv("QWEN_VL_MODEL") or "qwen/qwen3-vl-32b-instruct"
QWEN_VL_MAX_TOKENS = 2048


def answer_prompt(query, chunks):
    return ANSWER_PROMPT.format(documents="\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks)), query=query)


@register("qa_reader", "deepseek_v4_flash")
class LegacyReader:
    name = "deepseek_v4_flash"

    def __init__(self, argument=None, bench=BENCH, **_):
        self.folder = work("qa_cache", "answers", bench=bench)

    def settings(self):
        return {"reader": self.name, "model": GENERATOR, "provider": GENERATOR_PROVIDER,
                "reasoning": GENERATOR_REASONING, "max_tokens": GENERATOR_MAX_TOKENS, "seed": None,
                "input": "top chunks as text"}

    def read(self, query, chunks, online):
        prompt = answer_prompt(query, chunks)
        reply = cached_or_refuse(self.folder, generator_key() + prompt, lambda: generate(prompt), online,
                                 "the deepseek answer for this query")
        return {"answer": reply["answer"], "gen_seconds": reply["gen_seconds"],
                "served_model": reply.get("served_model"), "served_provider": reply.get("served_provider")}


@register("qa_reader", "qwen3_vl_32b")
class QwenReader:
    name = "qwen3_vl_32b"

    def __init__(self, argument=None, bench=BENCH, **_):
        backend = argument or "openrouter"
        if backend == "openrouter":
            self.client = OpenRouterChat(QWEN_VL, os.getenv("QWEN_VL_PROVIDER"), QWEN_VL_MAX_TOKENS)
        elif backend == "endpoint":
            self.client = EndpointChat(os.getenv("QWEN_VL_ENDPOINT"), os.getenv("QWEN_VL_SERVED") or "qwen3-vl-32b",
                                       QWEN_VL_MAX_TOKENS)
        else:
            raise SystemExit(f"qwen3_vl_32b backend {backend!r}; available: openrouter, endpoint")
        self.folder = work("qa_cache", f"answers_{self.name}_{backend}", bench=bench)

    def settings(self):
        return {"reader": self.name, **self.client.settings(), "input": "top chunks as text"}

    def read(self, query, chunks, online):
        prompt = answer_prompt(query, chunks)

        def call():
            reply = self.client(prompt)
            return {"answer": reply["text"], "gen_seconds": reply["seconds"], "served_model": reply["served_model"],
                    "served_provider": reply["served_provider"], "usage": reply["usage"]}

        return cached_or_refuse(self.folder, self.client.key() + prompt, call, online,
                                f"the {self.name} answer for this query")
