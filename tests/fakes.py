"""Shared fake LLM / Redis for unit tests."""

from __future__ import annotations

from typing import Any


class FakeLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    def invoke(self, _messages: Any) -> Any:
        self.calls += 1
        return type("R", (), {"content": self.content})()

    def stream(self, messages: Any):
        """步骤：01 把 invoke 结果当单块吐出，模拟 ChatOpenAI.stream（协议名不能改）。"""
        yield self.invoke(messages)

    async def astream(self, messages: Any):
        """步骤：01 异步吐出 invoke 结果，模拟 ChatOpenAI.astream（协议名不能改）。"""
        yield self.invoke(messages)


class BoomLLM:
    def invoke(self, _messages: Any) -> Any:
        raise RuntimeError("llm down")


class FakeRedis:
    """Minimal List + ping/delete compatible with ConversationStore."""

    def __init__(self) -> None:
        self.lists: dict[str, list[str]] = {}
        self.fail = False

    def ping(self) -> bool:
        if self.fail:
            raise ConnectionError("redis down")
        return True

    def rpush(self, key: str, *values: str) -> int:
        if self.fail:
            raise ConnectionError("redis down")
        self.lists.setdefault(key, []).extend(values)
        return len(self.lists[key])

    def ltrim(self, key: str, start: int, end: int) -> bool:
        if self.fail:
            raise ConnectionError("redis down")
        lst = self.lists.get(key, [])
        n = len(lst)
        if n == 0:
            return True
        s = start if start >= 0 else n + start
        e = end if end >= 0 else n + end
        s = max(0, s)
        e = min(n - 1, e)
        if s > e:
            self.lists[key] = []
        else:
            self.lists[key] = lst[s : e + 1]
        return True

    def lrange(self, key: str, start: int, end: int) -> list[str]:
        if self.fail:
            raise ConnectionError("redis down")
        lst = self.lists.get(key, [])
        n = len(lst)
        if n == 0:
            return []
        s = start if start >= 0 else n + start
        e = end if end >= 0 else n + end
        s = max(0, s)
        e = min(n - 1, e)
        if s > e:
            return []
        return lst[s : e + 1]

    def delete(self, *keys: str) -> int:
        if self.fail:
            raise ConnectionError("redis down")
        count = 0
        for key in keys:
            if key in self.lists:
                del self.lists[key]
                count += 1
        return count
