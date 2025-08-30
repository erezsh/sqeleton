from typing import Any, Dict
from abc import ABC, abstractmethod


class AbstractCompiler(ABC):
    @abstractmethod
    def compile(self, elem: Any, params: Dict[str, Any] = None) -> str: ...


class Compilable(ABC):
    "Marks an item as compilable. Needs to be implemented through multiple-dispatch."
