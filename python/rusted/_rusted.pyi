from collections.abc import Callable
from typing import Any

SPEC_ABI: int
BUILD_PROFILE: str

class Compiled:
    def load(self, data: Any, format: str = 'json') -> Any: ...
    def dump(self, obj: Any, format: str = 'json') -> dict[str, Any]: ...

def compile_spec(
    spec: dict[str, Any],
) -> tuple[Callable[..., Any], Callable[..., Any]] | None: ...
