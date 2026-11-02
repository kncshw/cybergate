"""The cross-process decision store, as a protocol.

Two implementations: `ipc_shm.ShmDecisionStore` (the real one, `/dev/shm`) and
`memory.MemoryDecisionStore` (in-process dict, for tests). Making this an explicit named layer is the
point of REFACTOR_PLAN.md §2.1 -- in the canonical package the `/dev/shm` reads and writes are
scattered through the policy, which is where the whole class of state bugs comes from.

No TTL and no consume-on-read: neither exists in the canonical package today, and this port does not
add behaviour. Phase 6 adds `ttl_s` to `put` and a consuming `take`, which is why they are named here
but not declared.
"""
from __future__ import annotations

from typing import Any, Iterator, Protocol, runtime_checkable

__all__ = ["DecisionStore"]


@runtime_checkable
class DecisionStore(Protocol):
    """A keyed store of gate-decision records shared across processes.

    Keys come from `state.keys`. Values are the plain decision dicts the worker hook builds.

    Every method must swallow its own I/O errors: implementations are called from the model-worker
    forward path, where an exception would propagate into the engine. A failed `put` is lost
    silently, and a failed `get` returns `None`.
    """

    def get(self, key: str) -> dict[str, Any] | None:
        """Return the record stored under `key`, or `None` if absent or unreadable."""
        ...

    def put(self, key: str, record: dict[str, Any]) -> None:
        """Store `record` under `key`, replacing any existing record atomically."""
        ...

    def delete(self, key: str) -> None:
        """Remove `key`. A missing key is not an error."""
        ...

    def iter_keys(self) -> Iterator[str]:
        """Yield every key currently stored."""
        ...
