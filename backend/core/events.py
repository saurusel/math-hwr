# Placeholder for event schema helpers if needed later
from typing import Dict, Any
def make_event(kind: str, **kwargs) -> Dict[str, Any]:
    out = {"event": kind}
    out.update(kwargs)
    return out
