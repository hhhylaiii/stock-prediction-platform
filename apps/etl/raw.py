"""Raw-layer writes shared by every source client (SPEC §4 raw layer)."""

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from apps.etl.models import RawRecord


def payload_hash(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def store_raw(
    raw_model: type[RawRecord], dataset: str, params: Mapping[str, Any], payload: Any
) -> RawRecord:
    """Keep one row per distinct response; refetching the same payload adds nothing."""
    record, _ = raw_model._default_manager.get_or_create(
        dataset=dataset,
        source_hash=payload_hash(payload),
        defaults={"request_params": dict(params), "payload": payload},
    )
    return record
