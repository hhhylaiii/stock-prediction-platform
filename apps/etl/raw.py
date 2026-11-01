"""Raw-layer writes shared by every source client (SPEC §4 raw layer)."""

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from apps.etl.models import RawRecord


def answer_hash(params: Mapping[str, Any], payload: Any) -> str:
    """Hash of the request and its response, so equal answers to different requests both stay."""
    answer = {"params": dict(params), "payload": payload}
    canonical = json.dumps(answer, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def store_raw(
    raw_model: type[RawRecord], dataset: str, params: Mapping[str, Any], payload: Any
) -> RawRecord:
    """Keep one row per distinct answer; asking the same and hearing the same adds nothing."""
    record, _ = raw_model._default_manager.get_or_create(
        dataset=dataset,
        source_hash=answer_hash(params, payload),
        defaults={"request_params": dict(params), "payload": payload},
    )
    return record
