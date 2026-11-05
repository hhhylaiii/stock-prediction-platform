"""Query-string validation; a bad value answers 400 with the failing fields."""

from typing import Any

from rest_framework import serializers
from rest_framework.request import Request


def parse(query: type[serializers.Serializer[Any]], request: Request) -> dict[str, Any]:
    serializer = query(data=request.query_params)
    serializer.is_valid(raise_exception=True)
    return dict(serializer.validated_data)


class PriceQuery(serializers.Serializer[Any]):
    days = serializers.IntegerField(min_value=1, max_value=500, default=60)


class RunQuery(serializers.Serializer[Any]):
    date = serializers.DateField(required=False)


class QualityQuery(serializers.Serializer[Any]):
    days = serializers.IntegerField(min_value=1, max_value=120, default=30)
