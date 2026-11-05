"""Query-string validation; a bad value answers 400 with the failing fields."""

from typing import Any

from rest_framework import serializers
from rest_framework.request import Request

from apps.market.models import Market


def parse(query: type[serializers.Serializer[Any]], request: Request) -> dict[str, Any]:
    serializer = query(data=request.query_params)
    serializer.is_valid(raise_exception=True)
    return dict(serializer.validated_data)


class PriceQuery(serializers.Serializer[Any]):
    days = serializers.IntegerField(min_value=1, max_value=500, default=60)


class RunQuery(serializers.Serializer[Any]):
    date = serializers.DateField(required=False)
    scope = serializers.ChoiceField(choices=["day", "covering"], default="day")


class QualityQuery(serializers.Serializer[Any]):
    days = serializers.IntegerField(min_value=1, max_value=120, default=30)


class PredictionQuery(serializers.Serializer[Any]):
    horizon = serializers.ChoiceField(choices=[5, 20], default=20)
    date = serializers.DateField(required=False)
    market = serializers.ChoiceField(choices=Market.values, required=False)
    q = serializers.CharField(max_length=32, required=False, allow_blank=True)
