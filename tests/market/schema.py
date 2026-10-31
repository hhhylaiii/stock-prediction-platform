"""Introspection helpers shared by the standard-table tests."""

from django.db import connection


def primary_key_columns(table: str) -> list[list[str]]:
    with connection.cursor() as cursor:
        constraints = connection.introspection.get_constraints(cursor, table)
    return [info["columns"] for info in constraints.values() if info["primary_key"]]


def index_columns(table: str) -> list[list[str]]:
    with connection.cursor() as cursor:
        constraints = connection.introspection.get_constraints(cursor, table)
    return [info["columns"] for info in constraints.values() if info["index"]]
