from apps.api.feature_text import GROUP_LABELS, LABELS, UNITS
from apps.research.features.registry import FEATURES, Group


def test_every_registered_feature_has_a_label_and_a_known_unit() -> None:
    assert set(LABELS) == {spec.name for spec in FEATURES}
    assert {unit for _, unit in LABELS.values()} <= set(UNITS)


def test_labels_are_distinct() -> None:
    labels = [label for label, _ in LABELS.values()]
    assert len(labels) == len(set(labels))


def test_every_group_has_a_label() -> None:
    assert set(GROUP_LABELS) == {group.value for group in Group}
