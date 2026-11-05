from apps.api.feature_text import LABELS, UNITS
from apps.research.features.registry import FEATURES


def test_every_registered_feature_has_a_label_and_a_known_unit() -> None:
    assert set(LABELS) == {spec.name for spec in FEATURES}
    assert {unit for _, unit in LABELS.values()} <= set(UNITS)


def test_labels_are_distinct() -> None:
    labels = [label for label, _ in LABELS.values()]
    assert len(labels) == len(set(labels))


def test_point_differences_are_not_shown_as_percentages() -> None:
    assert {name for name, (_, unit) in LABELS.items() if unit == "points"} == {
        "fund_gm_yoy_chg",
        "fund_opm_yoy_chg",
        "ind_ep_neutral",
    }
