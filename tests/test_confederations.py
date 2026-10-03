"""Tests for the team → confederation mapping."""
from __future__ import annotations

import pytest

from mpp.confederations import get_confederation


@pytest.mark.parametrize(
    "team, conf",
    [
        # Dataset spells these "Saint …", the map used to say "St …".
        ("Saint Kitts and Nevis", "CONCACAF"),
        ("Saint Vincent and the Grenadines", "CONCACAF"),
        # Confederation members that used to fall into the fake "OTHER" pooling group.
        ("Israel", "UEFA"),
        ("Hong Kong", "AFC"),
        ("Taiwan", "AFC"),
        ("Bermuda", "CONCACAF"),
        ("Guadeloupe", "CONCACAF"),
        ("Tonga", "OFC"),
    ],
)
def test_confederation_members_are_mapped(team, conf):
    assert get_confederation(team) == conf


def test_non_fifa_sides_stay_other():
    assert get_confederation("Padania") == "OTHER"
