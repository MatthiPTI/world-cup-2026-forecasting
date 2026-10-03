"""World Cup 2026 group stage fixtures and team name mappings."""
from __future__ import annotations

# Maps display names (from official fixtures) to names used in the historical dataset.
TEAM_NAME_MAP: dict[str, str] = {
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
}

# All 72 group-stage matches. team_a is listed first in each fixture.
WC2026_FIXTURES: list[dict] = [
    # ── Group A ──────────────────────────────────────────────────────────────
    {"group": "A", "date": "2026-06-11", "team_a": "Mexico", "team_b": "South Africa"},
    {"group": "A", "date": "2026-06-11", "team_a": "South Korea", "team_b": "Czechia"},
    {"group": "A", "date": "2026-06-18", "team_a": "Czechia", "team_b": "South Africa"},
    {"group": "A", "date": "2026-06-18", "team_a": "Mexico", "team_b": "South Korea"},
    {"group": "A", "date": "2026-06-24", "team_a": "South Africa", "team_b": "South Korea"},
    {"group": "A", "date": "2026-06-24", "team_a": "Czechia", "team_b": "Mexico"},
    # ── Group B ──────────────────────────────────────────────────────────────
    {"group": "B", "date": "2026-06-12", "team_a": "Canada", "team_b": "Bosnia and Herzegovina"},
    {"group": "B", "date": "2026-06-13", "team_a": "Qatar", "team_b": "Switzerland"},
    {"group": "B", "date": "2026-06-18",
     "team_a": "Switzerland", "team_b": "Bosnia and Herzegovina"},
    {"group": "B", "date": "2026-06-18", "team_a": "Canada", "team_b": "Qatar"},
    {"group": "B", "date": "2026-06-24", "team_a": "Switzerland", "team_b": "Canada"},
    {"group": "B", "date": "2026-06-24", "team_a": "Bosnia and Herzegovina", "team_b": "Qatar"},
    # ── Group C ──────────────────────────────────────────────────────────────
    {"group": "C", "date": "2026-06-13", "team_a": "Brazil", "team_b": "Morocco"},
    {"group": "C", "date": "2026-06-13", "team_a": "Haiti", "team_b": "Scotland"},
    {"group": "C", "date": "2026-06-19", "team_a": "Scotland", "team_b": "Morocco"},
    {"group": "C", "date": "2026-06-19", "team_a": "Brazil", "team_b": "Haiti"},
    {"group": "C", "date": "2026-06-24", "team_a": "Scotland", "team_b": "Brazil"},
    {"group": "C", "date": "2026-06-24", "team_a": "Morocco", "team_b": "Haiti"},
    # ── Group D ──────────────────────────────────────────────────────────────
    {"group": "D", "date": "2026-06-12", "team_a": "United States", "team_b": "Paraguay"},
    {"group": "D", "date": "2026-06-13", "team_a": "Australia", "team_b": "Türkiye"},
    {"group": "D", "date": "2026-06-19", "team_a": "United States", "team_b": "Australia"},
    {"group": "D", "date": "2026-06-19", "team_a": "Türkiye", "team_b": "Paraguay"},
    {"group": "D", "date": "2026-06-25", "team_a": "Türkiye", "team_b": "United States"},
    {"group": "D", "date": "2026-06-25", "team_a": "Paraguay", "team_b": "Australia"},
    # ── Group E ──────────────────────────────────────────────────────────────
    {"group": "E", "date": "2026-06-14", "team_a": "Germany", "team_b": "Curaçao"},
    {"group": "E", "date": "2026-06-14", "team_a": "Ivory Coast", "team_b": "Ecuador"},
    {"group": "E", "date": "2026-06-20", "team_a": "Germany", "team_b": "Ivory Coast"},
    {"group": "E", "date": "2026-06-20", "team_a": "Ecuador", "team_b": "Curaçao"},
    {"group": "E", "date": "2026-06-25", "team_a": "Ecuador", "team_b": "Germany"},
    {"group": "E", "date": "2026-06-25", "team_a": "Curaçao", "team_b": "Ivory Coast"},
    # ── Group F ──────────────────────────────────────────────────────────────
    {"group": "F", "date": "2026-06-14", "team_a": "Netherlands", "team_b": "Japan"},
    {"group": "F", "date": "2026-06-14", "team_a": "Sweden", "team_b": "Tunisia"},
    {"group": "F", "date": "2026-06-20", "team_a": "Netherlands", "team_b": "Sweden"},
    {"group": "F", "date": "2026-06-20", "team_a": "Tunisia", "team_b": "Japan"},
    {"group": "F", "date": "2026-06-25", "team_a": "Japan", "team_b": "Sweden"},
    {"group": "F", "date": "2026-06-25", "team_a": "Tunisia", "team_b": "Netherlands"},
    # ── Group G ──────────────────────────────────────────────────────────────
    {"group": "G", "date": "2026-06-15", "team_a": "Belgium", "team_b": "Egypt"},
    {"group": "G", "date": "2026-06-15", "team_a": "Iran", "team_b": "New Zealand"},
    {"group": "G", "date": "2026-06-21", "team_a": "Belgium", "team_b": "Iran"},
    {"group": "G", "date": "2026-06-21", "team_a": "New Zealand", "team_b": "Egypt"},
    {"group": "G", "date": "2026-06-26", "team_a": "Egypt", "team_b": "Iran"},
    {"group": "G", "date": "2026-06-26", "team_a": "New Zealand", "team_b": "Belgium"},
    # ── Group H ──────────────────────────────────────────────────────────────
    {"group": "H", "date": "2026-06-15", "team_a": "Spain", "team_b": "Cape Verde"},
    {"group": "H", "date": "2026-06-15", "team_a": "Saudi Arabia", "team_b": "Uruguay"},
    {"group": "H", "date": "2026-06-21", "team_a": "Spain", "team_b": "Saudi Arabia"},
    {"group": "H", "date": "2026-06-21", "team_a": "Uruguay", "team_b": "Cape Verde"},
    {"group": "H", "date": "2026-06-26", "team_a": "Cape Verde", "team_b": "Saudi Arabia"},
    {"group": "H", "date": "2026-06-26", "team_a": "Uruguay", "team_b": "Spain"},
    # ── Group I ──────────────────────────────────────────────────────────────
    {"group": "I", "date": "2026-06-16", "team_a": "France", "team_b": "Senegal"},
    {"group": "I", "date": "2026-06-16", "team_a": "Iraq", "team_b": "Norway"},
    {"group": "I", "date": "2026-06-22", "team_a": "France", "team_b": "Iraq"},
    {"group": "I", "date": "2026-06-22", "team_a": "Norway", "team_b": "Senegal"},
    {"group": "I", "date": "2026-06-26", "team_a": "Norway", "team_b": "France"},
    {"group": "I", "date": "2026-06-26", "team_a": "Senegal", "team_b": "Iraq"},
    # ── Group J ──────────────────────────────────────────────────────────────
    {"group": "J", "date": "2026-06-16", "team_a": "Argentina", "team_b": "Algeria"},
    {"group": "J", "date": "2026-06-16", "team_a": "Austria", "team_b": "Jordan"},
    {"group": "J", "date": "2026-06-22", "team_a": "Argentina", "team_b": "Austria"},
    {"group": "J", "date": "2026-06-22", "team_a": "Jordan", "team_b": "Algeria"},
    {"group": "J", "date": "2026-06-27", "team_a": "Algeria", "team_b": "Austria"},
    {"group": "J", "date": "2026-06-27", "team_a": "Jordan", "team_b": "Argentina"},
    # ── Group K ──────────────────────────────────────────────────────────────
    {"group": "K", "date": "2026-06-17", "team_a": "Portugal", "team_b": "DR Congo"},
    {"group": "K", "date": "2026-06-17", "team_a": "Uzbekistan", "team_b": "Colombia"},
    {"group": "K", "date": "2026-06-23", "team_a": "Portugal", "team_b": "Uzbekistan"},
    {"group": "K", "date": "2026-06-23", "team_a": "Colombia", "team_b": "DR Congo"},
    {"group": "K", "date": "2026-06-27", "team_a": "Colombia", "team_b": "Portugal"},
    {"group": "K", "date": "2026-06-27", "team_a": "DR Congo", "team_b": "Uzbekistan"},
    # ── Group L ──────────────────────────────────────────────────────────────
    {"group": "L", "date": "2026-06-17", "team_a": "England", "team_b": "Croatia"},
    {"group": "L", "date": "2026-06-17", "team_a": "Ghana", "team_b": "Panama"},
    {"group": "L", "date": "2026-06-23", "team_a": "England", "team_b": "Ghana"},
    {"group": "L", "date": "2026-06-23", "team_a": "Panama", "team_b": "Croatia"},
    {"group": "L", "date": "2026-06-27", "team_a": "Panama", "team_b": "England"},
    {"group": "L", "date": "2026-06-27", "team_a": "Croatia", "team_b": "Ghana"},
]
