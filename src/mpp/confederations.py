"""Static mapping from team name (as used in the historical dataset) to FIFA confederation."""
from __future__ import annotations

CONFEDERATION_MAP: dict[str, str] = {
    # UEFA
    "Germany": "UEFA", "France": "UEFA", "Spain": "UEFA", "England": "UEFA",
    "Netherlands": "UEFA", "Belgium": "UEFA", "Portugal": "UEFA", "Switzerland": "UEFA",
    "Czech Republic": "UEFA", "Croatia": "UEFA", "Austria": "UEFA", "Scotland": "UEFA",
    "Sweden": "UEFA", "Norway": "UEFA", "Bosnia and Herzegovina": "UEFA",
    "Italy": "UEFA", "Russia": "UEFA", "Poland": "UEFA", "Ukraine": "UEFA",
    "Serbia": "UEFA", "Denmark": "UEFA", "Greece": "UEFA", "Hungary": "UEFA",
    "Romania": "UEFA", "Slovakia": "UEFA", "Slovenia": "UEFA", "Finland": "UEFA",
    "Northern Ireland": "UEFA", "Wales": "UEFA", "Republic of Ireland": "UEFA",
    "Iceland": "UEFA", "Albania": "UEFA", "Montenegro": "UEFA",
    "North Macedonia": "UEFA", "Kosovo": "UEFA", "Bulgaria": "UEFA",
    "Georgia": "UEFA", "Armenia": "UEFA", "Azerbaijan": "UEFA", "Belarus": "UEFA",
    "Cyprus": "UEFA", "Estonia": "UEFA", "Faroe Islands": "UEFA", "Latvia": "UEFA",
    "Lithuania": "UEFA", "Luxembourg": "UEFA", "Malta": "UEFA", "Moldova": "UEFA",
    "Turkey": "UEFA", "Kazakhstan": "UEFA", "Liechtenstein": "UEFA",
    "Andorra": "UEFA", "Gibraltar": "UEFA", "San Marino": "UEFA",
    # CONMEBOL
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Colombia": "CONMEBOL",
    "Uruguay": "CONMEBOL", "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL",
    "Chile": "CONMEBOL", "Peru": "CONMEBOL", "Bolivia": "CONMEBOL", "Venezuela": "CONMEBOL",
    # CONCACAF
    "Mexico": "CONCACAF", "United States": "CONCACAF", "Canada": "CONCACAF",
    "Haiti": "CONCACAF", "Panama": "CONCACAF",
    "Curaçao": "CONCACAF", "Curacao": "CONCACAF",
    "Costa Rica": "CONCACAF", "Honduras": "CONCACAF", "Jamaica": "CONCACAF",
    "El Salvador": "CONCACAF", "Guatemala": "CONCACAF", "Trinidad and Tobago": "CONCACAF",
    "Cuba": "CONCACAF", "Belize": "CONCACAF", "Barbados": "CONCACAF",
    "Grenada": "CONCACAF", "Dominican Republic": "CONCACAF", "Dominica": "CONCACAF",
    "Suriname": "CONCACAF", "Guyana": "CONCACAF", "Nicaragua": "CONCACAF",
    "Antigua and Barbuda": "CONCACAF", "Saint Lucia": "CONCACAF",
    "St Kitts and Nevis": "CONCACAF", "St Vincent and the Grenadines": "CONCACAF",
    "Puerto Rico": "CONCACAF",
    # CAF
    "Morocco": "CAF", "Senegal": "CAF", "Algeria": "CAF", "Egypt": "CAF",
    "Tunisia": "CAF", "Ghana": "CAF", "Cape Verde": "CAF", "Ivory Coast": "CAF",
    "DR Congo": "CAF", "South Africa": "CAF", "Nigeria": "CAF", "Cameroon": "CAF",
    "Mali": "CAF", "Guinea": "CAF", "Burkina Faso": "CAF", "Uganda": "CAF",
    "Tanzania": "CAF", "Kenya": "CAF", "Zimbabwe": "CAF", "Zambia": "CAF",
    "Angola": "CAF", "Ethiopia": "CAF", "Mozambique": "CAF", "Libya": "CAF",
    "Congo": "CAF", "Equatorial Guinea": "CAF", "Gabon": "CAF",
    "Sierra Leone": "CAF", "Liberia": "CAF", "Guinea-Bissau": "CAF",
    "Gambia": "CAF", "Mauritania": "CAF", "Niger": "CAF", "Togo": "CAF",
    "Benin": "CAF", "Rwanda": "CAF", "Burundi": "CAF", "Madagascar": "CAF",
    "Malawi": "CAF", "Namibia": "CAF", "Botswana": "CAF", "Lesotho": "CAF",
    "Eswatini": "CAF", "Swaziland": "CAF", "Comoros": "CAF",
    "Sudan": "CAF", "South Sudan": "CAF", "Chad": "CAF",
    "Central African Republic": "CAF", "Djibouti": "CAF", "Eritrea": "CAF",
    "Somalia": "CAF", "Mauritius": "CAF", "Seychelles": "CAF",
    "São Tomé and Príncipe": "CAF", "Cape Verde Islands": "CAF",
    # AFC
    "South Korea": "AFC", "Japan": "AFC", "Australia": "AFC", "Saudi Arabia": "AFC",
    "Iran": "AFC", "Iraq": "AFC", "Qatar": "AFC", "Jordan": "AFC", "Uzbekistan": "AFC",
    "China": "AFC", "India": "AFC", "United Arab Emirates": "AFC", "Bahrain": "AFC",
    "Kuwait": "AFC", "Oman": "AFC", "Yemen": "AFC", "Vietnam": "AFC",
    "Thailand": "AFC", "Indonesia": "AFC", "Malaysia": "AFC", "Philippines": "AFC",
    "Myanmar": "AFC", "Singapore": "AFC", "Nepal": "AFC", "Sri Lanka": "AFC",
    "Bangladesh": "AFC", "Pakistan": "AFC", "Afghanistan": "AFC", "Tajikistan": "AFC",
    "Kyrgyzstan": "AFC", "Turkmenistan": "AFC", "Mongolia": "AFC", "North Korea": "AFC",
    "Chinese Taipei": "AFC", "Palestine": "AFC", "Lebanon": "AFC", "Syria": "AFC",
    "Cambodia": "AFC", "Laos": "AFC", "Maldives": "AFC", "Bhutan": "AFC",
    "Timor-Leste": "AFC",
    # OFC
    "New Zealand": "OFC", "Papua New Guinea": "OFC", "Solomon Islands": "OFC",
    "Fiji": "OFC", "Vanuatu": "OFC", "Samoa": "OFC", "Tahiti": "OFC",
    "New Caledonia": "OFC", "Cook Islands": "OFC",
}


def get_confederation(team: str) -> str:
    """Return the confederation for a team name, or 'OTHER' if unknown."""
    return CONFEDERATION_MAP.get(team, "OTHER")
