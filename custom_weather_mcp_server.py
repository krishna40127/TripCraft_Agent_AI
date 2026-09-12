"""
custom_weather_mcp_server.py
=============================
A custom MCP (Model Context Protocol) server that exposes one tool,
`get_weather_outlook`, giving the Weather Agent a real, live data source for
"is this a good time to visit this destination" checks -- using Open-Meteo
(https://open-meteo.com), which is free and requires no API key.

Two data sources are blended depending on how far out the trip is:
  - Trip starts within the next 16 days -> Open-Meteo's forecast API (real
    forecast for those exact dates).
  - Trip is further out (the common case for travel planning, e.g. "in
    December") -> Open-Meteo's historical archive API, pulling the same
    calendar dates from the previous year as a "typical climate" proxy.
    This is exactly what a human travel agent does when a real forecast
    isn't available yet ("December in Goa is usually dry and warm").

Run standalone for a quick manual check:
    python custom_weather_mcp_server.py --self-test "Goa,India" 2025-12-10 2025-12-14

Run as an MCP stdio server (what mcp_client.py talks to):
    python custom_weather_mcp_server.py
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

FORECAST_HORIZON_DAYS = 16

# WMO weather codes -> short human description (subset covering common cases)
WMO_CODES: dict[int, str] = {
    0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow", 77: "snow grains",
    80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    85: "slight snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}

mcp = FastMCP("tripcraft-weather")


# Open-Meteo's gazetteer indexes some well-known Indian tourist states/regions
# under an obscure, sparsely-populated village of the same name rather than
# their well-known capital/hub city, e.g. searching "Goa" returns a village
# in Rajasthan (~24.6N) instead of the coastal state (~15.3N) -- geocoding
# would silently return weather for the wrong place. A small curated alias
# table routes these to the city that actually represents the destination.
_DESTINATION_ALIASES = {
    "goa": "Panjim",
    "kashmir": "Srinagar",
    "ladakh": "Leh",
}

# Rough "how official/major is this place" ranking, best match wins ties.
_FEATURE_CODE_RANK = {
    "PPLC": 5,   # capital city
    "PPLA": 4,   # first-order admin seat
    "PPLA2": 3,
    "PPLA3": 2,
    "PPLA4": 1,
}


def _geocode(destination: str) -> dict[str, Any]:
    """Resolve a free-text destination to lat/lon via Open-Meteo geocoding.

    "Goa" alone is genuinely ambiguous -- Open-Meteo's gazetteer has a "Goa"
    in Russia, Botswana, Indonesia, the Philippines, India and more, several
    with no population figure to rank by, plus a much more populous *near*
    match ("Genoa", Italy) that a naive "first result" pick would grab
    instead. We score every candidate instead of trusting result order:
    exact name match, a country hint if the caller gave one (e.g.
    "Goa, India"), and how administratively significant the place is.
    """
    parts = [p.strip() for p in destination.split(",")]
    query_name = parts[0]
    country_hint = parts[1].lower() if len(parts) > 1 else None
    query_name = _DESTINATION_ALIASES.get(query_name.lower(), query_name)

    resp = httpx.get(GEOCODE_URL, params={"name": query_name, "count": 20}, timeout=10.0)
    resp.raise_for_status()
    results = resp.json().get("results") or []
    if not results:
        raise ValueError(f"Could not geocode destination '{destination}'.")

    def score(r: dict[str, Any]) -> float:
        s = 0.0
        if r.get("name", "").lower() == query_name.lower():
            s += 100.0
        if country_hint:
            country = (r.get("country") or "").lower()
            code = (r.get("country_code") or "").lower()
            if country_hint == country or country_hint == code:
                s += 1000.0
        s += _FEATURE_CODE_RANK.get(r.get("feature_code", ""), 0) * 5.0
        s += (r.get("population") or 0) / 1_000_000.0
        return s

    top = max(results, key=score)
    return {
        "name": top.get("name"),
        "country": top.get("country"),
        "latitude": top["latitude"],
        "longitude": top["longitude"],
        "timezone": top.get("timezone", "auto"),
    }


def _parse_date(d: str) -> date:
    return datetime.strptime(d, "%Y-%m-%d").date()


def _summarize_daily(daily: dict[str, Any]) -> dict[str, Any]:
    highs = [t for t in daily.get("temperature_2m_max", []) if t is not None]
    lows = [t for t in daily.get("temperature_2m_min", []) if t is not None]
    precip = [p for p in daily.get("precipitation_sum", []) if p is not None]
    codes = [c for c in daily.get("weathercode", []) if c is not None]

    avg_high = round(sum(highs) / len(highs), 1) if highs else None
    avg_low = round(sum(lows) / len(lows), 1) if lows else None
    total_precip = round(sum(precip), 1) if precip else 0.0
    rainy_days = sum(1 for p in precip if p and p >= 1.0)

    conditions = [WMO_CODES.get(c, "variable conditions") for c in codes]
    dominant = max(set(conditions), key=conditions.count) if conditions else "unknown"

    risk_flags: list[str] = []
    if avg_high is not None and avg_high >= 35:
        risk_flags.append("extreme heat risk")
    if avg_low is not None and avg_low <= 5:
        risk_flags.append("cold weather risk")
    num_days = len(daily.get("time", [])) or 1
    if rainy_days >= max(1, num_days // 2):
        risk_flags.append("monsoon/heavy rain risk")

    return {
        "avg_high_c": avg_high,
        "avg_low_c": avg_low,
        "total_precipitation_mm": total_precip,
        "rainy_days": rainy_days,
        "days_covered": num_days,
        "dominant_condition": dominant,
        "risk_flags": risk_flags,
    }


def _fetch_forecast(lat: float, lon: float, start: str, end: str) -> dict[str, Any]:
    resp = httpx.get(
        FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode",
            "timezone": "auto",
            "start_date": start,
            "end_date": end,
        },
        timeout=10.0,
    )
    resp.raise_for_status()
    return resp.json().get("daily", {})


def _fetch_archive(lat: float, lon: float, start: str, end: str) -> dict[str, Any]:
    resp = httpx.get(
        ARCHIVE_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode",
            "timezone": "auto",
            "start_date": start,
            "end_date": end,
        },
        timeout=15.0,
    )
    resp.raise_for_status()
    return resp.json().get("daily", {})


def get_weather_outlook_impl(destination: str, start_date: str, end_date: str) -> dict[str, Any]:
    """Core implementation, kept separate from the @mcp.tool wrapper so it can
    be unit-tested / called directly without going through the MCP protocol."""
    try:
        start = _parse_date(start_date)
        end = _parse_date(end_date)
    except ValueError as e:
        return {"ok": False, "error": f"Invalid date format, expected YYYY-MM-DD: {e}"}

    if end < start:
        return {"ok": False, "error": "end_date is before start_date."}

    try:
        location = _geocode(destination)
    except Exception as e:  # noqa: BLE001 - surface as a graceful tool error
        return {"ok": False, "error": f"Could not resolve destination '{destination}': {e}"}

    today = date.today()
    days_out = (start - today).days
    source: str

    try:
        if 0 <= days_out <= FORECAST_HORIZON_DAYS:
            daily = _fetch_forecast(location["latitude"], location["longitude"], start_date, end_date)
            source = "live_forecast"
        else:
            # Use the same calendar dates one year back as a typical-climate proxy.
            ref_start = (start - timedelta(days=365)).isoformat()
            ref_end = (end - timedelta(days=365)).isoformat()
            daily = _fetch_archive(location["latitude"], location["longitude"], ref_start, ref_end)
            source = "historical_climate_estimate"
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"Weather data temporarily unavailable ({e}). Try again shortly.",
        }

    if not daily or not daily.get("time"):
        return {"ok": False, "error": "Weather provider returned no data for these dates."}

    summary = _summarize_daily(daily)
    label = "Live forecast" if source == "live_forecast" else "Typical climate (based on last year's data for these dates)"

    text_bits = [
        f"{label} for {location['name']}, {location['country']}: ",
        f"avg high {summary['avg_high_c']}°C / avg low {summary['avg_low_c']}°C, ",
        f"mostly {summary['dominant_condition']}, ",
        f"{summary['rainy_days']} of {summary['days_covered']} days with meaningful rain.",
    ]
    if summary["risk_flags"]:
        text_bits.append(" Flags: " + ", ".join(summary["risk_flags"]) + ".")

    return {
        "ok": True,
        "source": source,
        "location": location,
        "summary": summary,
        "text": "".join(text_bits),
    }


@mcp.tool()
def get_weather_outlook(destination: str, start_date: str, end_date: str) -> dict[str, Any]:
    """Get a live forecast (trips <=16 days out) or a typical-climate estimate
    (trips further out, using last year's data for the same dates) for a
    destination and date range.

    Args:
        destination: City/region name, e.g. "Goa" or "Goa, India".
        start_date: Trip start date, ISO format YYYY-MM-DD.
        end_date: Trip end date, ISO format YYYY-MM-DD.
    """
    return get_weather_outlook_impl(destination, start_date, end_date)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        dest, sd, ed = sys.argv[2], sys.argv[3], sys.argv[4]
        import json

        print(json.dumps(get_weather_outlook_impl(dest, sd, ed), indent=2))
    else:
        mcp.run(transport="stdio")
