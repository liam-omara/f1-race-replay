import os
import fastf1

os.makedirs("cache", exist_ok=True)
fastf1.Cache.enable_cache("cache")


def get_race_names(year):
    """All Grand Prix names for a season."""
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    return schedule["EventName"].tolist()


def load_session(year, race, session_type="R"):
    """Load a session including position/car data (needed for the replay)."""
    session = fastf1.get_session(year, race, session_type)
    session.load(laps=True, telemetry=True, weather=False, messages=False)
    return session