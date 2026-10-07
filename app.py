import streamlit as st
import streamlit.components.v1 as components

from core.loader import get_race_names, load_session
from core.player import PLAYER_HEIGHT, build_html
from core.replay import build_replay

st.set_page_config(page_title="F1 Race Replay", layout="wide")
st.title("🏁 F1 Race Replay")

YEARS = [2025, 2024, 2023, 2022, 2021, 2020, 2019]


@st.cache_data(show_spinner=False)
def race_options(year):
    return get_race_names(year)


@st.cache_resource(show_spinner="Loading race data and building the replay (the first time can take a few minutes)...")
def get_player_html(year, race):
    # 0.5 s grid: cars move smoothly because the browser interpolates between samples
    return build_html(build_replay(load_session(year, race), dt=0.5))


def show_player(html):
    """Newer Streamlit has st.iframe; older versions use components.html."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=PLAYER_HEIGHT)
    else:
        components.html(html, height=PLAYER_HEIGHT, scrolling=False)


c1, c2 = st.columns([1, 4])
with c1:
    year = st.selectbox("Year", YEARS, index=YEARS.index(2023))
with c2:
    try:
        race = st.selectbox("Race", race_options(year))
    except Exception:
        st.warning("Couldn't load the race calendar.")
        st.stop()

try:
    html = get_player_html(year, race)
except Exception as e:
    st.error(f"Couldn't build the replay for {race} {year}: {e}")
    st.stop()

show_player(html)
st.caption(
    "The replay runs in your browser, so playback is smooth. "
    "Timing tower comes from lap timing data; car positions and telemetry come from FastF1 car data."
)