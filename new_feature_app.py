import streamlit as st
import duckdb
import pandas as pd
import math
import os
import json
import re
from math import radians, cos, sin, asin, sqrt
from openai import OpenAI


if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OPENAI_API_KEY environment variable is not set.")
    st.stop()

# =====================================
# Streamlit Config
# =====================================
st.set_page_config(
    page_title="Dynamic Account Intelligence (DuckDB + Streamlit)",
    layout="wide",
)

st.title("🗺️ Dynamic Account Intelligence (DuckDB + Streamlit)")
st.write("Now uses online geocoding via LLM tool-calling.")

# =====================================
# Hard-Coded Prompt
# =====================================
USER_PROMPT = """
I am in Ames, Iowa and have two hours available.
Please provide me with 15 accounts that I could visit within 500 miles radius.
"""

st.subheader("📋 Active Prompt")
st.code(USER_PROMPT.strip())

# =====================================
# Prompt Parsing
# =====================================
def parse_prompt(prompt: str):
    city = state = None
    radius = 30
    limit = 3

    place = re.search(r"in\s+([A-Za-z\s]+),\s*([A-Za-z]+)", prompt)
    if place:
        city, state = place.group(1).strip(), place.group(2).strip()

    radius_match = re.search(r"(\d+)\s*mile", prompt, re.IGNORECASE)
    if radius_match:
        radius = int(radius_match.group(1))

    limit_match = re.search(r"(\d+)\s*account", prompt, re.IGNORECASE)
    if limit_match:
        limit = int(limit_match.group(1))

    return city, state, radius, limit


city, state, radius_miles, limit = parse_prompt(USER_PROMPT)

st.subheader("🔎 Interpreted Prompt Parameters")
st.json(
    {
        "city": city,
        "state": state,
        "radius_miles": radius_miles,
        "top_n_accounts": limit,
    }
)

# =====================================
# LLM Tool: Online Geocoding
# =====================================
geo_tool = [
    {
        "type": "function",
        "function": {
            "name": "extract_coordinates",
            "description": "Extract latitude and longitude for a city and state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
            },
        },
    }
]


def get_coordinates_from_llm(city: str, state: str):
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "user",
                "content": f"What are the latitude and longitude of {city}, {state}?",
            }
        ],
        tools=geo_tool,
        tool_choice={
            "type": "function",
            "function": {"name": "extract_coordinates"},
        },
    )

    tool_args_json = response.choices[0].message.tool_calls[0].function.arguments
    tool_args = json.loads(tool_args_json)

    return float(tool_args["latitude"]), float(tool_args["longitude"])


# =====================================
# Resolve User Coordinates
# =====================================
try:
    user_lat, user_lon = get_coordinates_from_llm(city, state)
    st.success(
        f"📍 User location resolved online: {city}, {state} "
        f"({user_lat:.4f}, {user_lon:.4f})"
    )
except Exception as e:
    st.error("❌ Failed to resolve user location using online geocoding.")
    st.stop()

# =====================================
# DuckDB Setup
# =====================================
@st.cache_resource
def init_duckdb():
    conn = duckdb.connect(database=":memory:")
    df = pd.read_csv("data/accounts.csv")
    conn.execute("CREATE TABLE accounts AS SELECT * FROM df")
    return conn


conn = init_duckdb()

# =====================================
# Distance Calculation
# =====================================
def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * asin(sqrt(a)) * R


accounts_df = conn.execute(
    """
    SELECT
        account,
        sector,
        city,
        state,
        latitude,
        longitude,
        revenue,
        employees
    FROM accounts
    """
).df()

accounts_df["distance_miles"] = accounts_df.apply(
    lambda r: haversine(user_lat, user_lon, r.latitude, r.longitude),
    axis=1,
)

results = (
    accounts_df[accounts_df["distance_miles"] <= radius_miles]
    .sort_values("distance_miles")
    .head(limit)
)

# =====================================
# Output
# =====================================
st.subheader("📍 Nearby Accounts")

if results.empty:
    st.warning("No accounts found within the specified radius.")
else:
    st.dataframe(
        results[
            [
                "account",
                "sector",
                "city",
                "state",
                "distance_miles",
                "revenue",
                "employees",
            ]
        ],
        use_container_width=True,
    )

st.success("✅ Online geocoding + DuckDB distance calculation completed.")
