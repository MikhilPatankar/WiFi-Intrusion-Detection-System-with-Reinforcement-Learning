# src/dashboard/dashboard.py

import streamlit as st
import requests
import pandas as pd
import datetime
import os
import logging
from streamlit_autorefresh import st_autorefresh # Optional: Install with pip install streamlit-autorefresh

# --- Configuration ---
# Get API URL from environment variable or use a default
# Make sure the backend is running and accessible from where you run Streamlit
API_BASE_URL = os.getenv("WIDS_API_URL", "http://127.0.0.1:8000")
EVENTS_ENDPOINT = f"{API_BASE_URL}/events/"
LABEL_ENDPOINT = f"{API_BASE_URL}/events/label"
REFRESH_INTERVAL_SECONDS = 60 # Auto-refresh interval (e.g., 60 seconds)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions for API Interaction ---

# Use Streamlit's caching to avoid re-fetching data on every interaction
# TTL sets how long the cache is valid (in seconds)
@st.cache_data(ttl=30)
def get_events(skip: int = 0, limit: int = 100, unlabeled_only: bool = False):
    """Fetches event logs from the backend API."""
    params = {"skip": skip, "limit": limit, "unlabeled_only": unlabeled_only}
    try:
        response = requests.get(EVENTS_ENDPOINT, params=params, timeout=10) # Added timeout
        response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
        data = response.json()
        logging.info(f"Fetched {len(data.get('events', []))} events (total: {data.get('total_count', 0)}) from API.")
        # Convert list of event dicts to DataFrame
        if data.get("events"):
            df = pd.DataFrame(data["events"])
            # Convert timestamp string to datetime objects if needed (depends on API output)
            if 'timestamp' in df.columns:
                 df['timestamp'] = pd.to_datetime(df['timestamp'])
            if 'label_timestamp' in df.columns:
                 df['label_timestamp'] = pd.to_datetime(df['label_timestamp'])
            return df, data.get("total_count", 0)
        else:
            return pd.DataFrame(), 0
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {e}")
        logging.error(f"API connection error: {e}", exc_info=True)
        return pd.DataFrame(), 0
    except Exception as e:
        st.error(f"An error occurred while fetching events: {e}")
        logging.error(f"Error fetching events: {e}", exc_info=True)
        return pd.DataFrame(), 0

def submit_label(event_uid: str, label: str):
    """Submits a human label for an event to the backend API."""
    payload = {"event_uid": event_uid, "human_label": label}
    try:
        response = requests.post(LABEL_ENDPOINT, json=payload, timeout=10)
        response.raise_for_status()
        logging.info(f"Submitted label '{label}' for event UID: {event_uid}. Status: {response.status_code}")
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error submitting label: {e}")
        logging.error(f"API error submitting label for {event_uid}: {e}", exc_info=True)
        return None
    except Exception as e:
        st.error(f"An error occurred while submitting label: {e}")
        logging.error(f"Error submitting label: {e}", exc_info=True)
        return None

# --- Streamlit App Layout ---

st.set_page_config(page_title="WIDS Monitoring Dashboard", layout="wide")

st.title("📶 WiFi Intrusion Detection System Dashboard")
st.caption(f"Connected to API: {API_BASE_URL}")

# --- Auto-Refresh ---
# Optional: Auto-refresh the page periodically
refresh_count = st_autorefresh(interval=REFRESH_INTERVAL_SECONDS * 1000, key="datarefresh")
# Or use a manual refresh button:
# if st.button("Refresh Data"):
#     st.cache_data.clear() # Clear cache on manual refresh
#     st.rerun()

# --- Display Options ---
st.sidebar.header("Display Options")
unlabeled_filter = st.sidebar.checkbox("Show Only Unlabeled Events", value=True)
limit_filter = st.sidebar.slider("Max Events per Page", min_value=10, max_value=500, value=50, step=10)

# --- Fetch Data ---
# Note: Pagination needs more complex state management if loading more than `limit`
# For now, we just fetch the top `limit` events based on the filter
events_df, total_events = get_events(limit=limit_filter, unlabeled_only=unlabeled_filter)

# --- Event Display Section ---
st.header("Logged Events")
st.write(f"Displaying latest {len(events_df)} of {total_events} events.")

if not events_df.empty:
    # Select columns to display in the main table
    display_columns = ['timestamp', 'event_uid', 'prediction', 'human_label', 'label_timestamp']
    # Filter columns that actually exist in the DataFrame
    display_columns = [col for col in display_columns if col in events_df.columns]

    st.dataframe(events_df[display_columns], use_container_width=True)

    # --- Event Labeling Section ---
    st.header("Event Labeling")

    # Allow selecting an event for labeling (use UID for robustness)
    # Filter DataFrame based on checkbox for selectbox options
    labelable_events_df = events_df if not unlabeled_filter else events_df[events_df['human_label'].isnull()]

    if not labelable_events_df.empty:
        # Create options for the selectbox: "UID - Timestamp"
        event_options = {
            f"{row['event_uid']} ({row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')})": row['event_uid']
            for index, row in labelable_events_df.iterrows()
        }
        selected_event_display = st.selectbox(
            "Select Event to Label:",
            options=event_options.keys(),
            index=0 # Default to the first event
        )

        if selected_event_display:
            selected_event_uid = event_options[selected_event_display]
            # Get the full details for the selected event
            selected_event_details = events_df[events_df['event_uid'] == selected_event_uid].iloc[0]

            st.subheader(f"Details for Event: {selected_event_uid}")
            # Display relevant details (customize as needed)
            st.json(selected_event_details.to_dict(), expanded=False) # Show all data collapsed

            # Labeling options
            label_options = ["Confirmed Attack", "False Positive", "Uncertain"] # Add more specific types if desired
            chosen_label = st.radio(
                "Assign Label:",
                options=label_options,
                key=f"label_{selected_event_uid}" # Unique key per event prevents state issues
            )

            if st.button("Submit Label", key=f"submit_{selected_event_uid}"):
                with st.spinner("Submitting label..."):
                    result = submit_label(selected_event_uid, chosen_label)
                    if result and result.get("status") == "success":
                        st.success(f"Successfully labeled event {selected_event_uid} as '{chosen_label}'.")
                        # Clear cache to force data refresh on next run
                        st.cache_data.clear()
                        st.rerun() # Rerun immediately to show updated table
                    else:
                        st.error("Failed to submit label.")
    else:
        st.info("No events available for labeling based on current filters.")

else:
    st.info("No events found matching the current criteria.")

# --- Placeholder for other sections ---
# st.header("Reward History")
# st.info("Reward history visualization coming soon.")

# st.header("Network Statistics")
# st.info("Network statistics visualization coming soon.")

