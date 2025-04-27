# src/dashboard/dashboard_v2.py
# A rebuilt version focusing on core functionality first.

import streamlit as st
import requests
import pandas as pd
import datetime
import os
import logging

# --- Configuration ---
API_BASE_URL = os.getenv("WIDS_API_URL", "http://127.0.0.1:8000")
EVENTS_ENDPOINT = f"{API_BASE_URL}/events/"
LABEL_ENDPOINT = f"{API_BASE_URL}/events/label"

# Setup logging
# In Streamlit, logging might go to the console where you ran `streamlit run`
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__) # Use a named logger

# --- Session State Initialization ---
# Use session state to store fetched data and avoid re-fetching on every minor interaction
if 'events_data' not in st.session_state:
    st.session_state.events_data = pd.DataFrame()
    st.session_state.total_events = 0
    st.session_state.fetch_error = None
    st.session_state.last_fetch_time = None

# --- Helper Functions ---

def fetch_data_from_api(skip: int = 0, limit: int = 100, unlabeled_only: bool = False):
    """Fetches data and updates session state. Returns True on success, False on error."""
    params = {"skip": skip, "limit": limit, "unlabeled_only": unlabeled_only}
    api_url = EVENTS_ENDPOINT
    log.info(f"Attempting to fetch events from: {api_url} with params: {params}")
    st.session_state.fetch_error = None # Reset error state
    try:
        response = requests.get(api_url, params=params, timeout=15) # Increased timeout
        response.raise_for_status() # Check for HTTP errors
        data = response.json()
        log.info(f"API Response Status: {response.status_code}. Fetched {len(data.get('events', []))} events (total: {data.get('total_count', 0)}).")

        st.session_state.total_events = data.get("total_count", 0)
        if data.get("events"):
            df = pd.DataFrame(data["events"])
            # Safely convert timestamps
            if 'timestamp' in df.columns:
                 df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            if 'label_timestamp' in df.columns:
                 df['label_timestamp'] = pd.to_datetime(df['label_timestamp'], errors='coerce')
            # Keep rows even if timestamp conversion fails for now, handle in display
            st.session_state.events_data = df
        else:
            log.info("API returned no events.")
            st.session_state.events_data = pd.DataFrame() # Store empty DataFrame

        st.session_state.last_fetch_time = datetime.datetime.now()
        return True # Success

    except requests.exceptions.ConnectionError as e:
        error_msg = f"Connection Error: Could not connect to API at {api_url}. Backend running?"
        st.session_state.fetch_error = error_msg
        log.error(f"API connection error: {e}", exc_info=True)
        return False
    except requests.exceptions.Timeout as e:
        error_msg = f"Timeout Error: Request to API timed out ({api_url})."
        st.session_state.fetch_error = error_msg
        log.error(f"API timeout error: {e}", exc_info=True)
        return False
    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else 'N/A'
        response_text = e.response.text if e.response is not None else 'No response'
        error_msg = f"API Request Error: Failed fetching data from {api_url}. Status: {status_code}. Response: '{response_text[:200]}...'" # Limit response length
        st.session_state.fetch_error = error_msg
        log.error(f"API request error: {e}", exc_info=True)
        return False
    except Exception as e:
        error_msg = f"Unexpected error fetching/processing events: {e}"
        st.session_state.fetch_error = error_msg
        log.error(f"Error fetching/processing events: {e}", exc_info=True)
        return False

def submit_label_api(event_uid: str, label: str):
    """Submits a label via API. Returns True on success, False on error."""
    payload = {"event_uid": event_uid, "human_label": label}
    api_url = LABEL_ENDPOINT
    log.info(f"Submitting label for {event_uid} to {api_url}")
    try:
        response = requests.post(api_url, json=payload, timeout=10)
        response.raise_for_status()
        log.info(f"Submitted label '{label}'. Status: {response.status_code}")
        return True
    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else 'N/A'
        response_text = e.response.text if e.response is not None else 'No response'
        st.error(f"API Error Submitting Label: {e}. Status: {status_code}. Response: '{response_text[:200]}...'")
        log.error(f"API error submitting label for {event_uid}: {e}", exc_info=True)
        return False
    except Exception as e:
        st.error(f"Unexpected error submitting label: {e}")
        log.error(f"Error submitting label: {e}", exc_info=True)
        return False

# --- Streamlit App Layout ---

st.set_page_config(page_title="WIDS Dashboard v2", layout="wide")

st.title("📶 WiFi Intrusion Detection Dashboard (v2)")
st.caption(f"API Target: {API_BASE_URL}")

# --- Control Sidebar ---
with st.sidebar:
    st.header("Controls")
    # Manual Refresh Button
    if st.button("🔄 Refresh Data"):
        with st.spinner("Fetching data..."):
            fetch_data_from_api() # Fetch will update session state

    # Display last fetch time
    if st.session_state.last_fetch_time:
        st.caption(f"Data last fetched: {st.session_state.last_fetch_time.strftime('%H:%M:%S')}")
    else:
        st.caption("Data not fetched yet.")

    st.divider()
    st.header("Filters")
    unlabeled_filter = st.checkbox("Show Only Unlabeled Events", value=True, key="unlabeled_chk")
    limit_filter = st.slider("Max Events to Fetch", min_value=10, max_value=500, value=50, step=10, key="limit_sld")

# --- Main Display Area ---

# Trigger initial data fetch if needed (or after settings change)
# We fetch if data is empty OR if filters changed (simple check, could be more sophisticated)
# This logic might need refinement depending on desired refresh behavior vs sidebar interaction
if st.session_state.events_data.empty or 'last_filters' not in st.session_state or \
   st.session_state.last_filters != (limit_filter, unlabeled_filter):
    with st.spinner("Fetching initial data..."):
        fetch_data_from_api(limit=limit_filter, unlabeled_only=unlabeled_filter)
        st.session_state.last_filters = (limit_filter, unlabeled_filter) # Store filters used

# Display any fetch errors prominently
if st.session_state.fetch_error:
    st.error(st.session_state.fetch_error)

# --- Display Events Table ---
st.header("🔎 Logged Events")
st.write(f"Showing {len(st.session_state.events_data)} events (Total matching criteria: {st.session_state.total_events})")

if not st.session_state.events_data.empty:
    display_df = st.session_state.events_data.copy()
    # Format timestamp for display, handle potential NaT values gracefully
    if 'timestamp' in display_df.columns:
        display_df['timestamp_str'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').fillna('Invalid Date')
    if 'label_timestamp' in display_df.columns:
        display_df['label_timestamp_str'] = display_df['label_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')

    display_columns = ['timestamp_str', 'event_uid', 'prediction', 'human_label', 'label_timestamp_str']
    display_columns = [col for col in display_columns if col in display_df.columns] # Ensure columns exist

    st.dataframe(display_df[display_columns], use_container_width=True)

    # --- Event Labeling Section ---
    st.divider()
    st.header("🏷️ Event Labeling")

    # Filter for labelable events based on the checkbox *state*
    labelable_events_df = st.session_state.events_data[st.session_state.events_data['human_label'].isnull()] if unlabeled_filter else st.session_state.events_data

    if not labelable_events_df.empty:
        # Create options for the selectbox
        try:
            event_options = {
                f"{row['event_uid']} ({row['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row['timestamp']) else 'No Timestamp'})": row['event_uid']
                for index, row in labelable_events_df.iterrows()
            }
            event_uids_list = list(event_options.keys())
        except Exception as e:
             st.error(f"Error creating dropdown options: {e}")
             log.error(f"Error formatting event options: {e}", exc_info=True)
             event_options = {}
             event_uids_list = []

        if not event_uids_list:
             st.warning("Could not generate event selection dropdown.")
        else:
            selected_event_display = st.selectbox(
                "Select Event to Label:",
                options=event_uids_list,
                index=0,
                key="event_selector" # Assign a key
            )

            if selected_event_display:
                selected_event_uid = event_options[selected_event_display]

                # Display details using columns for better layout
                st.subheader(f"Details for Event: {selected_event_uid}")
                selected_event_details = st.session_state.events_data[st.session_state.events_data['event_uid'] == selected_event_uid].iloc[0]

                col1, col2 = st.columns(2)
                with col1:
                    st.text(f"Timestamp: {selected_event_details.get('timestamp', 'N/A')}")
                    st.text(f"Prediction: {'Anomaly' if selected_event_details.get('prediction') == 1 else 'Normal' if selected_event_details.get('prediction') == 0 else 'N/A'}")
                with col2:
                    st.text(f"Current Label: {selected_event_details.get('human_label', 'None')}")
                    st.text(f"Label Timestamp: {selected_event_details.get('label_timestamp', 'N/A')}")

                st.text("Features Data:")
                st.json(selected_event_details.get('features_data', {}), expanded=False)

                # Labeling Form
                st.subheader("Assign New Label")
                label_options = ["Confirmed Attack", "False Positive", "Uncertain"] # Add more specific types if desired
                # Use st.radio within a form for better state handling on button click
                with st.form(key=f"label_form_{selected_event_uid}"):
                    chosen_label = st.radio(
                        "Select Label:",
                        options=label_options,
                        key=f"label_radio_{selected_event_uid}" # Unique key
                    )
                    submitted = st.form_submit_button("Submit Label")

                    if submitted:
                        log.info(f"Submit button clicked for {selected_event_uid} with label {chosen_label}")
                        with st.spinner("Submitting label..."):
                            success = submit_label_api(selected_event_uid, chosen_label)
                            if success:
                                st.success(f"Successfully submitted label '{chosen_label}' for event {selected_event_uid}.")
                                # Clear data from session state to force refresh on rerun
                                st.session_state.events_data = pd.DataFrame()
                                st.session_state.last_fetch_time = None
                                st.rerun() # Rerun to fetch updated data
                            else:
                                # Error is displayed by submit_label_api
                                pass
    else:
        st.info("No events available for labeling based on current filters.")

elif st.session_state.fetch_error is None: # Only show if no fetch error occurred
     st.info("No events found matching the current criteria.")


# --- Placeholder for other sections ---
# st.divider()
# st.header("📊 Other Visualizations")
# st.info("Reward history and network statistics visualizations coming soon.")

