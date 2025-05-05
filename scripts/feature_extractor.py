#!/usr/bin/env python3

import scapy.all as scapy
from scapy.layers.dot11 import Dot11, Dot11Beacon, Dot11ProbeReq, Dot11ProbeResp, \
                               Dot11AssoReq, Dot11AssoResp, Dot11Auth, Dot11Deauth, \
                               Dot11Disas, RadioTap, Dot11QoS, LLC, Dot11Elt, \
                               Dot11ReassoReq, Dot11ReassoResp, Dot11ATIM, Dot11Action, \
                               Dot11WEP
from scapy.layers.eap import EAPOL, EAP

from scapy.packet import Packet, Raw
import time
import collections
import statistics
import math
import argparse
import sys
import signal
import numpy as np
import json
import csv
import os
from traceback import print_exc

# --- Async / Concurrency Imports ---
import asyncio
import aiohttp # For async HTTP requests
# Optional: uvloop for potentially faster asyncio event loop
try:
    import uvloop
    uvloop.install()
    print("Using uvloop event loop.")
except ImportError:
    print("uvloop not found, using default asyncio event loop.")
    pass


# --- Configuration ---
DEFAULT_ENDPOINT_URL = 'http://127.0.0.1:5000/events'
DEFAULT_FLOW_TIMEOUT = 10
DEFAULT_CLEANUP_INTERVAL = 10
DEFAULT_PACKET_COUNT = 0
DEFAULT_INTERFACE = "wlan0mon"
IDLE_THRESHOLD = 1.0
BULK_PACKET_THRESHOLD = 5
BULK_SIZE_THRESHOLD = 50
POST_TIMEOUT = 10 # Timeout for aiohttp requests
CSV_BATCH_SIZE = 5
PACKET_QUEUE_SIZE = 1000 # Max packets to buffer before dropping
NUM_WORKERS = 40 or os.cpu_count()  # Number of packet processing workers

# --- Global State ---
flows = {} # Dictionary to store active flow data
flows_lock = asyncio.Lock() # Lock to protect access to 'flows' dictionary
packet_queue = asyncio.Queue(maxsize=PACKET_QUEUE_SIZE) # Queue for packets from sniffer
stop_capture_event = asyncio.Event() # Event to signal shutdown

total_flows = 0
packet_counter = 0 # Note: May not be perfectly accurate without locking increment
total_packets_to_capture = 0
capture_start_time = 0 # Will be set in main
total_bytes_captured = 0 # Note: May not be perfectly accurate without locking increment
webhook_url = DEFAULT_ENDPOINT_URL
monitor_interface_name = DEFAULT_INTERFACE

# CSV related globals
csv_writer = None
csv_file = None
csv_header_written = False
csv_feature_keys = None
processed_flow_buffer = [] # Buffer for batch CSV writing

# --- Helper Functions (Largely unchanged, but called from async context) ---

def get_radiotap_details(packet):
    # (Original function code - no async changes needed)
    details = {
        'dbm_antsignal': None, 'dbm_antnoise': None, 'channel_freq': None,
        'channel_flags': None, 'rate_mbps': None, 'mcs_index': None,
        'bandwidth_mhz': None, 'guard_interval_ns': None, 'ampdu_status': None
    }
    if not packet.haslayer(RadioTap):
        try:
            rt_layer = RadioTap(packet.original)
            if rt_layer: packet = rt_layer
            else: return details
        except Exception: return details

    rt = packet[RadioTap]
    try:
        if hasattr(rt, 'dBm_AntSignal'): details['dbm_antsignal'] = rt.dBm_AntSignal
        if hasattr(rt, 'dBm_AntNoise'): details['dbm_antnoise'] = rt.dBm_AntNoise
        if hasattr(rt, 'ChannelFrequency'): details['channel_freq'] = rt.ChannelFrequency
        if hasattr(rt, 'ChannelFlags'): details['channel_flags'] = rt.ChannelFlags
        if hasattr(rt, 'Rate'): details['rate_mbps'] = rt.Rate * 0.5
        if hasattr(rt, 'MCS'):
            known = getattr(rt.MCS, 'known', 0); flags = getattr(rt.MCS, 'flags', 0)
            details['mcs_index'] = getattr(rt.MCS, 'index', None)
            if known & 1:
                bw_enum = (flags >> 0) & 0b11
                if bw_enum == 0: details['bandwidth_mhz'] = 20
                elif bw_enum == 1: details['bandwidth_mhz'] = 40
            if known & 2:
                gi_enum = (flags >> 2) & 0b1
                details['guard_interval_ns'] = 400 if gi_enum else 800
        if hasattr(rt, 'AMPDU_Status'): details['ampdu_status'] = rt.AMPDU_Status
    except (IndexError, AttributeError, TypeError): pass
    return details

def is_broadcast_multicast(mac_address):
    # (Original function code - no async changes needed)
    if not mac_address or not isinstance(mac_address, str): return False
    mac_lower = mac_address.lower()
    if mac_lower == 'ff:ff:ff:ff:ff:ff': return True
    if mac_lower.startswith("01:00:5e") or mac_lower.startswith("33:33:"): return True
    try:
        first_octet = int(mac_lower.split(':')[0], 16)
        if (first_octet & 1): return True
    except (ValueError, IndexError): pass
    return False

def get_wifi_flow_id(packet: Packet):
    # (Original function code - no async changes needed)
    if not packet.haslayer(Dot11): return None, None, None, None, None
    dot11 = packet.getlayer(Dot11)
    addr1 = getattr(dot11, 'addr1', None); addr2 = getattr(dot11, 'addr2', None)
    addr3 = getattr(dot11, 'addr3', None); addr4 = getattr(dot11, 'addr4', None)
    if addr1 is None or addr2 is None: return None, None, None, None, None
    FC = dot11.FCfield
    sa, da, bssid = None, None, None; direction = None; flow_id = None
    try:
        if dot11.type == 0: sa = addr2; da = addr1; bssid = addr3 # Mgmt
        elif dot11.type == 2: # Data
            if FC.from_ds == 0 and FC.to_ds == 0: sa = addr2; da = addr1; bssid = addr3 # AdHoc/IBSS or Mesh (rare)
            elif FC.from_ds == 1 and FC.to_ds == 0: sa = addr3; da = addr1; bssid = addr2 # AP -> STA
            elif FC.from_ds == 0 and FC.to_ds == 1: sa = addr2; da = addr3; bssid = addr1 # STA -> AP
            elif FC.from_ds == 1 and FC.to_ds == 1: sa = addr4; da = addr3; bssid = None # WDS/Mesh
        elif dot11.type == 1: sa = addr2; da = addr1; bssid = None # Control
        if not sa or not da or not isinstance(sa, str) or not isinstance(da, str): return None, None, None, None, None
        is_bcast_mcast_da = is_broadcast_multicast(da)
        if is_bcast_mcast_da: flow_id = (sa, da); direction = 'uni' # Treat broadcast/multicast as unidirectional from SA
        else:
            # Sort MACs to ensure consistent flow ID regardless of direction
            mac1, mac2 = sorted((sa, da)); flow_id = (mac1, mac2)
            direction = 'fwd' if (sa, da) == (mac1, mac2) else 'bwd'
        bssid = bssid if bssid and isinstance(bssid, str) else "NA"
    except AttributeError: return None, None, None, None, None
    return flow_id, direction, sa, da, bssid

def safe_stat_calc(data, func, default=0.0):
    # (Original function code - no async changes needed)
    if not data: return default
    # Ensure data contains only valid numbers before calculation
    numeric_data = [x for x in data if isinstance(x, (int, float)) and math.isfinite(x)]
    if not numeric_data: return default

    if func in [statistics.stdev, statistics.variance]:
        if len(numeric_data) < 2: return 0.0
        else:
            try: return func(numeric_data)
            except statistics.StatisticsError: return 0.0
    else:
        try: return func(numeric_data)
        except (statistics.StatisticsError, TypeError, ValueError): return default


def parse_ies(packet):
    # (Original function code - no async changes needed)
    ies = {}
    if not packet.haslayer(Dot11Elt): return ies
    current_elt = packet.getlayer(Dot11Elt)
    while current_elt:
        try:
            if current_elt.ID == 0: ies['ssid'] = current_elt.info.decode('utf-8', errors='ignore')
            elif current_elt.ID == 1: ies['supported_rates_mbps'] = [r & 0x7F * 0.5 for r in current_elt.info]
            elif current_elt.ID == 50: ies.setdefault('supported_rates_mbps', []).extend([r & 0x7F * 0.5 for r in current_elt.info]) # Extended rates
            elif current_elt.ID == 48: ies['rsn_info'] = current_elt.info.hex()
            elif current_elt.ID == 221: ies.setdefault('vendor_specific', []).append(current_elt.info.hex())
            elif current_elt.ID == 5: # TIM
                ies['tim_dtim_count'] = current_elt.info[0] if len(current_elt.info) > 0 else None
                ies['tim_dtim_period'] = current_elt.info[1] if len(current_elt.info) > 1 else None
            elif current_elt.ID == 7: ies['country_code'] = current_elt.info[0:2].decode('ascii', errors='ignore')
            elif current_elt.ID == 45: ies['ht_capabilities'] = current_elt.info.hex()
            # Add more IE parsing here if needed (e.g., VHT, HE)
        except Exception: pass # Ignore parsing errors for specific IEs
        # Navigate payload chain to find next Dot11Elt
        if isinstance(current_elt.payload, Dot11Elt):
            current_elt = current_elt.payload
        else:
            # Sometimes the next Elt is nested within a Raw layer or another layer
            current_elt = current_elt.payload.getlayer(Dot11Elt)
    return ies

def parse_eapol(packet):
    # (Original function code - no async changes needed)
    eapol_data = {}
    if not packet.haslayer(EAPOL): return eapol_data
    eapol_layer = packet[EAPOL]
    eapol_data['version'] = eapol_layer.version; eapol_data['type'] = eapol_layer.type
    # print("EAPOL:", eapol_data) # Debug print
    if eapol_layer.type == 3 and hasattr(eapol_layer, 'key_descriptor_type'): # EAPOL-Key
        eapol_data['key_descriptor_type'] = eapol_layer.key_descriptor_type
        key_info = getattr(eapol_layer, 'key_info', 0)
        # Decode Key Information field (based on IEEE 802.11-2016, Figure 12-26)
        eapol_data['key_info_key_desc_ver'] = (key_info >> 0) & 0b111 # Bits 0-2
        eapol_data['key_info_key_type'] = (key_info >> 3) & 0b1      # Bit 3 (1=Pairwise, 0=Group)
        is_pairwise = eapol_data['key_info_key_type'] == 1
        # Bit 4-5 Reserved
        eapol_data['key_info_install'] = (key_info >> 6) & 0b1       # Bit 6
        is_install = eapol_data['key_info_install'] == 1
        eapol_data['key_info_key_ack'] = (key_info >> 7) & 0b1        # Bit 7
        is_ack = eapol_data['key_info_key_ack'] == 1
        eapol_data['key_info_key_mic'] = (key_info >> 8) & 0b1        # Bit 8
        has_mic = eapol_data['key_info_key_mic'] == 1
        eapol_data['key_info_secure'] = (key_info >> 9) & 0b1       # Bit 9
        is_secure = eapol_data['key_info_secure'] == 1
        eapol_data['key_info_error'] = (key_info >> 10) & 0b1      # Bit 10
        eapol_data['key_info_request'] = (key_info >> 11) & 0b1     # Bit 11
        eapol_data['key_info_encrypted'] = (key_info >> 12) & 0b1 # Bit 12
        # Bit 13-15 Reserved
        eapol_data['key_length'] = getattr(eapol_layer, 'key_length', None)
        eapol_data['replay_counter'] = getattr(eapol_layer, 'replay_counter', None)
        eapol_data['key_nonce'] = getattr(eapol_layer, 'key_nonce', b'').hex()
        eapol_data['key_iv'] = getattr(eapol_layer, 'key_iv', b'').hex()
        eapol_data['key_rsc'] = getattr(eapol_layer, 'key_rsc', None)
        eapol_data['key_mic'] = getattr(eapol_layer, 'key_mic', b'').hex()
        eapol_data['key_data_length'] = getattr(eapol_layer, 'key_data_length', None)

        # Determine Handshake Message (simplified based on common patterns)
        if is_pairwise and not has_mic and is_ack and not is_install: eapol_data['handshake_msg'] = 'M1' # ANonce
        elif is_pairwise and has_mic and not is_ack and not is_install: eapol_data['handshake_msg'] = 'M2' # SNonce, MIC
        elif is_pairwise and has_mic and is_ack and is_install and is_secure: eapol_data['handshake_msg'] = 'M3' # GTK, MIC, Install=1, Secure=1
        elif is_pairwise and has_mic and not is_ack and not is_install and is_secure: eapol_data['handshake_msg'] = 'M4' # Ack, MIC, Secure=1
        elif not is_pairwise and has_mic and is_ack and is_secure: eapol_data['handshake_msg'] = 'G1' # Group Key Handshake Msg 1
        elif not is_pairwise and has_mic and not is_ack and is_secure: eapol_data['handshake_msg'] = 'G2' # Group Key Handshake Msg 2
        else: eapol_data['handshake_msg'] = 'Unknown'
    return eapol_data

def initialize_wifi_flow_data(packet_time, bssid="", flow_type='bi'):
    # (Original function code - no async changes needed)
    return {
        'start_time': packet_time, 'last_seen': packet_time, 'pkt_count': 0,
        'fwd_pkt_count': 0, 'bwd_pkt_count': 0, 'fwd_bytes_total': 0, 'bwd_bytes_total': 0,
        'all_pkt_times': [], 'fwd_pkt_times': [], 'bwd_pkt_times': [],
        'fwd_frame_lengths': [], 'bwd_frame_lengths': [], 'bssid': bssid, 'flow_type': flow_type,
        'fwd_signal_dbm': [], 'bwd_signal_dbm': [], 'fwd_noise_dbm': [], 'bwd_noise_dbm': [],
        'fwd_data_rate_mbps': [], 'bwd_data_rate_mbps': [], 'fwd_mcs_indices': [], 'bwd_mcs_indices': [],
        'fwd_bandwidths_mhz': collections.Counter(), 'bwd_bandwidths_mhz': collections.Counter(),
        'fwd_guard_intervals_ns': collections.Counter(), 'bwd_guard_intervals_ns': collections.Counter(),
        'channels': set(), 'frame_types': collections.Counter(), 'mgmt_subtypes': collections.Counter(),
        'ctrl_subtypes': collections.Counter(), 'data_subtypes': collections.Counter(),
        'retry_count': 0, 'protected_count': 0, 'ssids': set(),
        'ies': {'rsn_info': None, 'country_code': None, 'supported_rates_mbps': [], 'vendor_specific_count': 0},
        'eapol_msgs': collections.Counter(), 'eapol_key_nonces': {'fwd': set(), 'bwd': set()},
        'eapol_mic_present_count': 0, 'eapol_install_flag_count': 0,
        # Sequence tracking flags
        'seq_auth_req_seen': 0, 'seq_auth_resp_seen': 0, 'seq_assoc_req_seen': 0, 'seq_assoc_resp_seen': 0,
        'seq_eapol_m1_seen': 0, 'seq_eapol_m2_seen': 0, 'seq_eapol_m3_seen': 0, 'seq_eapol_m4_seen': 0,
        # Active/Idle tracking
        'last_active_time': packet_time, 'current_active_start': packet_time, 'active_periods': [],
        'last_idle_time': packet_time, 'current_idle_start': packet_time, 'idle_periods': [], 'is_idle': False,
        # Bulk transfer tracking
        'bulk_last_dir': None, 'bulk_pkt_count_in_phase': 0, 'bulk_start_time': 0,
        'bulk_fwd_count': 0, 'bulk_fwd_pkt_count': 0, 'bulk_fwd_bytes': 0, 'bulk_fwd_duration': 0,
        'bulk_bwd_count': 0, 'bulk_bwd_pkt_count': 0, 'bulk_bwd_bytes': 0, 'bulk_bwd_duration': 0,
    }

def update_active_idle_state(flow_data, packet_time, idle_threshold):
    # (Original function code - no async changes needed)
    if flow_data['pkt_count'] <= 1: # Initialize on first packet seen after creation
        flow_data['current_active_start'] = packet_time
        flow_data['last_active_time'] = packet_time
        flow_data['last_idle_time'] = packet_time # Avoid large initial idle period
        flow_data['is_idle'] = False
        return

    last_time = flow_data['last_seen']
    iat = packet_time - last_time # Inter-arrival time

    if iat > idle_threshold: # Transition to Idle or continue Idle
        if not flow_data['is_idle']: # Transition Active -> Idle
            active_duration = last_time - flow_data['current_active_start']
            if active_duration > 0: flow_data['active_periods'].append(active_duration)
            flow_data['current_idle_start'] = last_time # Start idle period at last packet time
            flow_data['is_idle'] = True
        # If already idle, the idle period continues implicitly
    else: # Continue Active or transition to Active
        if flow_data['is_idle']: # Transition Idle -> Active
            idle_duration = packet_time - flow_data['current_idle_start']
            if idle_duration > 0: flow_data['idle_periods'].append(idle_duration)
            flow_data['current_active_start'] = packet_time # Start active period now
            flow_data['is_idle'] = False
        # If already active, the active period continues implicitly

    # Update last times regardless of state
    if not flow_data['is_idle']: flow_data['last_active_time'] = packet_time
    flow_data['last_idle_time'] = packet_time # Tracks last packet time overall

def end_active_idle_periods(flow_data, end_time):
    # (Original function code - no async changes needed)
     # Ensure at least one period is recorded even for short flows
     if flow_data['pkt_count'] > 0:
         if flow_data['is_idle']:
             # End the last idle period
             idle_duration = end_time - flow_data['current_idle_start']
             if idle_duration > 0: flow_data['idle_periods'].append(idle_duration)
             # If the flow started and ended idle without becoming active
             if not flow_data['active_periods'] and flow_data['pkt_count'] <=1 :
                 flow_data['active_periods'].append(0.0) # Record zero active time
         else:
             # End the last active period
             active_duration = end_time - flow_data['current_active_start']
             if active_duration > 0: flow_data['active_periods'].append(active_duration)
             # If the flow started and ended active without becoming idle
             if not flow_data['idle_periods'] and flow_data['pkt_count'] <=1 :
                  flow_data['idle_periods'].append(0.0) # Record zero idle time

     # Handle flows with only one packet (zero duration active/idle)
     elif flow_data['pkt_count'] == 1:
         flow_data['active_periods'].append(0.0)
         flow_data['idle_periods'].append(0.0)


def update_bulk_state(flow_data, direction, packet_time, frame_len, bulk_pkt_thresh, bulk_size_thresh):
    # (Original function code - no async changes needed)
    # Use 'fwd' for 'uni' direction in bulk analysis
    effective_direction = 'fwd' if direction == 'uni' else direction

    is_bulk_candidate = frame_len > bulk_size_thresh

    if not is_bulk_candidate:
        # Packet doesn't meet size threshold, end current bulk phase if it was active
        if flow_data['bulk_pkt_count_in_phase'] >= bulk_pkt_thresh:
            duration = flow_data['last_seen'] - flow_data['bulk_start_time']
            if flow_data['bulk_last_dir'] == 'fwd': flow_data['bulk_fwd_duration'] += duration
            elif flow_data['bulk_last_dir'] == 'bwd': flow_data['bulk_bwd_duration'] += duration
        # Reset phase tracking
        flow_data['bulk_pkt_count_in_phase'] = 0
        flow_data['bulk_last_dir'] = None
        return

    # Packet meets size threshold
    if effective_direction == flow_data['bulk_last_dir']:
        # Continue current bulk phase
        flow_data['bulk_pkt_count_in_phase'] += 1
        if effective_direction == 'fwd':
            flow_data['bulk_fwd_bytes'] += frame_len
            flow_data['bulk_fwd_pkt_count'] += 1
        else: # bwd
            flow_data['bulk_bwd_bytes'] += frame_len
            flow_data['bulk_bwd_pkt_count'] += 1
    else:
        # Direction changed, end previous phase (if it met threshold)
        if flow_data['bulk_pkt_count_in_phase'] >= bulk_pkt_thresh:
            duration = flow_data['last_seen'] - flow_data['bulk_start_time']
            if flow_data['bulk_last_dir'] == 'fwd': flow_data['bulk_fwd_duration'] += duration
            elif flow_data['bulk_last_dir'] == 'bwd': flow_data['bulk_bwd_duration'] += duration

        # Start new bulk phase
        flow_data['bulk_last_dir'] = effective_direction
        flow_data['bulk_pkt_count_in_phase'] = 1
        flow_data['bulk_start_time'] = packet_time
        if effective_direction == 'fwd':
            flow_data['bulk_fwd_count'] += 1 # Increment count of fwd bulk phases
            flow_data['bulk_fwd_bytes'] += frame_len
            flow_data['bulk_fwd_pkt_count'] += 1
        else: # bwd
            flow_data['bulk_bwd_count'] += 1 # Increment count of bwd bulk phases
            flow_data['bulk_bwd_bytes'] += frame_len
            flow_data['bulk_bwd_pkt_count'] += 1

def end_bulk_phase(flow_data):
    # (Original function code - no async changes needed)
     # Check if the last phase met the threshold and add its duration
     if flow_data['bulk_pkt_count_in_phase'] >= BULK_PACKET_THRESHOLD:
          duration = flow_data['last_seen'] - flow_data['bulk_start_time']
          if flow_data['bulk_last_dir'] == 'fwd': flow_data['bulk_fwd_duration'] += duration
          elif flow_data['bulk_last_dir'] == 'bwd': flow_data['bulk_bwd_duration'] += duration
     # Reset phase tracking for safety, although flow is ending
     flow_data['bulk_pkt_count_in_phase'] = 0
     flow_data['bulk_last_dir'] = None

def sanitize_for_json(data):
    # (Original function code - no async changes needed)
    if isinstance(data, dict):
        # Handle Counter explicitly if needed, otherwise treat as dict
        if isinstance(data, collections.Counter):
            return {str(k): sanitize_for_json(v) for k, v in data.items()}
        return {str(k): sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, (list, set, tuple)):
        return [sanitize_for_json(item) for item in data]
    elif isinstance(data, float):
        # Replace NaN/Infinity with None (or 0 or specific string if preferred)
        return None if math.isinf(data) or math.isnan(data) else data
    elif isinstance(data, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(data) # Convert numpy ints to standard Python int
    elif isinstance(data, (np.float16, np.float32, np.float64)):
         py_float = float(data) # Convert numpy floats to standard Python float
         return None if math.isinf(py_float) or math.isnan(py_float) else py_float
    elif isinstance(data, np.bool_):
        return bool(data) # Convert numpy bool to standard Python bool
    elif isinstance(data, np.ndarray):
        return sanitize_for_json(data.tolist()) # Convert numpy arrays to lists
    elif isinstance(data, bytes):
        return data.decode('utf-8', errors='ignore') # Decode bytes to string
    else:
        # Assume other types are JSON serializable (int, str, bool, None)
        return data

def calculate_wifi_flow_features(flow_id, flow_data):
    # --- Calculate Features (No async changes needed in calculation logic) ---
    features = collections.OrderedDict()
    try:
        # Timestamp
        features['timestamp'] = flow_data['start_time']
        # Flow Identifiers
        features['flow_type'] = flow_data['flow_type']
        features['mac1'] = flow_id[0]
        features['mac2'] = flow_id[1]
        features['bssid'] = flow_data['bssid']
        ssids = ";".join(sorted(list(flow_data['ssids'])))
        features['ssids'] = ssids if ssids != "" else "NA"
        features['channels'] = ";".join(sorted([str(c) for c in flow_data['channels']]))
        # Basic Counts & Duration
        pkt_count = flow_data['pkt_count']; fwd_pkt_count = flow_data['fwd_pkt_count']; bwd_pkt_count = flow_data['bwd_pkt_count']
        features['tot_pkts'] = pkt_count; features['tot_fwd_pkts'] = fwd_pkt_count; features['tot_bwd_pkts'] = bwd_pkt_count
        features['totlen_fwd_bytes'] = flow_data['fwd_bytes_total']; features['totlen_bwd_bytes'] = flow_data['bwd_bytes_total'] # Use pre-summed values
        features['totlen_bytes'] = features['totlen_fwd_bytes'] + features['totlen_bwd_bytes']
        flow_start_time = flow_data['start_time']; flow_end_time = flow_data['last_seen']
        flow_duration_sec = max(0.0, flow_end_time - flow_start_time) # Ensure non-negative
        features['flow_duration'] = int(flow_duration_sec * 1_000_000) # Microseconds
        #
        # # --- Inter-Arrival Time (IAT) Stats ---
        # all_times = sorted(flow_data['all_pkt_times'])
        # all_iats = [all_times[i] - all_times[i-1] for i in range(1, len(all_times))]
        # features['flow_iat_mean'] = safe_stat_calc(all_iats, statistics.mean, 0.0) * 1e6 # Microseconds
        # features['flow_iat_std'] = safe_stat_calc(all_iats, statistics.stdev, 0.0) * 1e6
        # features['flow_iat_max'] = safe_stat_calc(all_iats, max, 0.0) * 1e6
        # features['flow_iat_min'] = safe_stat_calc(all_iats, min, 0.0) * 1e6 if all_iats else 0.0

        fwd_times = sorted(flow_data['fwd_pkt_times'])
        fwd_iats = [fwd_times[i] - fwd_times[i-1] for i in range(1, len(fwd_times))]
        features['fwd_iat_mean'] = safe_stat_calc(fwd_iats, statistics.mean, 0.0) * 1e6
        features['fwd_iat_std'] = safe_stat_calc(fwd_iats, statistics.stdev, 0.0) * 1e6
        features['fwd_iat_max'] = safe_stat_calc(fwd_iats, max, 0.0) * 1e6
        features['fwd_iat_min'] = safe_stat_calc(fwd_iats, min, 0.0) * 1e6 if fwd_iats else 0.0

        bwd_times = sorted(flow_data['bwd_pkt_times'])
        bwd_iats = [bwd_times[i] - bwd_times[i-1] for i in range(1, len(bwd_times))]
        features['bwd_iat_mean'] = safe_stat_calc(bwd_iats, statistics.mean, 0.0) * 1e6
        features['bwd_iat_std'] = safe_stat_calc(bwd_iats, statistics.stdev, 0.0) * 1e6
        features['bwd_iat_max'] = safe_stat_calc(bwd_iats, max, 0.0) * 1e6
        features['bwd_iat_min'] = safe_stat_calc(bwd_iats, min, 0.0) * 1e6 if bwd_iats else 0.0

        # --- Frame Length Stats ---
        fwd_lengths = flow_data['fwd_frame_lengths']; bwd_lengths = flow_data['bwd_frame_lengths']; all_lengths = fwd_lengths + bwd_lengths
        features['fwd_pkt_len_max'] = safe_stat_calc(fwd_lengths, max, 0); features['fwd_pkt_len_min'] = safe_stat_calc(fwd_lengths, min, 0) if fwd_lengths else 0
        features['fwd_pkt_len_mean'] = safe_stat_calc(fwd_lengths, statistics.mean, 0.0); features['fwd_pkt_len_std'] = safe_stat_calc(fwd_lengths, statistics.stdev, 0.0)
        features['bwd_pkt_len_max'] = safe_stat_calc(bwd_lengths, max, 0); features['bwd_pkt_len_min'] = safe_stat_calc(bwd_lengths, min, 0) if bwd_lengths else 0
        features['bwd_pkt_len_mean'] = safe_stat_calc(bwd_lengths, statistics.mean, 0.0); features['bwd_pkt_len_std'] = safe_stat_calc(bwd_lengths, statistics.stdev, 0.0)
        # features['pkt_len_max'] = safe_stat_calc(all_lengths, max, 0)
        # features['pkt_len_min'] = safe_stat_calc(all_lengths, min, 0) if all_lengths else 0
        # features['pkt_len_mean'] = safe_stat_calc(all_lengths, statistics.mean, 0.0)
        # features['pkt_len_std'] = safe_stat_calc(all_lengths, statistics.stdev, 0.0)
        # features['pkt_len_var'] = safe_stat_calc(all_lengths, statistics.variance, 0.0)
        #
        # # --- Flow Bytes/Packets per Second ---
        # flow_dur_sec = flow_duration_sec if flow_duration_sec > 0 else 1e-9 # Avoid division by zero
        # features['fwd_pkts_s'] = fwd_pkt_count / flow_dur_sec
        # features['bwd_pkts_s'] = bwd_pkt_count / flow_dur_sec
        # features['tot_pkts_s'] = pkt_count / flow_dur_sec
        # features['fwd_bytes_s'] = features['totlen_fwd_bytes'] / flow_dur_sec
        # features['bwd_bytes_s'] = features['totlen_bwd_bytes'] / flow_dur_sec
        # features['tot_bytes_s'] = features['totlen_bytes'] / flow_dur_sec

        # --- RadioTap Stats ---
        features['fwd_signal_dbm_mean'] = safe_stat_calc(flow_data['fwd_signal_dbm'], statistics.mean, -100.0); features['fwd_signal_dbm_std'] = safe_stat_calc(flow_data['fwd_signal_dbm'], statistics.stdev, 0.0)
        features['bwd_signal_dbm_mean'] = safe_stat_calc(flow_data['bwd_signal_dbm'], statistics.mean, -100.0); features['bwd_signal_dbm_std'] = safe_stat_calc(flow_data['bwd_signal_dbm'], statistics.stdev, 0.0)
        # features['fwd_noise_dbm_mean'] = safe_stat_calc(flow_data['fwd_noise_dbm'], statistics.mean, -100.0); features['bwd_noise_dbm_mean'] = safe_stat_calc(flow_data['bwd_noise_dbm'], statistics.mean, -100.0) # Noise std might be less useful
        features['fwd_data_rate_mean'] = safe_stat_calc(flow_data['fwd_data_rate_mbps'], statistics.mean, 0.0); features['bwd_data_rate_mean'] = safe_stat_calc(flow_data['bwd_data_rate_mbps'], statistics.mean, 0.0)
        features['fwd_mcs_mean'] = safe_stat_calc(flow_data['fwd_mcs_indices'], statistics.mean, 0.0); features['bwd_mcs_mean'] = safe_stat_calc(flow_data['bwd_mcs_indices'], statistics.mean, 0.0)
        # Add counts for specific BW/GI values if they exist
        for bw, count in flow_data['fwd_bandwidths_mhz'].items(): features[f'fwd_bw_{bw}mhz_count'] = count
        for bw, count in flow_data['bwd_bandwidths_mhz'].items(): features[f'bwd_bw_{bw}mhz_count'] = count
        for gi, count in flow_data['fwd_guard_intervals_ns'].items(): features[f'fwd_gi_{gi}ns_count'] = count
        for gi, count in flow_data['bwd_guard_intervals_ns'].items(): features[f'bwd_gi_{gi}ns_count'] = count

        # --- Dot11 Stats ---
        features['retry_pkts'] = flow_data['retry_count']; features['protected_pkts'] = flow_data['protected_count']
        features['mgmt_frame_count'] = flow_data['frame_types'].get(0, 0); features['ctrl_frame_count'] = flow_data['frame_types'].get(1, 0); features['data_frame_count'] = flow_data['frame_types'].get(2, 0)
        # Management Subtypes
        features['auth_req_count'] = flow_data['mgmt_subtypes'].get(11, 0); features['auth_resp_count'] = flow_data['mgmt_subtypes'].get(12, 0)
        features['assoc_req_count'] = flow_data['mgmt_subtypes'].get(0, 0); features['assoc_resp_count'] = flow_data['mgmt_subtypes'].get(1, 0)
        features['reassoc_req_count'] = flow_data['mgmt_subtypes'].get(2, 0); features['reassoc_resp_count'] = flow_data['mgmt_subtypes'].get(3, 0)
        features['probe_req_count'] = flow_data['mgmt_subtypes'].get(4, 0); features['probe_resp_count'] = flow_data['mgmt_subtypes'].get(5, 0)
        features['beacon_count'] = flow_data['mgmt_subtypes'].get(8, 0); features['atim_count'] = flow_data['mgmt_subtypes'].get(9, 0)
        features['disassoc_count'] = flow_data['mgmt_subtypes'].get(10, 0); features['deauth_count'] = flow_data['mgmt_subtypes'].get(12, 0) # Corrected subtype for Deauth
        features['action_count'] = flow_data['mgmt_subtypes'].get(13, 0)
        # Control Subtypes
        features['blockackreq_count'] = flow_data['ctrl_subtypes'].get(8, 0); features['blockack_count'] = flow_data['ctrl_subtypes'].get(9, 0)
        features['pspoll_count'] = flow_data['ctrl_subtypes'].get(10, 0); features['rts_count'] = flow_data['ctrl_subtypes'].get(11, 0)
        features['cts_count'] = flow_data['ctrl_subtypes'].get(12, 0); features['ack_count'] = flow_data['ctrl_subtypes'].get(13, 0)
        # Data Subtypes
        features['data_qos_count'] = flow_data['data_subtypes'].get(8, 0) # QoS Data subtype is 8

        # --- IE Information ---
        features['ie_vendor_specific_count'] = flow_data['ies']['vendor_specific_count']
        features['ie_rsn_present'] = 1 if flow_data['ies']['rsn_info'] else 0
        features['ie_country_code_present'] = 1 if flow_data['ies']['country_code'] else 0
        features['ie_supported_rates_count'] = len(flow_data['ies']['supported_rates_mbps'])

        # --- EAPOL Summary ---
        features['eapol_m1_count'] = flow_data['eapol_msgs'].get('M1', 0); features['eapol_m2_count'] = flow_data['eapol_msgs'].get('M2', 0)
        features['eapol_m3_count'] = flow_data['eapol_msgs'].get('M3', 0); features['eapol_m4_count'] = flow_data['eapol_msgs'].get('M4', 0)
        features['eapol_g1_count'] = flow_data['eapol_msgs'].get('G1', 0); features['eapol_g2_count'] = flow_data['eapol_msgs'].get('G2', 0)
        features['eapol_mic_present_count'] = flow_data['eapol_mic_present_count']; features['eapol_install_flag_count'] = flow_data['eapol_install_flag_count']
        features['eapol_fwd_nonces_count'] = len(flow_data['eapol_key_nonces']['fwd']); features['eapol_bwd_nonces_count'] = len(flow_data['eapol_key_nonces']['bwd'])
        features['eapol_total_msgs'] = sum(flow_data['eapol_msgs'].values())
        #
        # # --- Active/Idle Time Stats ---
        # features['active_time_mean'] = safe_stat_calc(flow_data['active_periods'], statistics.mean, 0.0) * 1e6 # Microseconds
        # features['active_time_std'] = safe_stat_calc(flow_data['active_periods'], statistics.stdev, 0.0) * 1e6
        # features['active_time_max'] = safe_stat_calc(flow_data['active_periods'], max, 0.0) * 1e6
        # features['active_time_min'] = safe_stat_calc(flow_data['active_periods'], min, 0.0) * 1e6 if flow_data['active_periods'] else 0.0
        # features['active_time_total'] = sum(flow_data['active_periods']) * 1e6
        #
        # features['idle_time_mean'] = safe_stat_calc(flow_data['idle_periods'], statistics.mean, 0.0) * 1e6
        # features['idle_time_std'] = safe_stat_calc(flow_data['idle_periods'], statistics.stdev, 0.0) * 1e6
        # features['idle_time_max'] = safe_stat_calc(flow_data['idle_periods'], max, 0.0) * 1e6
        # features['idle_time_min'] = safe_stat_calc(flow_data['idle_periods'], min, 0.0) * 1e6 if flow_data['idle_periods'] else 0.0
        # features['idle_time_total'] = sum(flow_data['idle_periods']) * 1e6
        #
        # # --- Bulk Transfer Stats ---
        # features['fwd_bulk_count'] = flow_data['bulk_fwd_count']
        # features['fwd_bulk_pkt_count'] = flow_data['bulk_fwd_pkt_count']
        # features['fwd_bulk_byte_count'] = flow_data['bulk_fwd_bytes']
        # features['fwd_bulk_duration'] = flow_data['bulk_fwd_duration'] * 1e6 # Microseconds
        # features['fwd_bulk_rate'] = (flow_data['bulk_fwd_bytes'] * 8 / flow_data['bulk_fwd_duration']) if flow_data['bulk_fwd_duration'] > 0 else 0.0 # bps
        #
        # features['bwd_bulk_count'] = flow_data['bulk_bwd_count']
        # features['bwd_bulk_pkt_count'] = flow_data['bulk_bwd_pkt_count']
        # features['bwd_bulk_byte_count'] = flow_data['bulk_bwd_bytes']
        # features['bwd_bulk_duration'] = flow_data['bulk_bwd_duration'] * 1e6 # Microseconds
        # features['bwd_bulk_rate'] = (flow_data['bulk_bwd_bytes'] * 8 / flow_data['bulk_bwd_duration']) if flow_data['bulk_bwd_duration'] > 0 else 0.0 # bps
        #
        # # --- Sequence Flags ---
        # features['seq_auth_req'] = flow_data['seq_auth_req_seen']
        # features['seq_auth_resp'] = flow_data['seq_auth_resp_seen']
        # features['seq_assoc_req'] = flow_data['seq_assoc_req_seen']
        # features['seq_assoc_resp'] = flow_data['seq_assoc_resp_seen']
        # features['seq_eapol_m1'] = flow_data['seq_eapol_m1_seen']
        # features['seq_eapol_m2'] = flow_data['seq_eapol_m2_seen']
        # features['seq_eapol_m3'] = flow_data['seq_eapol_m3_seen']
        # features['seq_eapol_m4'] = flow_data['seq_eapol_m4_seen']

    except Exception as e:
        print(f"\nError calculating features for flow {flow_id}: {e}", file=sys.stderr)
        print_exc()
        return None # Return None if feature calculation fails

    return features


# --- Asynchronous Functions ---

async def post_features_to_endpoint(features_dict, url, session):
    """Sends features dictionary to the specified webhook URL asynchronously."""
    if not url or not session:
        # print("Webhook URL or session not provided, skipping POST.") # Debug
        return
    try:
        sanitized_features = sanitize_for_json(features_dict)
        headers = {'Content-Type': 'application/json'}
        # Use the passed aiohttp session
        # Set ssl=False to ignore SSL certificate verification if needed (like for localhost)
        async with session.post(url, json=sanitized_features, headers=headers, ssl=False, timeout=POST_TIMEOUT) as response:
             response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
             # print(f"Successfully posted flow {features_dict.get('mac1', '')}-{features_dict.get('mac2', '')} to {url}") # Debug
    except asyncio.TimeoutError:
        print(f"\nError: Timeout posting data to {url}", file=sys.stderr)
    except aiohttp.ClientResponseError as e:
        # Log detailed error including status and message
        print(f"\nError: HTTP Error {e.status} posting data to {url}: {e.message}", file=sys.stderr)
        # Optionally log response body if available: print(await e.response.text())
    except aiohttp.ClientConnectionError as e:
        print(f"\nError: Network connection error posting data to {url}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"\nUnexpected error during POST to {url}: {e}", file=sys.stderr)
        print_exc() # Print traceback for unexpected errors


# --- CSV Writing (Synchronous Batching) ---
# Note: This part remains synchronous for simplicity.
# For very high throughput, consider aiofiles or a dedicated writer thread.

def write_features_to_csv(list_of_features, writer):
    """Writes a batch of feature dictionaries to the CSV file."""
    global csv_header_written, csv_feature_keys
    if not writer or not list_of_features: return

    try:
        # Write header only if it hasn't been written yet and we have features
        if not csv_header_written:
            if list_of_features:
                # Use the keys from the first feature dict as header
                # Ensure consistent order using the OrderedDict from calculation
                csv_feature_keys = list(list_of_features[0].keys())
                writer.writerow(csv_feature_keys)
                csv_header_written = True
            else:
                return # No features to write header or data from

        # Write feature rows
        for features_dict in list_of_features:
             if csv_feature_keys: # Ensure header keys are available
                 row_values = []
                 for key in csv_feature_keys:
                     value = features_dict.get(key, '') # Get value or default to empty string
                     # Convert complex types to string representations for CSV
                     if isinstance(value, (list, set, tuple)):
                         # Sort for consistent output
                         value = ";".join(map(str, sorted(list(value))))
                     elif isinstance(value, collections.Counter):
                         # Format as key:value pairs, sorted by key
                         value = ";".join([f"{k}:{v}" for k, v in sorted(value.items())])
                     elif isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                          value = '' # Represent NaN/Inf as empty string in CSV
                     row_values.append(value)
                 writer.writerow(row_values)

    except Exception as e:
        print(f"\nError writing batch of features to CSV: {e}", file=sys.stderr)
        print_exc()

def flush_buffer_to_csv():
    """Writes the content of the processed flow buffer to the CSV file."""
    global processed_flow_buffer, csv_writer
    if csv_writer and processed_flow_buffer:
        # print(f"Flushing {len(processed_flow_buffer)} flows to CSV...") # Debug
        write_features_to_csv(processed_flow_buffer, csv_writer)
        processed_flow_buffer.clear() # Clear buffer after writing


async def process_and_output_flow(flow_id, flow_data, session, is_final_cleanup=False):
    """Calculates features, sends to webhook, and adds to CSV buffer."""
    global webhook_url, processed_flow_buffer, total_flows, csv_writer # Need access

    try:
        # Increment flow counter (relatively safe without lock if only done here)
        total_flows += 1

        # Finalize calculations that depend on the end time
        last_time = flow_data['last_seen']
        end_active_idle_periods(flow_data, last_time)
        end_bulk_phase(flow_data)

        # Calculate all features
        features = calculate_wifi_flow_features(flow_id, flow_data)

        if features:
            # Asynchronously post features to the endpoint
            await post_features_to_endpoint(features, webhook_url, session)

            # Add features to the CSV buffer (synchronous operation)
            # No lock needed here as buffer is likely managed per task context
            # or sequentially during cleanup/finalization.
            processed_flow_buffer.append(features)

            # Check if buffer needs flushing (synchronous write)
            if csv_writer and len(processed_flow_buffer) >= CSV_BATCH_SIZE:
                flush_buffer_to_csv() # This call is synchronous

    except Exception as e:
         print(f"\nError processing {'final ' if is_final_cleanup else ''}flow {flow_id}: {e}", file=sys.stderr)
         print_exc()


async def cleanup_timed_out_flows_async(flow_timeout, cleanup_interval, session):
    """Periodically checks for timed-out flows and processes them."""
    global flows, flows_lock, stop_capture_event # Use asyncio event and lock

    print("Flow cleanup task started.")
    while not stop_capture_event.is_set():
        start_cleanup_time = time.time()
        try:
            current_time = time.time()
            timed_out_flow_ids = []

            # --- Safely identify timed-out flows ---
            async with flows_lock:
                # Iterate over a copy of keys to avoid issues if dict changes during iteration (though lock prevents it here)
                flow_ids_snapshot = list(flows.keys())
                for flow_id in flow_ids_snapshot:
                    # Check if flow still exists after acquiring lock (should always be true here)
                    if flow_id in flows:
                        flow_data = flows[flow_id]
                        if current_time - flow_data['last_seen'] > flow_timeout:
                            timed_out_flow_ids.append(flow_id)

            # --- Safely remove and store data for timed-out flows ---
            flows_to_process = {}
            if timed_out_flow_ids:
                # print(f"Found {len(timed_out_flow_ids)} timed-out flows.") # Debug
                async with flows_lock:
                    for flow_id in timed_out_flow_ids:
                        if flow_id in flows: # Check again before popping, just in case
                            flows_to_process[flow_id] = flows.pop(flow_id)
                        # else: print(f"Flow {flow_id} already removed before popping in cleanup.") # Debug

            # --- Process the popped flows outside the main lock ---
            if flows_to_process:
                # print(f"Processing {len(flows_to_process)} timed-out flows...") # Debug
                tasks = []
                for flow_id, flow_data in flows_to_process.items():
                    # Create a task for each flow to process them concurrently
                    # Pass the session for HTTP requests
                    tasks.append(asyncio.create_task(process_and_output_flow(flow_id, flow_data, session)))
                # Wait for all processing tasks for this batch to complete
                if tasks:
                    await asyncio.gather(*tasks)
                # print("Finished processing batch of timed-out flows.") # Debug


        except asyncio.CancelledError:
             print("Flow cleanup task cancelled.")
             break # Exit loop if cancelled
        except Exception as e:
            print(f"\nError during async flow cleanup loop: {e}", file=sys.stderr)
            print_exc()

        # --- Wait for the next interval ---
        elapsed = time.time() - start_cleanup_time
        sleep_time = max(0.1, cleanup_interval - elapsed) # Ensure positive sleep time
        try:
            # Use asyncio.sleep, which allows other tasks to run
            await asyncio.sleep(sleep_time)
        except asyncio.CancelledError:
             print("Flow cleanup task sleep cancelled.")
             break # Exit loop if cancelled
    print("Flow cleanup task finished.")


# --- Packet Processing Worker ---

async def process_packet_async(packet: Packet):
    """Core logic to process a single packet and update flow state."""
    global flows, flows_lock, packet_counter, total_packets_to_capture, stop_capture_event
    global capture_start_time, total_bytes_captured, IDLE_THRESHOLD

    # Avoid processing if shutdown is signaled
    if stop_capture_event.is_set():
        return

    try:
        packet_time = float(packet.time)
        # Use packet.original if available (often the raw bytes), otherwise len(packet)
        packet_len = len(packet.original) if hasattr(packet, 'original') else len(packet)

        # --- Extract Flow ID ---
        flow_id, direction, sa, da, bssid = get_wifi_flow_id(packet)
        if flow_id is None or direction is None:
             # print(f"Could not determine flow ID for packet: {packet.summary()}") # Debug
             return # Ignore packets that don't form a valid flow ID

        # --- Update Global Counters (Potential minor race condition without lock) ---
        # For high accuracy, these would need locking or atomic operations,
        # but for general stats, this might be acceptable.
        packet_counter += 1
        total_bytes_captured += packet_len

        # --- Print Progress Periodically ---
        # This check is done per packet, might be slightly inaccurate due to concurrency
        if packet_counter % 100 == 0: # Print every 100 packets
            current_time = time.time()
            elapsed_time = current_time - capture_start_time if capture_start_time else 1.0
            pps = packet_counter / elapsed_time if elapsed_time > 0 else 0
            bps = total_bytes_captured * 8 / elapsed_time if elapsed_time > 0 else 0
            rate_unit = "Mbps" if bps >= 1_000_000 else "Kbps" if bps >= 1000 else "bps"
            rate_val = bps / 1_000_000 if rate_unit == "Mbps" else bps / 1000 if rate_unit == "Kbps" else bps
            # Get current flow count (needs lock for accuracy)
            async with flows_lock:
                active_flows = len(flows)
            print(f"Processed: ~{packet_counter} pkts ({active_flows} active flows) [{pps:.1f} pps | {rate_val:.2f} {rate_unit}]", end='\r')

        # --- Check Packet Capture Limit ---
        if total_packets_to_capture > 0 and packet_counter >= total_packets_to_capture:
            if not stop_capture_event.is_set():
                 print(f"\nReached packet capture limit ({total_packets_to_capture}). Signaling stop...")
                 stop_capture_event.set() # Signal all tasks to stop
            return # Stop processing this packet

        # --- Update Flow State (Requires Lock) ---
        flow_data = None
        async with flows_lock:
            if flow_id not in flows:
                # Initialize new flow
                flow_type = 'uni' if direction == 'uni' else 'bi'
                flows[flow_id] = initialize_wifi_flow_data(packet_time, bssid, flow_type)
                # print(f"New flow created: {flow_id}") # Debug

            # Get flow data - IMPORTANT: do this inside the lock
            # as cleanup task might remove it between check and access
            if flow_id in flows:
                flow_data = flows[flow_id]

                # Update BSSID if it wasn't set initially
                if not flow_data['bssid'] or flow_data['bssid'] == "NA":
                     if bssid and bssid != "NA": flow_data['bssid'] = bssid

                # Update active/idle state (uses last_seen, which is updated later)
                update_active_idle_state(flow_data, packet_time, IDLE_THRESHOLD)

                # Increment packet counts and add times/lengths
                flow_data['pkt_count'] += 1
                flow_data['all_pkt_times'].append(packet_time)

                # Direction-specific updates
                dir_data_map = {
                    'pkt_count': 'fwd_pkt_count' if direction in ['fwd', 'uni'] else 'bwd_pkt_count',
                    'bytes_total': 'fwd_bytes_total' if direction in ['fwd', 'uni'] else 'bwd_bytes_total',
                    'frame_lengths': 'fwd_frame_lengths' if direction in ['fwd', 'uni'] else 'bwd_frame_lengths',
                    'pkt_times': 'fwd_pkt_times' if direction in ['fwd', 'uni'] else 'bwd_pkt_times',
                    'signal_dbm': 'fwd_signal_dbm' if direction in ['fwd', 'uni'] else 'bwd_signal_dbm',
                    'noise_dbm': 'fwd_noise_dbm' if direction in ['fwd', 'uni'] else 'bwd_noise_dbm',
                    'rate_mbps': 'fwd_data_rate_mbps' if direction in ['fwd', 'uni'] else 'bwd_data_rate_mbps',
                    'mcs_indices': 'fwd_mcs_indices' if direction in ['fwd', 'uni'] else 'bwd_mcs_indices',
                    'bandwidths_mhz': 'fwd_bandwidths_mhz' if direction in ['fwd', 'uni'] else 'bwd_bandwidths_mhz',
                    'guard_intervals_ns': 'fwd_guard_intervals_ns' if direction in ['fwd', 'uni'] else 'bwd_guard_intervals_ns',
                }
                flow_data[dir_data_map['pkt_count']] += 1
                flow_data[dir_data_map['bytes_total']] += packet_len
                flow_data[dir_data_map['frame_lengths']].append(packet_len)
                flow_data[dir_data_map['pkt_times']].append(packet_time)

                # --- Extract and Store RadioTap Details ---
                rt_details = get_radiotap_details(packet)
                if rt_details['dbm_antsignal'] is not None: flow_data[dir_data_map['signal_dbm']].append(rt_details['dbm_antsignal'])
                if rt_details['dbm_antnoise'] is not None: flow_data[dir_data_map['noise_dbm']].append(rt_details['dbm_antnoise'])
                if rt_details['rate_mbps'] is not None: flow_data[dir_data_map['rate_mbps']].append(rt_details['rate_mbps'])
                if rt_details['mcs_index'] is not None: flow_data[dir_data_map['mcs_indices']].append(rt_details['mcs_index'])
                if rt_details['bandwidth_mhz'] is not None: flow_data[dir_data_map['bandwidths_mhz']][rt_details['bandwidth_mhz']] += 1
                if rt_details['guard_interval_ns'] is not None: flow_data[dir_data_map['guard_intervals_ns']][rt_details['guard_interval_ns']] += 1
                if rt_details['channel_freq']:
                     freq = rt_details['channel_freq']
                     # Simple channel calculation (adjust if needed for specific bands/regulations)
                     if 2412 <= freq <= 2484: channel = int((freq - 2407) / 5) # 2.4 GHz
                     elif 5180 <= freq <= 5825: channel = int((freq - 5000) / 5) # 5 GHz
                     else: channel = freq # Use frequency if outside common bands
                     flow_data['channels'].add(channel)

                # --- Extract and Store Dot11 Details ---
                if packet.haslayer(Dot11):
                    dot11 = packet.getlayer(Dot11)
                    frame_type = dot11.type; frame_subtype = dot11.subtype
                    flow_data['frame_types'][frame_type] += 1
                    if dot11.FCfield.protected: flow_data['protected_count'] += 1
                    if dot11.FCfield.retry: flow_data['retry_count'] += 1

                    # Track subtypes and specific sequences
                    if frame_type == 0: # Management
                        flow_data['mgmt_subtypes'][frame_subtype] += 1
                        if frame_subtype == 11: flow_data['seq_auth_req_seen'] = 1
                        elif frame_subtype == 12: flow_data['seq_auth_resp_seen'] = 1
                        elif frame_subtype == 0: flow_data['seq_assoc_req_seen'] = 1
                        elif frame_subtype == 1: flow_data['seq_assoc_resp_seen'] = 1
                    elif frame_type == 1: # Control
                        flow_data['ctrl_subtypes'][frame_subtype] += 1
                    elif frame_type == 2: # Data
                        flow_data['data_subtypes'][frame_subtype] += 1 # Track data subtypes (e.g., QoS Data)

                    # --- Parse and Store IEs ---
                    if packet.haslayer(Dot11Elt):
                        ies = parse_ies(packet)
                        if 'ssid' in ies: flow_data['ssids'].add(ies['ssid'])
                        if 'rsn_info' in ies and not flow_data['ies']['rsn_info']: flow_data['ies']['rsn_info'] = ies['rsn_info']
                        if 'country_code' in ies and not flow_data['ies']['country_code']: flow_data['ies']['country_code'] = ies['country_code']
                        if 'supported_rates_mbps' in ies:
                            # Merge rates without duplicates
                            current_rates = set(flow_data['ies']['supported_rates_mbps'])
                            current_rates.update(ies['supported_rates_mbps'])
                            flow_data['ies']['supported_rates_mbps'] = sorted(list(current_rates))
                        if 'vendor_specific' in ies: flow_data['ies']['vendor_specific_count'] += len(ies['vendor_specific'])

                    # --- Parse and Store EAPOL Info ---
                    if packet.haslayer(EAPOL):
                        eapol_info = parse_eapol(packet)
                        if 'handshake_msg' in eapol_info:
                            msg = eapol_info['handshake_msg']
                            flow_data['eapol_msgs'][msg] += 1
                            # Track handshake sequence flags
                            if msg == 'M1': flow_data['seq_eapol_m1_seen'] = 1;
                            elif msg == 'M2': flow_data['seq_eapol_m2_seen'] = 1;
                            elif msg == 'M3': flow_data['seq_eapol_m3_seen'] = 1;
                            elif msg == 'M4': flow_data['seq_eapol_m4_seen'] = 1;
                        if eapol_info.get('key_info_key_mic'): flow_data['eapol_mic_present_count'] += 1
                        if eapol_info.get('key_info_install'): flow_data['eapol_install_flag_count'] += 1
                        if 'key_nonce' in eapol_info and eapol_info['key_nonce']:
                             # Determine direction for nonce storage
                             eapol_dir = 'fwd' if flow_data['flow_type'] == 'uni' or (sa, da) == flow_id else 'bwd'
                             flow_data['eapol_key_nonces'][eapol_dir].add(eapol_info['key_nonce'])

                # --- Update Bulk State ---
                # Pass frame_len instead of packet_len if different
                update_bulk_state(flow_data, direction, packet_time, packet_len, BULK_PACKET_THRESHOLD, BULK_SIZE_THRESHOLD)

                # --- Update Last Seen Timestamp ---
                flow_data['last_seen'] = packet_time

            # else: print(f"Flow {flow_id} was removed before processing packet.") # Debug: Flow removed by cleanup

    # except AttributeError as e:
    #     # Potentially ignore packets with missing attributes if common
    #     # print(f"\nAttributeError processing packet: {e} - {packet.summary()}", file=sys.stderr)
    #     pass
    # except IndexError as e:
    #     # Potentially ignore packets causing index errors if common
    #     # print(f"\nIndexError processing packet: {e} - {packet.summary()}", file=sys.stderr)
    #     pass
    except Exception as e:
        print(f"\nGeneral error processing packet: {e}", file=sys.stderr)
        print(f"Packet summary: {packet.summary()}")
        print_exc()


async def packet_worker(worker_id):
    """Async worker task that retrieves packets from the queue and processes them."""
    global packet_queue, stop_capture_event
    print(f"Worker {worker_id} started.")
    while not stop_capture_event.is_set():
        try:
            # Wait for a packet from the queue
            # Use a timeout to allow checking stop_capture_event periodically
            packet = await asyncio.wait_for(packet_queue.get(), timeout=0.5)

            if packet is None: # Sentinel value received
                # print(f"Worker {worker_id} received None sentinel.") # Debug
                packet_queue.task_done()
                break # Exit the loop

            # Process the packet using the async processing function
            await process_packet_async(packet)
            packet_queue.task_done() # Signal that processing for this packet is done

        except asyncio.TimeoutError:
            # No packet received within timeout, loop continues to check stop_capture_event
            continue
        except asyncio.CancelledError:
            print(f"Worker {worker_id} cancelled.")
            break # Exit loop if cancelled
        except Exception as e:
            print(f"\nError in worker {worker_id}: {e}", file=sys.stderr)
            print_exc()
            # Ensure task_done is called even if processing fails
            # Check if get() succeeded before calling task_done if packet is None or error occurred before get()
            if 'packet' in locals() and packet is not None:
                 packet_queue.task_done()

    print(f"Worker {worker_id} finished.")


# --- Synchronous Sniffer Callback and Runner ---

def sniff_callback(packet):
    """Callback function for scapy.sniff (runs in sniffer thread).
       Puts the packet onto the asyncio queue.
    """
    global packet_queue
    try:
        # Non-blocking put. If the queue is full, it will raise asyncio.QueueFull.
        packet_queue.put_nowait(packet)
    except asyncio.QueueFull:
        # This indicates workers can't keep up or queue size is too small
        print("Warning: Packet queue is full, dropping packet. Consider increasing NUM_WORKERS or PACKET_QUEUE_SIZE.", file=sys.stderr)
    except Exception as e:
         # Catch other potential errors during put_nowait
         print(f"Error in sniff_callback putting packet to queue: {e}", file=sys.stderr)


def run_sniffer(iface, bpf_filter, stop_event_check):
    """Target function to run scapy.sniff in a separate thread."""
    print(f"Sniffer thread started on {iface}.")
    try:
        scapy.sniff(
            iface=iface,
            prn=sniff_callback, # Use the simple callback to queue packets
            store=0,            # Do not store packets in memory
            filter=bpf_filter,
            stop_filter=lambda x: stop_event_check.is_set() # Check the asyncio event
        )
    except PermissionError:
        print(f"\nFatal Error: Permission denied on '{iface}'. Try running with sudo.", file=sys.stderr)
        # Signal main thread to stop
        if not stop_event_check.is_set(): stop_event_check.set()
    except OSError as e:
         if "No such device" in str(e):
             print(f"\nFatal Error: Interface '{iface}' not found.", file=sys.stderr)
         else:
             print(f"\nFatal Error starting capture on '{iface}': {e}.", file=sys.stderr)
         if not stop_event_check.is_set(): stop_event_check.set()
    except Exception as e:
        print(f"\nUnexpected error during capture in sniffer thread: {e}", file=sys.stderr)
        print_exc()
        if not stop_event_check.is_set(): stop_event_check.set()
    finally:
        print("Sniffer thread finished.")
        # It's good practice to signal completion, although stop_event might already be set
        if not stop_event_check.is_set():
             print("Sniffer thread finished unexpectedly, setting stop event.")
             stop_event_check.set()


# --- Main Asynchronous Execution ---

async def main():
    global flows, flows_lock, webhook_url, monitor_interface_name, stop_capture_event
    global total_packets_to_capture, capture_start_time, total_bytes_captured
    global IDLE_THRESHOLD, BULK_PACKET_THRESHOLD, BULK_SIZE_THRESHOLD
    global csv_writer, csv_file, csv_header_written, CSV_BATCH_SIZE, packet_queue
    global total_flows, packet_counter # Access globals

    parser = argparse.ArgumentParser(description="Async Live WiFi Monitor -> Webhook & CSV.")
    parser.add_argument('-i', '--interface', type=str, required=True, help="Monitor mode network interface (e.g., wlan0mon). REQUIRED.")
    parser.add_argument('-e', '--endpoint', type=str, default=DEFAULT_ENDPOINT_URL, help=f"Webhook URL endpoint (default: {DEFAULT_ENDPOINT_URL}). Set to empty ('') to disable.")
    parser.add_argument('--csv', type=str, default=None, help="Output CSV file path (e.g., features.csv). If not specified, CSV output is disabled.")
    parser.add_argument('-c', '--count', type=int, default=DEFAULT_PACKET_COUNT, help="Number of packets to capture (0 for indefinite).")
    parser.add_argument('-t', '--timeout', type=int, default=DEFAULT_FLOW_TIMEOUT, help=f"Flow inactivity timeout seconds (default: {DEFAULT_FLOW_TIMEOUT}).")
    parser.add_argument('--cleanup', type=int, default=DEFAULT_CLEANUP_INTERVAL, help=f"Flow timeout check interval seconds (default: {DEFAULT_CLEANUP_INTERVAL}).")
    parser.add_argument('-f', '--filter', type=str, default=None, help="BPF filter string (e.g., 'wlan type data').")
    parser.add_argument('--idle-threshold', type=float, default=IDLE_THRESHOLD, help=f"Seconds threshold to mark flow idle (default: {IDLE_THRESHOLD}).")
    parser.add_argument('--bulk-pkt-thresh', type=int, default=BULK_PACKET_THRESHOLD, help=f"Min packets in sequence for bulk detection (default: {BULK_PACKET_THRESHOLD}).")
    parser.add_argument('--bulk-size-thresh', type=int, default=BULK_SIZE_THRESHOLD, help=f"Min frame size (bytes) for bulk candidate (default: {BULK_SIZE_THRESHOLD}).")
    parser.add_argument('--csv-batch-size', type=int, default=CSV_BATCH_SIZE, help=f"Number of flows to buffer before writing to CSV (default: {CSV_BATCH_SIZE}).")
    parser.add_argument('--workers', type=int, default=NUM_WORKERS, help=f"Number of packet processing workers (default: {NUM_WORKERS}).")
    parser.add_argument('--queue-size', type=int, default=PACKET_QUEUE_SIZE, help=f"Max packets in processing queue (default: {PACKET_QUEUE_SIZE}).")


    args = parser.parse_args()

    # --- Apply Args to Globals ---
    monitor_interface_name = args.interface
    webhook_url = args.endpoint if args.endpoint else None # Set to None if empty string
    output_csv_filename = args.csv
    total_packets_to_capture = args.count
    flow_timeout = args.timeout
    cleanup_interval = args.cleanup
    bpf_filter = args.filter
    IDLE_THRESHOLD = args.idle_threshold
    BULK_PACKET_THRESHOLD = args.bulk_pkt_thresh
    BULK_SIZE_THRESHOLD = args.bulk_size_thresh
    CSV_BATCH_SIZE = args.csv_batch_size
    num_workers = args.workers
    queue_size = args.queue_size

    # Re-initialize queue with specified size
    packet_queue = asyncio.Queue(maxsize=queue_size)

    print("--- Async Live WiFi Monitor ---")
    print(f"Interface: {monitor_interface_name}")
    print(f"Webhook URL: {webhook_url if webhook_url else 'DISABLED'}")
    print(f"CSV Output File: {output_csv_filename if output_csv_filename else 'DISABLED'}")
    if output_csv_filename: print(f"CSV Batch Size: {CSV_BATCH_SIZE}")
    print(f"Packet Limit: {'Infinite' if total_packets_to_capture == 0 else total_packets_to_capture}")
    print(f"Flow Timeout: {flow_timeout}s | Cleanup Interval: {cleanup_interval}s | Idle Threshold: {IDLE_THRESHOLD}s")
    print(f"Bulk Detection: Pkt Thresh={BULK_PACKET_THRESHOLD}, Frame Size Thresh={BULK_SIZE_THRESHOLD} bytes")
    print(f"Packet Workers: {num_workers} | Queue Size: {queue_size}")
    if bpf_filter: print(f"BPF Filter: {bpf_filter}")
    print("Starting capture... Ensure interface is in monitor mode.")
    print("Run 'iw dev <interface> info' to check type and channel.")
    print("Press Ctrl+C to stop.")
    print("-" * 60)

    # --- Setup Signal Handling ---
    loop = asyncio.get_running_loop()

    def shutdown_signal_handler(sig):
        """Sets the stop event when a signal is received."""
        print(f"\nReceived signal {sig.name}. Initiating graceful shutdown...")
        if not stop_capture_event.is_set():
            stop_capture_event.set()

    # Add signal handlers for SIGINT (Ctrl+C) and SIGTERM
    for sig in (signal.SIGINT, signal.SIGTERM):
         try:
             loop.add_signal_handler(sig, shutdown_signal_handler, sig)
         except NotImplementedError:
             # Fallback for systems where add_signal_handler is not supported (like Windows)
             print(f"Warning: loop.add_signal_handler not fully supported on this system for {sig.name}. Using signal.signal fallback.")
             signal.signal(sig, lambda s, f: loop.call_soon_threadsafe(shutdown_signal_handler, s))


    # --- Setup CSV File ---
    # Use try/finally to ensure file is closed even on error during setup
    try:
        if output_csv_filename:
            try:
                file_exists = os.path.isfile(output_csv_filename)
                # Open in append mode ('a')
                csv_file = open(output_csv_filename, 'a', newline='', encoding='utf-8')
                csv_writer = csv.writer(csv_file)
                # Check if header needs to be written (only if file is new or empty)
                if not file_exists or os.path.getsize(output_csv_filename) == 0:
                    print(f"Writing features to new/empty CSV: {output_csv_filename}")
                    csv_header_written = False # Header will be written on first flush
                else:
                    print(f"Appending features to existing CSV: {output_csv_filename}")
                    # Assume header exists, but we don't know the columns yet.
                    # Header keys will be determined by the first batch written.
                    csv_header_written = True # Set to true, but keys are unknown
            except Exception as e:
                print(f"Error opening CSV file '{output_csv_filename}': {e}", file=sys.stderr)
                print("CSV writing disabled.", file=sys.stderr)
                csv_writer = None
                if csv_file: csv_file.close() # Close if opened partially
                csv_file = None

        # --- Setup HTTP Client Session ---
        async with aiohttp.ClientSession() as session:

            # --- Start Cleanup Task ---
            cleanup_task = asyncio.create_task(
                cleanup_timed_out_flows_async(flow_timeout, cleanup_interval, session)
            )

            # --- Start Worker Tasks ---
            print(f"Starting {num_workers} packet processing workers...")
            worker_tasks = [
                asyncio.create_task(packet_worker(i)) for i in range(num_workers)
            ]

            # --- Start Sniffer Thread ---
            print("Starting packet sniffer thread...")
            sniffer_future = loop.run_in_executor(
                None, # Use default ThreadPoolExecutor
                run_sniffer,
                monitor_interface_name,
                bpf_filter,
                stop_capture_event # Pass the event for the sniffer to check
            )

            # --- Main Execution: Wait for Stop Signal ---
            capture_start_time = time.time() # Record start time after setup
            print("Capture running...")

            # Wait until the stop event is set (by signal, packet limit, or sniffer error)
            await stop_capture_event.wait()

            print("\nStop event received. Proceeding with graceful shutdown...")

            # --- Graceful Shutdown ---

            # 1. Wait for Sniffer Thread to Finish
            print("Waiting for sniffer thread to exit...")
            # Wait for the executor future to complete
            try:
                await asyncio.wait_for(sniffer_future, timeout=10.0) # Add timeout
                print("Sniffer thread has exited.")
            except asyncio.TimeoutError:
                 print("Warning: Timeout waiting for sniffer thread to exit.")
            except Exception as e:
                 print(f"Error waiting for sniffer thread: {e}")


            # 2. Signal Workers to Stop and Wait for Queue
            print("Signalling packet workers to stop by adding None sentinel...")
            # Add None sentinel for each worker to ensure they all receive it
            for _ in range(num_workers):
                try:
                    # Use put instead of put_nowait to wait if queue is full briefly
                    await asyncio.wait_for(packet_queue.put(None), timeout=1.0)
                except asyncio.TimeoutError:
                    print("Warning: Timeout putting None sentinel in queue. Workers might not stop cleanly.")
                except Exception as e:
                     print(f"Error putting None sentinel in queue: {e}")

            print("Waiting for packet queue to be fully processed...")
            try:
                await asyncio.wait_for(packet_queue.join(), timeout=30.0) # Wait for queue empty
                print("Packet queue is empty.")
            except asyncio.TimeoutError:
                print("Warning: Timeout waiting for packet queue to empty. Some packets might be lost.")

            # 3. Wait for Worker Tasks to Finish
            print("Waiting for worker tasks to finish...")
            # Wait for all worker tasks to complete
            done, pending = await asyncio.wait(worker_tasks, timeout=10.0)
            if pending:
                print(f"Warning: {len(pending)} worker tasks did not finish gracefully. Cancelling...")
                for task in pending:
                    task.cancel()
                # Wait for cancellations to complete
                await asyncio.gather(*pending, return_exceptions=True)
            print("All worker tasks finished.")


            # 4. Cancel and Wait for Cleanup Task
            print("Cancelling cleanup task...")
            cleanup_task.cancel()
            print("Waiting for cleanup task to finish...")
            await asyncio.gather(cleanup_task, return_exceptions=True) # Wait for it to finish/handle cancellation
            print("Cleanup task finished.")


            # 5. Process Remaining Flows
            print("Processing any remaining flows...")
            final_flows_to_process = {}
            async with flows_lock: # Lock needed to safely get remaining flows
                if flows:
                    print(f"Found {len(flows)} remaining flows.")
                    final_flows_to_process = flows.copy() # Copy data
                    flows.clear() # Clear the original dict

            if final_flows_to_process:
                final_tasks = []
                for flow_id, flow_data in final_flows_to_process.items():
                     final_tasks.append(
                         asyncio.create_task(process_and_output_flow(flow_id, flow_data, session, is_final_cleanup=True))
                     )
                if final_tasks:
                     await asyncio.gather(*final_tasks) # Process final flows concurrently
                print("Final flow processing complete.")
            else:
                print("No remaining flows to process.")


            # 6. Final CSV Flush
            print("Flushing final CSV buffer...")
            flush_buffer_to_csv() # Synchronous flush


    finally:
        # --- Close CSV File ---
        if csv_file:
            try:
                csv_file.close()
                print(f"CSV file '{output_csv_filename}' closed.")
            except Exception as e:
                 print(f"Error closing CSV file: {e}", file=sys.stderr)

        # --- Final Stats ---
        print("-" * 60)
        elapsed_total_time = time.time() - capture_start_time if capture_start_time else 0
        # Note: packet_counter and total_bytes_captured might be slightly inaccurate
        # due to potential race conditions if not strictly locked during increment.
        print(f"Total flows processed: {total_flows}")
        print(f"Total packets processed by workers (approx): {packet_counter}")
        print(f"Total bytes captured by workers (approx): {total_bytes_captured}")
        print(f"Total capture duration: {elapsed_total_time:.2f} seconds")
        print("Script finished.")


# --- Script Entry Point ---
if __name__ == "__main__":
    # Ensure Scapy uses the correct loop if uvloop is installed
    # This might not be strictly necessary but can prevent potential conflicts
    # asyncio.set_event_loop_policy(uvloop.EventLoopPolicy()) # If using uvloop

    # Run the main asynchronous function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
         print("\nKeyboardInterrupt received in main. Exiting.")
    except Exception as e:
         print(f"\nUnhandled exception in main: {e}")
         print_exc()


