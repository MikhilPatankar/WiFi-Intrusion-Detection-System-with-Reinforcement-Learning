# src/preprocessing/feature_extractor.py
# Extracts features from PCAP files and saves to CSV.
# NOTE: This version does NOT add labels automatically unless parsing specific filenames
# or integrating with a labeled source. Labels might need to be merged later.

import pandas as pd
import numpy as np
from scapy.all import PcapReader, RadioTap, Dot11, IP, TCP, UDP, Raw
import logging
import os
import argparse
from pathlib import Path # Use pathlib for better path handling
import sys
from typing import List, Dict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Feature Extraction Functions ---

def extract_basic_features(packet):
    """
    Extracts features from a single packet. Returns dict or None.
    (Includes handling for common RadioTap variations)
    """
    features = {}
    try:
        timestamp = float(packet.time)
    except AttributeError:
        log.warning("Packet missing time attribute.")
        return None # Cannot process without timestamp

    features['timestamp'] = timestamp
    features['packet_len'] = len(packet)

    # RadioTap Layer
    signal_strength, channel_freq, data_rate = np.nan, np.nan, np.nan
    if packet.haslayer(RadioTap):
        # Iterate through RadioTap fields safely
        field_values = {}
        try:
             # Scapy's RadioTap layer decoding can be complex. Iterate fields.
             layer = packet[RadioTap]
             # This part might need adjustment based on specific Scapy versions & drivers
             # Example: Check common attributes directly first
             if hasattr(layer, 'dbm_antsignal'): field_values['signal'] = layer.dbm_antsignal
             elif hasattr(layer, 'dBm_AntSignal'): field_values['signal'] = layer.dBm_AntSignal
             if hasattr(layer, 'ChannelFrequency'): field_values['channel'] = layer.ChannelFrequency
             elif hasattr(layer, 'Channel'): field_values['channel'] = layer.Channel # Sometimes just channel number
             if hasattr(layer, 'Rate'): field_values['rate'] = layer.Rate

             # Fallback: Iterate present fields (more robust but slower)
             # if hasattr(layer, 'present_fields'):
             #     for field, value in layer.present_fields:
             #         # Map known field names to standardized keys
             #         if field in ['dbm_antsignal', 'dBm_AntSignal']: field_values['signal'] = value
             #         elif field in ['ChannelFrequency', 'Channel']: field_values['channel'] = value
             #         elif field == 'Rate': field_values['rate'] = value
        except Exception as e:
             log.debug(f"Error parsing RadioTap fields: {e}")
             pass # Ignore RadioTap parsing errors

        signal_strength = field_values.get('signal', np.nan)
        channel_freq = field_values.get('channel', np.nan)
        data_rate = field_values.get('rate', np.nan)

    features['signal_strength'] = signal_strength
    features['channel_freq'] = channel_freq
    features['data_rate'] = data_rate

    # Dot11 Layer
    if packet.haslayer(Dot11):
        dot11 = packet[Dot11]
        features['dot11_type'] = dot11.type
        features['dot11_subtype'] = dot11.subtype
        # Frame Control Flags (example)
        features['dot11_fc_tods'] = 1 if dot11.FCfield & 0x1 else 0
        features['dot11_fc_fromds'] = 1 if dot11.FCfield & 0x2 else 0
        features['dot11_fc_morefrag'] = 1 if dot11.FCfield & 0x4 else 0
        features['dot11_fc_retry'] = 1 if dot11.FCfield & 0x8 else 0
        features['dot11_fc_protected'] = 1 if dot11.FCfield & 0x40 else 0
        features['dot11_duration_id'] = dot11.ID
    else:
        # Fill with defaults if Dot11 missing (important for consistent columns)
        features.update({k: np.nan for k in ['dot11_type', 'dot11_subtype', 'dot11_duration_id']})
        features.update({k: 0 for k in ['dot11_fc_tods', 'dot11_fc_fromds', 'dot11_fc_morefrag', 'dot11_fc_retry', 'dot11_fc_protected']})

    # Higher Layers (IP/TCP/UDP) - Example features
    features['ip_present'] = 1 if packet.haslayer(IP) else 0
    features['tcp_present'] = 1 if packet.haslayer(TCP) else 0
    features['udp_present'] = 1 if packet.haslayer(UDP) else 0
    features['payload_size'] = len(packet[Raw]) if packet.haslayer(Raw) else 0

    if features['ip_present']:
        ip = packet[IP]
        features['ip_proto'] = ip.proto
        features['ip_len'] = ip.len
        features['ip_ttl'] = ip.ttl
    else:
        features.update({k: np.nan for k in ['ip_proto', 'ip_len', 'ip_ttl']})

    if features['tcp_present']:
        tcp = packet[TCP]
        features['tcp_sport'] = tcp.sport
        features['tcp_dport'] = tcp.dport
        # TCP Flags (example: individual flags)
        flags = tcp.flags
        features['tcp_flag_syn'] = 1 if 'S' in flags else 0
        features['tcp_flag_ack'] = 1 if 'A' in flags else 0
        features['tcp_flag_fin'] = 1 if 'F' in flags else 0
        features['tcp_flag_rst'] = 1 if 'R' in flags else 0
        features['tcp_flag_psh'] = 1 if 'P' in flags else 0
        features['tcp_flag_urg'] = 1 if 'U' in flags else 0
    else:
        features.update({k: np.nan for k in ['tcp_sport', 'tcp_dport']})
        features.update({k: 0 for k in ['tcp_flag_syn', 'tcp_flag_ack', 'tcp_flag_fin', 'tcp_flag_rst', 'tcp_flag_psh', 'tcp_flag_urg']})

    if features['udp_present']:
        udp = packet[UDP]
        features['udp_sport'] = udp.sport
        features['udp_dport'] = udp.dport
        features['udp_len'] = udp.len
    else:
        features.update({k: np.nan for k in ['udp_sport', 'udp_dport', 'udp_len']})

    # --- Add more features as needed ---
    # e.g., MAC addresses, protocol specific details, flow features (requires state)

    return features

def process_pcap(pcap_file_path: Path) -> List[Dict]:
    """Reads PCAP, extracts features per packet."""
    all_features = []
    packet_count = 0
    log.info(f"Starting processing of PCAP: {pcap_file_path}")
    try:
        with PcapReader(str(pcap_file_path)) as pcap_reader:
            for packet in pcap_reader:
                packet_count += 1
                features = extract_basic_features(packet)
                if features:
                    all_features.append(features)
                if packet_count % 5000 == 0:
                    log.info(f"Processed {packet_count} packets from {pcap_file_path.name}...")
    except FileNotFoundError:
        log.error(f"PCAP file not found: {pcap_file_path}")
        return []
    except Exception as e:
        log.error(f"Error processing {pcap_file_path}: {e}", exc_info=True)
        return []
    log.info(f"Finished {pcap_file_path.name}. Extracted features for {len(all_features)} packets out of {packet_count}.")
    return all_features

def features_to_dataframe(feature_list: List[Dict]) -> pd.DataFrame:
    """Converts list of feature dicts to DataFrame and performs basic cleaning."""
    if not feature_list:
        return pd.DataFrame()
    df = pd.DataFrame(feature_list)
    log.info(f"Created DataFrame with shape: {df.shape}")

    # Basic type conversion (ensure numeric types where expected)
    numeric_cols = [col for col, dtype in df.dtypes.items() if dtype in ['int64', 'float64']]
    # Convert potentially object columns that should be numeric
    potential_numeric = [
        'signal_strength', 'channel_freq', 'data_rate', 'packet_len',
        'dot11_type', 'dot11_subtype', 'dot11_duration_id',
        'ip_proto', 'ip_len', 'ip_ttl', 'tcp_sport', 'tcp_dport', 'udp_sport', 'udp_dport', 'udp_len', 'payload_size'
    ]
    for col in potential_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce errors to NaN

    # Handle NaNs (Example: fill with 0, consider median/mean for specific features)
    # IMPORTANT: Consistent NaN handling is crucial across all stages.
    # Using 0 might be okay for flags/counts, but not ideal for things like signal strength.
    # A better approach is to handle NaNs during scaling/training based on training set stats.
    # For simplicity here, we fill globally.
    df.fillna(0, inplace=True)
    log.info("Filled NaN values with 0 (consider more sophisticated imputation).")

    # Ensure consistent column order (optional but good practice)
    # Define your desired final column order here if needed
    # final_columns = ['timestamp', 'packet_len', 'signal_strength', ... , 'label'] # If label exists
    # df = df[final_columns]

    return df

def save_features(df: pd.DataFrame, output_path: Path):
    """Saves DataFrame to CSV."""
    if df.empty:
        log.warning("DataFrame is empty. No file saved.")
        return
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True) # Create directory if needed
        df.to_csv(output_path, index=False)
        log.info(f"Features saved to {output_path}")
    except Exception as e:
        log.error(f"Error saving DataFrame to {output_path}: {e}", exc_info=True)

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from WiFi PCAP files.")
    parser.add_argument("pcap_input", help="Path to the input PCAP file or directory containing PCAP files.")
    parser.add_argument("-o", "--output", default="../../data/processed/extracted_features.csv",
                        help="Path to save the output CSV file.")
    # Add argument for label (optional, e.g., if processing files for specific classes)
    parser.add_argument("--label", type=int, default=None, help="Optional label to assign to all packets in the file (e.g., 0 for normal, 1 for anomaly).")

    args = parser.parse_args()

    input_path = Path(args.pcap_input)
    output_path = Path(args.output)
    label = args.label

    all_extracted_data = []

    if input_path.is_file():
        all_extracted_data.extend(process_pcap(input_path))
    elif input_path.is_dir():
        log.info(f"Processing all PCAP files in directory: {input_path}")
        for pcap_file in input_path.glob('*.pcap'): # Adjust glob pattern if needed (*.pcapng etc.)
            all_extracted_data.extend(process_pcap(pcap_file))
    else:
        log.error(f"Input path is not a valid file or directory: {input_path}")
        sys.exit(1)

    if all_extracted_data:
        features_df = features_to_dataframe(all_extracted_data)

        # Assign label if provided via command line
        if label is not None:
            log.info(f"Assigning label '{label}' to all {len(features_df)} extracted records.")
            features_df['label'] = label
        elif 'label' not in features_df.columns:
             log.warning("No label column generated and --label not provided. Output will lack labels.")
             # Optionally add a default label (e.g., -1 for unknown)
             # features_df['label'] = -1

        save_features(features_df, output_path)
    else:
        log.error("No features were extracted.")

