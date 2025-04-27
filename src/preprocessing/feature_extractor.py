# src/preprocessing/feature_extractor.py

import pandas as pd
import numpy as np
from scapy.all import rdpcap, PcapReader, RadioTap, Dot11, IP, TCP, UDP, Raw
from collections import Counter
import logging
import os
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Feature Extraction Functions ---

def extract_basic_features(packet):
    """
    Extracts basic features from a single packet.
    Focuses on RadioTap, Dot11, IP, TCP, UDP layers.
    Returns a dictionary of features or None if layers are missing.
    """
    features = {}
    timestamp = float(packet.time) # Packet arrival time

    # --- RadioTap Layer (Physical Layer Info) ---
    signal_strength = None
    channel_freq = None
    data_rate = None
    if packet.haslayer(RadioTap):
        # Note: Field presence and names can vary based on driver/hardware
        try:
            # Try common field names/locations for signal strength
            if hasattr(packet[RadioTap], 'dbm_antsignal'):
                signal_strength = packet[RadioTap].dbm_antsignal
            elif hasattr(packet[RadioTap], 'dBm_AntSignal'):
                 signal_strength = packet[RadioTap].dBm_AntSignal
            # Add more checks if needed based on observed RadioTap formats

            if hasattr(packet[RadioTap], 'ChannelFrequency'):
                channel_freq = packet[RadioTap].ChannelFrequency
            elif hasattr(packet[RadioTap], 'channel'): # Sometimes nested
                 if hasattr(packet[RadioTap].channel, 'freq'):
                     channel_freq = packet[RadioTap].channel.freq

            if hasattr(packet[RadioTap], 'Rate'):
                data_rate = packet[RadioTap].Rate

        except AttributeError as e:
            # logging.debug(f"Attribute error accessing RadioTap field: {e}")
            pass # Ignore if specific fields are missing

    features['timestamp'] = timestamp
    features['signal_strength'] = signal_strength if signal_strength is not None else np.nan
    features['channel_freq'] = channel_freq if channel_freq is not None else np.nan
    features['data_rate'] = data_rate if data_rate is not None else np.nan
    features['packet_len'] = len(packet) # Overall packet length

    # --- Dot11 Layer (MAC Layer Info) ---
    dot11_present = packet.haslayer(Dot11)
    features['dot11_present'] = 1 if dot11_present else 0

    if dot11_present:
        dot11_layer = packet[Dot11]
        features['dot11_type'] = dot11_layer.type # 0: Mgmt, 1: Ctrl, 2: Data
        features['dot11_subtype'] = dot11_layer.subtype
        features['dot11_addr1'] = dot11_layer.addr1 # RA (Receiver Address) or DA (Destination Address)
        features['dot11_addr2'] = dot11_layer.addr2 # TA (Transmitter Address) or SA (Source Address)
        features['dot11_addr3'] = dot11_layer.addr3 # BSSID, DA/SA depending on flags
        features['dot11_addr4'] = dot11_layer.addr4 # Only in WDS frames

        # Frame Control Field Flags (Example: Check Protected Frame bit)
        fc_field = dot11_layer.FCfield
        features['dot11_fc_protected'] = 1 if fc_field.protected else 0
        features['dot11_fc_retry'] = 1 if fc_field.retry else 0
        features['dot11_fc_morefrag'] = 1 if fc_field.morefrag else 0
        features['dot11_fc_tods'] = 1 if fc_field.tods else 0
        features['dot11_fc_fromds'] = 1 if fc_field.fromds else 0

        # Duration/ID field
        features['dot11_duration_id'] = dot11_layer.ID if hasattr(dot11_layer, 'ID') else np.nan

    else:
        # Fill with default values if Dot11 layer is missing
        features.update({
            'dot11_type': np.nan, 'dot11_subtype': np.nan,
            'dot11_addr1': None, 'dot11_addr2': None, 'dot11_addr3': None, 'dot11_addr4': None,
            'dot11_fc_protected': 0, 'dot11_fc_retry': 0, 'dot11_fc_morefrag': 0,
            'dot11_fc_tods': 0, 'dot11_fc_fromds': 0, 'dot11_duration_id': np.nan
        })

    # --- Higher Layers (IP/TCP/UDP) ---
    ip_present = packet.haslayer(IP)
    tcp_present = packet.haslayer(TCP)
    udp_present = packet.haslayer(UDP)
    payload_size = len(packet[Raw]) if packet.haslayer(Raw) else 0

    features['ip_present'] = 1 if ip_present else 0
    features['tcp_present'] = 1 if tcp_present else 0
    features['udp_present'] = 1 if udp_present else 0
    features['payload_size'] = payload_size

    if ip_present:
        ip_layer = packet[IP]
        features['ip_src'] = ip_layer.src
        features['ip_dst'] = ip_layer.dst
        features['ip_proto'] = ip_layer.proto
        features['ip_len'] = ip_layer.len
        features['ip_ttl'] = ip_layer.ttl
    else:
        features.update({'ip_src': None, 'ip_dst': None, 'ip_proto': np.nan, 'ip_len': np.nan, 'ip_ttl': np.nan})

    if tcp_present:
        tcp_layer = packet[TCP]
        features['tcp_sport'] = tcp_layer.sport
        features['tcp_dport'] = tcp_layer.dport
        features['tcp_seq'] = tcp_layer.seq
        features['tcp_ack'] = tcp_layer.ack
        features['tcp_flags'] = str(tcp_layer.flags) # Represent flags as string (e.g., 'S', 'SA', 'A')
        features['tcp_window'] = tcp_layer.window
    else:
        features.update({'tcp_sport': np.nan, 'tcp_dport': np.nan, 'tcp_seq': np.nan, 'tcp_ack': np.nan, 'tcp_flags': None, 'tcp_window': np.nan})

    if udp_present:
        udp_layer = packet[UDP]
        features['udp_sport'] = udp_layer.sport
        features['udp_dport'] = udp_layer.dport
        features['udp_len'] = udp_layer.len
    else:
        features.update({'udp_sport': np.nan, 'udp_dport': np.nan, 'udp_len': np.nan})

    return features

def process_pcap(pcap_file_path):
    """
    Reads a PCAP file, extracts features from each packet, and returns a list of feature dicts.
    Uses PcapReader for memory efficiency with large files.
    """
    all_features = []
    packet_count = 0
    logging.info(f"Starting processing of PCAP file: {pcap_file_path}")
    try:
        with PcapReader(pcap_file_path) as pcap_reader:
            for packet in pcap_reader:
                packet_count += 1
                features = extract_basic_features(packet)
                if features:
                    all_features.append(features)

                if packet_count % 1000 == 0:
                    logging.info(f"Processed {packet_count} packets...")

    except FileNotFoundError:
        logging.error(f"PCAP file not found: {pcap_file_path}")
        return None
    except Exception as e:
        logging.error(f"Error processing PCAP file {pcap_file_path}: {e}", exc_info=True)
        return None

    logging.info(f"Finished processing PCAP file. Extracted features for {len(all_features)} packets out of {packet_count} total.")
    return all_features

def features_to_dataframe(feature_list):
    """Converts a list of feature dictionaries into a Pandas DataFrame."""
    if not feature_list:
        return pd.DataFrame()
    df = pd.DataFrame(feature_list)
    # Basic type conversion and handling (can be expanded)
    # Convert MAC/IP addresses to string type if they exist
    for col in ['dot11_addr1', 'dot11_addr2', 'dot11_addr3', 'dot11_addr4', 'ip_src', 'ip_dst', 'tcp_flags']:
         if col in df.columns:
            df[col] = df[col].astype(str).fillna('None') # Use 'None' string for missing addresses/flags

    # Convert numeric columns, coercing errors to NaN
    numeric_cols = [
        'timestamp', 'signal_strength', 'channel_freq', 'data_rate', 'packet_len',
        'dot11_present', 'dot11_type', 'dot11_subtype', 'dot11_fc_protected',
        'dot11_fc_retry', 'dot11_fc_morefrag', 'dot11_fc_tods', 'dot11_fc_fromds',
        'dot11_duration_id', 'ip_present', 'tcp_present', 'udp_present', 'payload_size',
        'ip_proto', 'ip_len', 'ip_ttl', 'tcp_sport', 'tcp_dport', 'tcp_seq', 'tcp_ack',
        'tcp_window', 'udp_sport', 'udp_dport', 'udp_len'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    logging.info(f"Created DataFrame with shape: {df.shape}")
    return df

def save_features(df, output_path):
    """Saves the DataFrame to a CSV file."""
    if df.empty:
        logging.warning("DataFrame is empty. No file saved.")
        return
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"Features saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving DataFrame to {output_path}: {e}", exc_info=True)

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from WiFi PCAP files.")
    parser.add_argument("pcap_file", help="Path to the input PCAP file.")
    parser.add_argument("-o", "--output", default="data/processed/extracted_features.csv",
                        help="Path to save the output CSV file.")
    args = parser.parse_args()

    # 1. Process PCAP and Extract Features
    extracted_data = process_pcap(args.pcap_file)

    if extracted_data:
        # 2. Convert to DataFrame
        features_df = features_to_dataframe(extracted_data)

        # 3. Save DataFrame to CSV
        save_features(features_df, args.output)
    else:
        logging.error("Feature extraction failed.")

