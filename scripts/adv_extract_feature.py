#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import math
import sys
from nfstream import NFStreamer

def calculate_duration_stats(time_intervals_ms):
    """
    Calculates statistics (mean, std, max, min) for a list of time intervals.

    Args:
        time_intervals_ms: A list of tuples, where each tuple is (start_ms, end_ms).

    Returns:
        A dictionary containing 'Mean', 'Std', 'Max', 'Min' duration in milliseconds.
        Returns stats as 0 if no intervals are provided.
    """
    if not time_intervals_ms:
        return {'Mean': 0.0, 'Std': 0.0, 'Max': 0.0, 'Min': 0.0}

    durations = [(end - start) for start, end in time_intervals_ms]

    if not durations:
         return {'Mean': 0.0, 'Std': 0.0, 'Max': 0.0, 'Min': 0.0}

    stats = {
        'Mean': float(np.mean(durations)),
        'Std': float(np.std(durations)),
        'Max': float(np.max(durations)),
        'Min': float(np.min(durations))
    }
    return stats

def safe_division(numerator, denominator, zero_result=0.0, inf_result=math.inf):
    """Performs division safely, handling zero denominators."""
    if denominator == 0 or denominator is None:
        if numerator == 0:
            return zero_result # 0/0 case
        else:
            # Return infinity or a large number, depending on context
            # For rates, infinity might be appropriate
            return inf_result
    try:
        result = float(numerator) / float(denominator)
        # Handle potential NaN or Inf from numpy types if applicable
        if np.isnan(result) or np.isinf(result):
             return inf_result # Or zero_result based on desired behavior for edge cases
        return result
    except (ZeroDivisionError, TypeError, ValueError):
        # Catch potential issues if inputs aren't numbers as expected
         return inf_result # Or zero_result

def extract_flow_features(flow):
    """
    Extracts a predefined set of features from an NFStream NFlow object.

    Args:
        flow: An NFStream NFlow object.

    Returns:
        A dictionary containing the extracted features.
    """
    features = {}

    # Basic Flow Identifiers
    features['Destination Port'] = flow.dst_port
    features['ip_proto'] = flow.protocol # Protocol number
    features['timestamp'] = flow.bidirectional_first_seen_ms # Start timestamp

    # Duration
    duration_sec = flow.bidirectional_duration_ms / 1000.0
    features['Flow Duration'] = duration_sec # In seconds

    # Packet Counts
    features['Total Fwd Packets'] = flow.src2dst_packets
    features['Total Backward Packets'] = flow.dst2src_packets

    # Byte Counts
    features['Total Length of Fwd Packets'] = flow.src2dst_bytes
    features['Total Length of Bwd Packets'] = flow.dst2src_bytes

    # Packet Length Statistics (Forward) - Requires statistical_analysis=True
    features['Fwd Packet Length Max'] = getattr(flow, 'src2dst_max_ps', 0)
    features['Fwd Packet Length Min'] = getattr(flow, 'src2dst_min_ps', 0)
    features['Fwd Packet Length Mean'] = getattr(flow, 'src2dst_mean_ps', 0.0)
    features['Fwd Packet Length Std'] = getattr(flow, 'src2dst_stddev_ps', 0.0)

    # Packet Length Statistics (Backward) - Requires statistical_analysis=True
    features['Bwd Packet Length Max'] = getattr(flow, 'dst2src_max_ps', 0)
    features['Bwd Packet Length Min'] = getattr(flow, 'dst2src_min_ps', 0)
    features['Bwd Packet Length Mean'] = getattr(flow, 'dst2src_mean_ps', 0.0)
    features['Bwd Packet Length Std'] = getattr(flow, 'dst2src_stddev_ps', 0.0)

    # Flow Rate Statistics
    features['Flow Bytes/s'] = safe_division(flow.bidirectional_bytes, duration_sec)
    features['Flow Packets/s'] = safe_division(flow.bidirectional_packets, duration_sec)

    # Inter-Arrival Time (IAT) Statistics (Flow) - Requires statistical_analysis=True
    features['Flow IAT Mean'] = getattr(flow, 'bidirectional_mean_piat_ms', 0.0)
    features['Flow IAT Std'] = getattr(flow, 'bidirectional_stddev_piat_ms', 0.0)
    features['Flow IAT Max'] = getattr(flow, 'bidirectional_max_piat_ms', 0.0)
    features['Flow IAT Min'] = getattr(flow, 'bidirectional_min_piat_ms', 0.0)

    # Inter-Arrival Time (IAT) Statistics (Forward) - Requires statistical_analysis=True
    fwd_duration_sec = flow.src2dst_duration_ms / 1000.0
    features['Fwd IAT Total'] = fwd_duration_sec * 1000.0 # src2dst_duration_ms is total time
    features['Fwd IAT Mean'] = getattr(flow, 'src2dst_mean_piat_ms', 0.0)
    features['Fwd IAT Std'] = getattr(flow, 'src2dst_stddev_piat_ms', 0.0)
    features['Fwd IAT Max'] = getattr(flow, 'src2dst_max_piat_ms', 0.0)
    features['Fwd IAT Min'] = getattr(flow, 'src2dst_min_piat_ms', 0.0)

    # Inter-Arrival Time (IAT) Statistics (Backward) - Requires statistical_analysis=True
    bwd_duration_sec = flow.dst2src_duration_ms / 1000.0
    features['Bwd IAT Total'] = bwd_duration_sec * 1000.0 # dst2src_duration_ms is total time
    features['Bwd IAT Mean'] = getattr(flow, 'dst2src_mean_piat_ms', 0.0)
    features['Bwd IAT Std'] = getattr(flow, 'dst2src_stddev_piat_ms', 0.0)
    features['Bwd IAT Max'] = getattr(flow, 'dst2src_max_piat_ms', 0.0)
    features['Bwd IAT Min'] = getattr(flow, 'dst2src_min_piat_ms', 0.0)

    # Flags Counts (Forward) - Requires statistical_analysis=True
    features['Fwd PSH Flags'] = getattr(flow, 'src2dst_psh_packets', 0)
    features['Fwd URG Flags'] = getattr(flow, 'src2dst_urg_packets', 0)
    # Note: NFStream provides counts of packets *with* the flag set.

    # Flags Counts (Backward) - Requires statistical_analysis=True
    features['Bwd PSH Flags'] = getattr(flow, 'dst2src_psh_packets', 0)
    features['Bwd URG Flags'] = getattr(flow, 'dst2src_urg_packets', 0)

    # Header Length
    features['Fwd Header Length'] = flow.src2dst_header_bytes
    features['Bwd Header Length'] = flow.dst2src_header_bytes
    # Note: This is total header bytes for the flow direction. User list might imply per-packet avg?
    # The user list has 'Fwd Header Length' twice. Using the total bytes here.

    # Packet Rate Statistics
    features['Fwd Packets/s'] = safe_division(flow.src2dst_packets, fwd_duration_sec)
    features['Bwd Packets/s'] = safe_division(flow.dst2src_packets, bwd_duration_sec)

    # Packet Length Statistics (Bidirectional) - Requires statistical_analysis=True
    features['Min Packet Length'] = getattr(flow, 'bidirectional_min_ps', 0)
    features['Max Packet Length'] = getattr(flow, 'bidirectional_max_ps', 0)
    features['Packet Length Mean'] = getattr(flow, 'bidirectional_mean_ps', 0.0)
    bidirectional_stddev_ps = getattr(flow, 'bidirectional_stddev_ps', 0.0)
    features['Packet Length Std'] = bidirectional_stddev_ps
    features['Packet Length Variance'] = bidirectional_stddev_ps ** 2 if bidirectional_stddev_ps is not None else 0.0

    # Flag Counts (Bidirectional) - Requires statistical_analysis=True
    features['FIN Flag Count'] = getattr(flow, 'bidirectional_fin_packets', 0)
    features['SYN Flag Count'] = getattr(flow, 'bidirectional_syn_packets', 0)
    features['RST Flag Count'] = getattr(flow, 'bidirectional_rst_packets', 0)
    features['PSH Flag Count'] = getattr(flow, 'bidirectional_psh_packets', 0)
    features['ACK Flag Count'] = getattr(flow, 'bidirectional_ack_packets', 0)
    features['URG Flag Count'] = getattr(flow, 'bidirectional_urg_packets', 0)
    features['CWE Flag Count'] = getattr(flow, 'bidirectional_cwr_packets', 0) # Note: CWR in NFStream
    features['ECE Flag Count'] = getattr(flow, 'bidirectional_ece_packets', 0)

    # Ratio and Average Sizes
    features['Down/Up Ratio'] = safe_division(flow.dst2src_packets, flow.src2dst_packets, zero_result=0.0, inf_result=0.0) # Ratio usually 0 or 1 if one side is 0
    features['Average Packet Size'] = getattr(flow, 'bidirectional_mean_ps', 0.0)
    features['Avg Fwd Segment Size'] = getattr(flow, 'src2dst_mean_ps', 0.0)
    features['Avg Bwd Segment Size'] = getattr(flow, 'dst2src_mean_ps', 0.0)

    # Bulk Rate Features - Not directly supported by default NFStream features
    # features['Fwd Avg Bytes/Bulk'] = 0
    # features['Fwd Avg Packets/Bulk'] = 0
    # features['Fwd Avg Bulk Rate'] = 0
    # features['Bwd Avg Bytes/Bulk'] = 0
    # features['Bwd Avg Packets/Bulk'] = 0
    # features['Bwd Avg Bulk Rate'] = 0

    # Subflow Features - Not directly supported by default NFStream features
    # features['Subflow Fwd Packets'] = 0
    # features['Subflow Fwd Bytes'] = 0
    # features['Subflow Bwd Packets'] = 0
    # features['Subflow Bwd Bytes'] = 0

    # Initial Window Sizes
    features['Init_Win_bytes_forward'] = getattr(flow, 'src_init_win_bytes', 0)
    features['Init_Win_bytes_backward'] = getattr(flow, 'dst_init_win_bytes', 0)

    # Other TCP Features
    features['act_data_pkt_fwd'] = getattr(flow, 'src_act_data_pkts', 0) # Packets with payload > 0
    features['min_seg_size_forward'] = getattr(flow, 'src_min_seg_size', 0) # Minimum segment size observed (might be MSS option)

    # Active/Idle Time Statistics - Requires idle/active timeouts set
    active_stats = calculate_duration_stats(getattr(flow, 'active', []))
    features['Active Mean'] = active_stats['Mean']
    features['Active Std'] = active_stats['Std']
    features['Active Max'] = active_stats['Max']
    features['Active Min'] = active_stats['Min']

    idle_stats = calculate_duration_stats(getattr(flow, 'idle', []))
    features['Idle Mean'] = idle_stats['Mean']
    features['Idle Std'] = idle_stats['Std']
    features['Idle Max'] = idle_stats['Max']
    features['Idle Min'] = idle_stats['Min']

    # Payload size (using application bytes as proxy)
    features['payload_size'] = flow.bidirectional_application_bytes

    # Packet level features (ip_len, ip_ttl, packet_len) are not directly available per flow
    # features['ip_len'] = 0 # Not applicable at flow level
    # features['ip_ttl'] = 0 # Not applicable at flow level
    # features['packet_len'] = 0 # Not applicable at flow level

    # TCP ports (already captured as src/dst port, depending on direction)
    features['tcp_sport'] = flow.src_port
    # features['tcp_dport'] = flow.dst_port # Already captured as 'Destination Port'

    return features

def main():
    parser = argparse.ArgumentParser(description="Extract features from a PCAP file using NFStream.")
    parser.add_argument("pcap_file", help="Path to the input PCAP file.")
    parser.add_argument("-o", "--output", default="pcap_features.csv", help="Path to the output CSV file (default: pcap_features.csv).")
    parser.add_argument("--idle-timeout", type=int, default=120, help="Idle timeout in seconds for flow expiration (default: 120).")
    parser.add_argument("--active-timeout", type=int, default=5, help="Active timeout in seconds for flow expiration (default: 5).")

    args = parser.parse_args()

    print(f"[*] Processing PCAP file: {args.pcap_file}")
    print(f"[*] Idle Timeout: {args.idle_timeout}s, Active Timeout: {args.active_timeout}s")
    print(f"[*] Output will be saved to: {args.output}")

    all_flow_features = []

    try:
        # Initialize NFStream from PCAP
        # Enable statistical analysis and set timeouts for active/idle calculation
        stream = NFStreamer(
            source=args.pcap_file,
            idle_timeout=args.idle_timeout * 1000, # Convert to ms
            active_timeout=args.active_timeout * 1000 # Convert to ms
        )

        print("[*] Starting feature extraction...")
        processed_flows = 0
        for flow in stream:
            flow_data = extract_flow_features(flow)
            all_flow_features.append(flow_data)
            processed_flows += 1
            if processed_flows % 100 == 0:
                 print(f"\r[*] Processed {processed_flows} flows...", end="")

        print(f"\n[*] Finished processing. Extracted features from {processed_flows} flows.")

        if not all_flow_features:
            print("[!] No flows found or extracted.")
            sys.exit(0)

        # Convert to Pandas DataFrame
        df = pd.DataFrame(all_flow_features)

        # Reorder columns to match the user's requested order as much as possible
        # (Some requested features are omitted or renamed)
        requested_order = [
            'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max',
            'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
            'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
            'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
            'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean',
            'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
            'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
            'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length',
            'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length',
            'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
            'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
            'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size',
            'Avg Fwd Segment Size', 'Avg Bwd Segment Size', #'Fwd Header Length', # Duplicated
            # 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', # Omitted
            # 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', # Omitted
            # 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes', # Omitted
            'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd',
            'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max', 'Active Min',
            'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'payload_size', 'ip_proto',
            # 'ip_len', 'ip_ttl', # Omitted
            'tcp_sport', #'tcp_dport', # Duplicated
            'timestamp',
            # 'packet_len' # Omitted
        ]

        # Get actual columns present in the DataFrame
        actual_columns = df.columns.tolist()
        # Create final column order: include requested columns that exist, then add any remaining
        final_columns = [col for col in requested_order if col in actual_columns]
        remaining_columns = [col for col in actual_columns if col not in final_columns]
        final_columns.extend(remaining_columns)

        df = df[final_columns]


        # Save to CSV
        df.to_csv(args.output, index=False)
        print(f"[+] Features saved successfully to {args.output}")

    except FileNotFoundError:
        print(f"[-] Error: PCAP file not found at {args.pcap_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[-] An error occurred during processing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
