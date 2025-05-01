#!/usr/bin/env python3

import argparse
import numpy as np
import math
import sys
import time
from nfstream import NFStreamer

# --- Helper functions (calculate_duration_stats, safe_division, extract_flow_features) ---
# --- Paste the exact same helper functions from the PCAP script here ---

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
    features['src_ip'] = flow.src_ip # Add IPs for context in live view
    features['dst_ip'] = flow.dst_ip # Add IPs for context in live view
    features['src_port'] = flow.src_port # Add IPs for context in live view


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
    # features['tcp_sport'] = flow.src_port # Already captured
    # features['tcp_dport'] = flow.dst_port # Already captured as 'Destination Port'

    return features
# --- End of Helper Functions ---


def main():
    parser = argparse.ArgumentParser(description="Extract features from live network traffic using NFStream.")
    parser.add_argument("-i", "--interface", required=True, help="Network interface name to capture from (e.g., eth0, wlan0).")
    parser.add_argument("--idle-timeout", type=int, default=120, help="Idle timeout in seconds for flow expiration (default: 120).")
    parser.add_argument("--active-timeout", type=int, default=5, help="Active timeout in seconds for flow expiration (default: 5).")

    args = parser.parse_args()

    print(f"[*] Starting live capture on interface: {args.interface}")
    print(f"[*] Idle Timeout: {args.idle_timeout}s, Active Timeout: {args.active_timeout}s")
    print("[*] Press Ctrl+C to stop.")

    try:
        # Initialize NFStream for live capture
        # Enable statistical analysis and set timeouts
        stream = NFStreamer(
            source=args.interface,
            idle_timeout=args.idle_timeout * 1000, # Convert to ms
            active_timeout=args.active_timeout * 1000, # Convert to ms
            # promisc_mode=True # Usually default, but can be explicit
        )

        print("[*] Waiting for flows...")
        flow_count = 0
        for flow in stream:
            flow_count += 1
            print("-" * 80)
            print(f"Flow #{flow_count} Detected:")
            try:
                features = extract_flow_features(flow)
                # Print features in a readable format
                for key, value in features.items():
                    # Format floats for better readability
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
            except Exception as e:
                print(f"[!] Error extracting features for flow {flow.id}: {e}")
            print("-" * 80)


    except KeyboardInterrupt:
        print("\n[*] Capture stopped by user (Ctrl+C).")
    except PermissionError:
        print(f"[-] Error: Insufficient permissions to capture on interface {args.interface}.", file=sys.stderr)
        print("[-] Try running the script with sudo: sudo python your_script_name.py ...", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
         if "No such device" in str(e) or "Cannot open interface" in str(e):
             print(f"[-] Error: Network interface '{args.interface}' not found or could not be opened.", file=sys.stderr)
             print("[-] Please check the interface name and ensure it is active.", file=sys.stderr)
         else:
             print(f"[-] An OS error occurred during capture setup: {e}", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"\n[-] An unexpected error occurred during live capture: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        print("[*] Exiting.")


if __name__ == "__main__":
    main()
