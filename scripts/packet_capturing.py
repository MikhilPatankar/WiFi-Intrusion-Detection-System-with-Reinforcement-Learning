#!/usr/bin/env python3

import os
import sys
import subprocess
import time
import argparse
import signal
import threading
from datetime import datetime
from shutil import which # Used to check if airmon-ng exists

# --- Scapy Import ---
# Hide verbose scapy messages
import logging
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
logging.getLogger("scapy.loading").setLevel(logging.ERROR)
try:
    from scapy.all import sniff, PcapWriter, Packet
except ImportError:
    print("[-] Scapy is not installed. Please install it: pip install scapy")
    sys.exit(1)

# --- Global Variables ---
monitor_interface = None
original_interface = None
capture_process = None
stop_event = threading.Event() # Used for stopping threads gracefully
channel_hopper_thread = None
pcap_writer = None
packet_count = 0
start_time = None
output_filename = ""

# --- Configuration ---
DEFAULT_CHANNELS_24GHZ = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
DEFAULT_CHANNELS_5GHZ = [36, 40, 44, 48, 52, 56, 60, 64, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 149, 153, 157, 161, 165]
DEFAULT_HOP_INTERVAL = 2 # seconds

# --- Helper Functions ---

def check_root():
    """Checks if the script is running as root."""
    if os.geteuid() != 0:
        print("[-] This script requires root privileges. Please run with sudo.")
        sys.exit(1)
    print("[+] Root privileges verified.")

def run_command(command, ignore_errors=False):
    """Executes a shell command."""
    print(f"[*] Executing: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=not ignore_errors, capture_output=True, text=True)
        if result.stdout:
            print(f"[+] STDOUT: {result.stdout.strip()}")
        if result.stderr:
            print(f"[!] STDERR: {result.stderr.strip()}")
        return result.returncode == 0, result.stdout, result.stderr
    except FileNotFoundError:
        print(f"[-] Error: Command not found: {command[0]}")
        return False, "", f"Command not found: {command[0]}"
    except subprocess.CalledProcessError as e:
        print(f"[-] Error executing command: {' '.join(command)}")
        print(f"[-] Return Code: {e.returncode}")
        print(f"[-] Output: {e.stdout.strip()}")
        print(f"[-] Error Output: {e.stderr.strip()}")
        return False, e.stdout, e.stderr
    except Exception as e:
        print(f"[-] An unexpected error occurred while running command: {e}")
        return False, "", str(e)

def find_interface(iface_name):
    """Checks if a network interface exists."""
    try:
        # Use 'ip link show' which is standard
        success, stdout, _ = run_command(['ip', 'link', 'show', iface_name], ignore_errors=True)
        # Check if stdout contains the interface name followed by ':'
        return success and f"{iface_name}:" in stdout.split('\n')[0]
    except Exception as e:
        print(f"[-] Error checking interface {iface_name}: {e}")
        return False

def enable_monitor_mode(iface):
    """Puts the specified wireless interface into monitor mode."""
    global monitor_interface, original_interface
    original_interface = iface
    print(f"[*] Attempting to enable monitor mode for {iface}...")

    if not find_interface(iface):
        print(f"[-] Error: Interface {iface} not found.")
        return None

    # Try airmon-ng first (preferred)
    airmon_path = which('airmon-ng')
    if airmon_path:
        print("[*] Using airmon-ng...")
        # Check for conflicting processes
        run_command([airmon_path, 'check', 'kill'], ignore_errors=True)
        # Start monitor mode
        success, stdout, stderr = run_command([airmon_path, 'start', iface])
        if success:
            # airmon-ng often creates a new interface (e.g., wlan0mon)
            # Try to parse the new interface name from stdout
            new_iface = None
            lines = stdout.splitlines()
            for line in lines:
                 # Adjust this parsing based on your airmon-ng version output
                if "monitor mode vif enabled" in line or "monitor mode enabled" in line:
                    parts = line.split()
                    # Look for common patterns like '(monitor mode vif enabled for [phyX]wlan0 on [phyX]wlan0mon)'
                    # or 'monitor mode enabled on mon0'
                    for part in parts:
                        if part.endswith('mon') or 'mon' in part and iface not in part:
                             # Basic check to avoid picking the original iface if it had 'mon'
                            if find_interface(part.strip('()[]')):
                                new_iface = part.strip('()[]')
                                break
                    if new_iface: break
            
            if new_iface:
                 print(f"[+] Monitor mode enabled on interface: {new_iface}")
                 monitor_interface = new_iface
                 # Verify it's actually in monitor mode
                 success_iw, _, _ = run_command(['iwconfig', new_iface], ignore_errors=True)
                 if not success_iw: # Sometimes takes a moment
                    time.sleep(2)
                    success_iw, _, _ = run_command(['iwconfig', new_iface], ignore_errors=True)

                 if success_iw:
                    return new_iface
                 else:
                     print(f"[-] Failed to verify monitor mode on {new_iface} with iwconfig. Problems might occur.")
                     # Fallback or proceed with caution
                     return new_iface # Return anyway, might work with ip/iw

            else:
                # Airmon ran but couldn't find new iface name? Maybe it modified in-place?
                print("[!] Warning: airmon-ng might have enabled monitor mode on the original interface name or failed silently.")
                # Check original interface name
                success, stdout, _ = run_command(['iwconfig', iface], ignore_errors=True)
                if success and "Mode:Monitor" in stdout:
                     print(f"[+] Monitor mode seems enabled on {iface}")
                     monitor_interface = iface
                     return iface
                else:
                    print("[-] airmon-ng did not seem to enable monitor mode successfully. Falling back to ip/iwconfig.")
                    # Proceed to fallback

        else:
            print("[-] airmon-ng failed. Falling back to ip/iwconfig...")
    else:
        print("[*] airmon-ng not found. Using ip/iwconfig...")

    # Fallback: Use ip link and iwconfig
    print(f"[*] Using ip/iwconfig to enable monitor mode on {iface}...")
    commands = [
        ['ip', 'link', 'set', iface, 'down'],
        ['iwconfig', iface, 'mode', 'monitor'],
        ['ip', 'link', 'set', iface, 'up']
    ]
    all_success = True
    for cmd in commands:
        success, _, stderr = run_command(cmd)
        if not success:
            # Sometimes 'Network is down' is expected after setting down, ignore specific error
            if not ('Network is down' in stderr and cmd[1:4] == ['link', 'set', iface] and cmd[4] == 'up'):
                 print(f"[-] Failed command: {' '.join(cmd)}")
                 all_success = False
                 # Attempt to bring interface back up if we downed it
                 if cmd[1:4] == ['link', 'set', iface] and cmd[4] == 'down':
                     run_command(['ip', 'link', 'set', iface, 'up'], ignore_errors=True)
                 # Attempt to reset mode if possible
                 run_command(['iwconfig', iface, 'mode', 'managed'], ignore_errors=True)
                 return None # Failed to enable monitor mode

    if all_success:
         # Verify mode
         time.sleep(1) # Give interface time to settle
         success, stdout, _ = run_command(['iwconfig', iface], ignore_errors=True)
         if success and "Mode:Monitor" in stdout:
             print(f"[+] Monitor mode enabled on interface: {iface}")
             monitor_interface = iface
             return iface
         else:
            print(f"[-] Failed to verify monitor mode on {iface} using iwconfig.")
            # Attempt cleanup
            run_command(['ip', 'link', 'set', iface, 'down'], ignore_errors=True)
            run_command(['iwconfig', iface, 'mode', 'managed'], ignore_errors=True)
            run_command(['ip', 'link', 'set', iface, 'up'], ignore_errors=True)
            return None

    return None # Failed if we reached here


def disable_monitor_mode():
    """Disables monitor mode and restores the original interface state."""
    global monitor_interface, original_interface
    if not monitor_interface or not original_interface:
        print("[*] No monitor mode interface active or original interface unknown.")
        return

    print(f"[*] Disabling monitor mode for {monitor_interface}...")

    # Try airmon-ng first if it was likely used (monitor name != original name or includes 'mon')
    airmon_path = which('airmon-ng')
    if airmon_path and (monitor_interface != original_interface or 'mon' in monitor_interface):
        print("[*] Using airmon-ng to stop...")
        success, _, _ = run_command([airmon_path, 'stop', monitor_interface], ignore_errors=True)
        if success:
             print(f"[+] Monitor mode likely stopped by airmon-ng. Interface should revert to {original_interface}.")
             # Check if original interface is up and in managed mode
             time.sleep(2) # Give it time
             if find_interface(original_interface):
                run_command(['ip', 'link', 'set', original_interface, 'up'], ignore_errors=True) # Ensure it's up
                run_command(['iwconfig', original_interface, 'mode', 'managed'], ignore_errors=True) # Ensure mode is managed
                print(f"[+] Interface {original_interface} seems restored.")
             else:
                print(f"[!] Warning: Original interface {original_interface} not found after airmon-ng stop.")
             monitor_interface = None
             return
        else:
            print("[-] airmon-ng stop failed or interface wasn't managed by it. Falling back to ip/iwconfig.")
            # Fall through to ip/iwconfig method as a cleanup attempt

    # Fallback or primary method if airmon wasn't used
    print(f"[*] Using ip/iwconfig to disable monitor mode for {monitor_interface} and restore {original_interface}...")
    commands = [
        ['ip', 'link', 'set', monitor_interface, 'down'],
        ['iwconfig', monitor_interface, 'mode', 'managed'],
    ]
    # If monitor interface is different from original, try setting original up too
    if monitor_interface != original_interface:
         commands.append(['ip', 'link', 'set', original_interface, 'up'])
         commands.append(['iwconfig', original_interface, 'mode', 'managed']) # Set mode just in case
    else:
         # If same interface, just bring it up after setting mode
         commands.append(['ip', 'link', 'set', monitor_interface, 'up'])

    for cmd in commands:
        run_command(cmd, ignore_errors=True) # Ignore errors during cleanup

    # Verify original interface state (best effort)
    if find_interface(original_interface):
        success, stdout, _ = run_command(['iwconfig', original_interface], ignore_errors=True)
        if success and ("Mode:Managed" in stdout or "Mode:Auto" in stdout): # Managed or Auto is usually the default
             print(f"[+] Interface {original_interface} restored to Managed/Auto mode.")
        else:
             print(f"[!] Warning: Could not fully verify {original_interface} state after cleanup.")
    else:
        print(f"[!] Warning: Original interface {original_interface} might not be available after cleanup.")

    monitor_interface = None


def set_channel(mon_iface, channel):
    """Sets the channel for the monitor interface."""
    print(f"[*] Setting channel to {channel} for {mon_iface}...")
    # Use iw first if available, as it's newer than iwconfig
    iw_path = which('iw')
    if iw_path:
        success, _, stderr = run_command([iw_path, 'dev', mon_iface, 'set', 'channel', str(channel)])
        if success:
            print(f"[+] Channel set to {channel} using iw.")
            return True
        else:
            print(f"[-] Failed to set channel using iw: {stderr.strip()}. Trying iwconfig...")
            # Fall through to iwconfig

    # Fallback to iwconfig
    success, _, stderr = run_command(['iwconfig', mon_iface, 'channel', str(channel)])
    if success:
        print(f"[+] Channel set to {channel} using iwconfig.")
        return True
    else:
        print(f"[-] Failed to set channel using iwconfig: {stderr.strip()}")
        return False

def channel_hopper(mon_iface, channels, interval):
    """Thread function to hop channels."""
    global stop_event
    print(f"[*] Starting channel hopper thread for channels: {channels} (Interval: {interval}s)")
    while not stop_event.is_set():
        for channel in channels:
            if stop_event.is_set():
                break
            set_channel(mon_iface, channel)
            # Sleep for the interval, but check stop_event frequently
            for _ in range(interval * 10): # Check every 100ms
                if stop_event.is_set():
                    break
                time.sleep(0.1)
    print("[*] Channel hopper thread stopped.")

def packet_handler(packet: Packet):
    """Callback function for each captured packet."""
    global pcap_writer, packet_count
    if pcap_writer:
        try:
            pcap_writer.write(packet)
            packet_count += 1
            # Print status periodically (e.g., every 100 packets)
            if packet_count % 100 == 0:
                 current_time = time.time()
                 elapsed = current_time - start_time
                 pps = packet_count / elapsed if elapsed > 0 else 0
                 print(f"\r[*] Captured: {packet_count} packets ({pps:.2f} pps)", end="")

        except Exception as e:
            print(f"\n[-] Error writing packet to PCAP: {e}")
            # Optionally stop capture on write error, or just log it
            # stop_capture(signal.SIGTERM, None) # Example: Force stop

def stop_capture(signum, frame):
    """Signal handler to stop the capture gracefully."""
    global stop_event, channel_hopper_thread, pcap_writer
    print("\n[*] Signal received, stopping capture...")
    stop_event.set() # Signal threads and sniff loop to stop

    # Wait briefly for channel hopper thread to exit
    if channel_hopper_thread and channel_hopper_thread.is_alive():
        print("[*] Waiting for channel hopper to finish...")
        channel_hopper_thread.join(timeout=DEFAULT_HOP_INTERVAL + 1) # Wait slightly longer than hop interval
        if channel_hopper_thread.is_alive():
             print("[!] Warning: Channel hopper thread did not exit cleanly.")

    # Close the pcap writer
    if pcap_writer:
        try:
            print("[*] Closing PCAP file...")
            pcap_writer.close()
            pcap_writer = None # Ensure it's cleared
            print(f"[+] PCAP file saved: {output_filename}")
            print(f"[+] Total packets captured: {packet_count}")
        except Exception as e:
            print(f"[-] Error closing PCAP writer: {e}")

    # Disable monitor mode - use a finally block in main for robustness
    # disable_monitor_mode() # Moved to finally block

    print("[*] Exiting script.")
    # sys.exit(0) # Exit is handled by the main thread finishing after sniff stops


def main():
    global monitor_interface, original_interface, stop_event, channel_hopper_thread
    global pcap_writer, packet_count, start_time, output_filename

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Advanced Wi-Fi Monitor Mode Packet Capture Script")
    parser.add_argument("-i", "--interface", required=True, help="Wireless interface to use (e.g., wlan0)")
    parser.add_argument("-d", "--duration", type=int, default=0, help="Capture duration in seconds (0 for indefinite)")
    parser.add_argument("-o", "--output", default="capture", help="Base name for the output PCAP file (timestamp will be added)")
    parser.add_argument("--channel", type=int, help="Specific channel to monitor (e.g., 6). Cannot be used with --hop.")
    parser.add_argument("--hop", action="store_true", help="Enable channel hopping across default 2.4GHz channels.")
    parser.add_argument("--hop-channels", default=None, help=f"Comma-separated list of channels to hop (e.g., '1,6,11'). Overrides --hop default. Default 2.4GHz: {DEFAULT_CHANNELS_24GHZ}")
    parser.add_argument("--hop-interval", type=int, default=DEFAULT_HOP_INTERVAL, help=f"Time in seconds to spend on each channel during hopping (default: {DEFAULT_HOP_INTERVAL}s)")
    parser.add_argument("--band", choices=['2.4', '5', 'all'], default='2.4', help="Specify band for default channel hopping (default: 2.4). 'all' uses common channels from both.")


    args = parser.parse_args()

    if args.channel and (args.hop or args.hop_channels):
        print("[-] Error: Cannot specify a single channel (--channel) and enable channel hopping (--hop or --hop-channels) simultaneously.")
        sys.exit(1)

    # --- Initial Checks ---
    check_root()

    # --- Setup Signal Handling ---
    signal.signal(signal.SIGINT, stop_capture)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, stop_capture) # Handle termination signals

    # --- Main Logic ---
    try:
        # Enable Monitor Mode
        monitor_interface = enable_monitor_mode(args.interface)
        if not monitor_interface:
            print("[-] Failed to enable monitor mode. Exiting.")
            sys.exit(1)

        # Prepare PCAP file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{args.output}_{monitor_interface}_{timestamp}.pcap"
        print(f"[*] Preparing PCAP file: {output_filename}")
        try:
             # Use PcapWriter for potentially large captures
             pcap_writer = PcapWriter(output_filename, append=False, sync=True) # sync=True writes more often
        except Exception as e:
             print(f"[-] Error opening PcapWriter for {output_filename}: {e}")
             raise # Re-raise to trigger finally block for cleanup

        # Configure Channels and Start Hopping if necessary
        channels_to_use = []
        if args.hop or args.hop_channels:
            if args.hop_channels:
                try:
                    channels_to_use = [int(ch.strip()) for ch in args.hop_channels.split(',')]
                except ValueError:
                    print(f"[-] Invalid channel list format: {args.hop_channels}")
                    raise ValueError("Invalid channels") # Trigger finally
            else:
                # Default hopping based on band
                if args.band == '2.4':
                    channels_to_use = DEFAULT_CHANNELS_24GHZ
                elif args.band == '5':
                    channels_to_use = DEFAULT_CHANNELS_5GHZ
                elif args.band == 'all':
                     channels_to_use = sorted(list(set(DEFAULT_CHANNELS_24GHZ + DEFAULT_CHANNELS_5GHZ)))
            
            if not channels_to_use:
                 print("[-] No channels selected for hopping. Exiting.")
                 raise ValueError("No channels for hopping") # Trigger finally

            # Start channel hopping in a separate thread
            channel_hopper_thread = threading.Thread(
                target=channel_hopper,
                args=(monitor_interface, channels_to_use, args.hop_interval),
                daemon=True # Allows main thread to exit even if this thread hangs (though we join later)
            )
            channel_hopper_thread.start()
            # Give the hopper a moment to set the first channel
            time.sleep(0.5)

        elif args.channel:
            # Set a specific channel
            if not set_channel(monitor_interface, args.channel):
                print(f"[-] Failed to set channel {args.channel}. Exiting.")
                raise RuntimeError("Failed to set channel") # Trigger finally
            channels_to_use = [args.channel] # For logging purposes
        else:
            # No channel specified, capture on current/default channel
            print("[!] Warning: No channel specified and hopping disabled. Capturing on current default channel.")
            # Optionally try reading current channel
            success, stdout, _ = run_command(['iwconfig', monitor_interface], ignore_errors=True)
            current_channel = "Unknown"
            if success:
                for line in stdout.splitlines():
                    if "Channel:" in line or "Frequency:" in line: # iwconfig format varies
                        parts = line.split()
                        for part in parts:
                             if part.startswith("Channel="): current_channel = part.split('=')[1]; break
                             if part.startswith("Channel:"): current_channel = part.split(':')[1]; break
                             # Could also parse Frequency and map to channel
                        if current_channel != "Unknown": break
            print(f"[*] Current channel appears to be: {current_channel}")


        # Start Sniffing
        print(f"[*] Starting packet capture on {monitor_interface}...")
        if args.duration > 0:
            print(f"[*] Capture duration: {args.duration} seconds")
        else:
            print("[*] Capture duration: Indefinite (Press Ctrl+C to stop)")

        packet_count = 0
        start_time = time.time()

        # Use stop_filter for cleaner stopping with Ctrl+C or timeout
        sniff(
            iface=monitor_interface,
            prn=packet_handler,       # Process packets using our handler
            store=False,              # Don't store packets in memory
            stop_filter=lambda x: stop_event.is_set(), # Check stop event
            timeout=args.duration if args.duration > 0 else None # Scapy timeout
        )

        # If sniff finished due to timeout (not Ctrl+C), call stop_capture manually
        if not stop_event.is_set() and args.duration > 0:
             print("\n[*] Capture duration reached.")
             stop_capture(signal.SIGALRM, None) # Simulate a signal to trigger cleanup

    except (KeyboardInterrupt, SystemExit):
        # This block might not be strictly needed if signal handler works perfectly
        # but provides an extra layer of safety.
        print("\n[*] Interrupted or exiting...")
        if not stop_event.is_set():
            stop_capture(signal.SIGINT, None) # Ensure cleanup runs if signal wasn't caught somehow

    except Exception as e:
        print(f"\n[-] An unexpected error occurred in main execution: {e}")
        # Ensure stop_event is set to signal potential running threads
        stop_event.set()
        # Cleanup is handled in the finally block

    finally:
        # --- Cleanup ---
        print("[*] Entering final cleanup phase...")

        # Ensure stop event is set if not already
        stop_event.set()

        # Ensure channel hopper is stopped and joined
        if channel_hopper_thread and channel_hopper_thread.is_alive():
            print("[*] Ensuring channel hopper thread is stopped...")
            channel_hopper_thread.join(timeout=2) # Short timeout here
            if channel_hopper_thread.is_alive():
                 print("[!] Warning: Channel hopper thread still alive during final cleanup.")

        # Ensure pcap writer is closed
        if pcap_writer:
             print("[*] Ensuring PCAP writer is closed...")
             try:
                pcap_writer.close()
                print(f"[+] Final check: PCAP file {output_filename} closed.")
                print(f"[+] Total packets captured: {packet_count}")
             except Exception as e:
                print(f"[-] Error during final PCAP writer close: {e}")

        # Disable monitor mode (most critical cleanup)
        if monitor_interface:
            disable_monitor_mode()
        else:
            # If monitor_interface wasn't set but original_interface was,
            # attempt a cleanup on original_interface just in case airmon failed partially
            if original_interface:
                 print(f"[*] Attempting cleanup on original interface {original_interface} as monitor mode setup might have failed.")
                 run_command(['ip', 'link', 'set', original_interface, 'down'], ignore_errors=True)
                 run_command(['iwconfig', original_interface, 'mode', 'managed'], ignore_errors=True)
                 run_command(['ip', 'link', 'set', original_interface, 'up'], ignore_errors=True)


        print("[+] Script finished.")


if __name__ == "__main__":
    main()