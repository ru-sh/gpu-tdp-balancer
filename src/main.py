import argparse
import signal
import sys
import logging
from gpu_tdp_balancer.balancer import GpuTdpBalancer, NVMLError

# Global variable to hold the balancer instance for signal handling
balancer_instance = None

def signal_handler(sig, frame):
    """Handles termination signals gracefully."""
    print("\nTermination signal received. Shutting down...")
    if balancer_instance:
        # Call the balancer's shutdown method directly.
        # The run loop's finally block might also call it, but this ensures it happens
        # even if the signal is caught outside the main loop's try block somehow.
        balancer_instance.shutdown()
    sys.exit(0)

def main():
    global balancer_instance

    parser = argparse.ArgumentParser(description="GPU TDP Balancer")
    parser.add_argument(
        "--max-total-tdp", type=int, default=800,
        help="Maximum total TDP allowed across all GPUs (Watts). Default: 800W"
    )
    parser.add_argument(
        "--min-gpu-tdp", type=int, default=100,
        help="Minimum TDP limit for any single GPU (Watts). Default: 100W (NVML min value)"
    )
    parser.add_argument(
        "--active-level", type=int, default=20,
        help="GPU utilization (%) threshold to consider a GPU 'active'. Default: 20%%"
    )
    parser.add_argument(
        "--passive-level", type=int, default=10,
        help="GPU utilization (%) threshold below which all GPUs must be for 'passive' state. Default: 10%%"
    )
    parser.add_argument(
        "--interval", type=float, default=1.0,
        help="Update interval in seconds. Default: 1.0s"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose (DEBUG level) logging."
    )

    args = parser.parse_args()

    # Setup logging level
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')


    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler) # Handle kill/system shutdown

    try:
        balancer_instance = GpuTdpBalancer(
            gpu_max_tdp_total_w=args.max_total_tdp,
            gpu_min_tdp_per_gpu_w=args.min_gpu_tdp,
            gpu_active_level_percent=args.active_level,
            gpu_passive_level_percent=args.passive_level,
            update_interval_sec=args.interval
        )
        # The run method contains the main loop
        balancer_instance.run()

    except NVMLError as e:
        logging.error(f"NVML Error during initialization or execution: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logging.error(f"Runtime Error: {e}")
        # Ensure NVML is shutdown if initialization partially completed
        if balancer_instance:
            balancer_instance.shutdown()
        sys.exit(1)
    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}") # Use exception for stack trace
        if balancer_instance:
            balancer_instance.shutdown()
        sys.exit(1)

if __name__ == "__main__":
    # Must be run as root/administrator to set power limits
    import os
    if os.name == 'posix' and os.geteuid() != 0:
        logging.warning("Script not running as root. Setting TDP limits will likely fail.")
    elif os.name == 'nt':
        # Check for admin rights on Windows (requires pywin32)
        try:
            import ctypes
            if not ctypes.windll.shell32.IsUserAnAdmin():
                logging.warning("Script not running as Administrator. Setting TDP limits will likely fail.")
        except ImportError:
             logging.warning("Cannot check for Administrator rights on Windows (pywin32 not found). Setting TDP limits might fail.")
        except Exception as e:
             logging.warning(f"Error checking for Administrator rights: {e}. Setting TDP limits might fail.")


    main()