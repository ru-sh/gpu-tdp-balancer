import time
import logging
from pynvml import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GpuTdpBalancer:
    """
    Dynamically balances TDP limits across multiple NVIDIA GPUs based on load.
    """
    def __init__(self,
                 gpu_max_tdp_total_w: int = 800,
                 gpu_min_tdp_per_gpu_w: int = 100,
                 gpu_active_level_percent: int = 20,
                 gpu_passive_level_percent: int = 10,
                 update_interval_sec: float = 1.0):
        """
        Initializes the GpuTdpBalancer.

        Args:
            gpu_max_tdp_total_w: Maximum total TDP allowed across all GPUs (Watts).
            gpu_min_tdp_per_gpu_w: Minimum TDP limit for any single GPU (Watts).
            gpu_active_level_percent: GPU utilization (%) threshold to consider a GPU "active".
            gpu_passive_level_percent: GPU utilization (%) threshold below which all GPUs must be
                                       to enter the "passive" state.
            update_interval_sec: How often to check and adjust TDP (seconds).
        """
        self.gpu_max_tdp_total_mw = gpu_max_tdp_total_w * 1000
        self.gpu_min_tdp_per_gpu_mw = gpu_min_tdp_per_gpu_w * 1000
        self.gpu_active_level = gpu_active_level_percent
        self.gpu_passive_level = gpu_passive_level_percent
        self.update_interval = update_interval_sec

        self.handles = []
        self.num_gpus = 0
        self.gpu_max_tdp_limits_mw = {} # Store individual GPU max TDP limits {index: max_tdp_mw}
        self.last_tdp_limits_mw = {} # Store last applied TDP limits {index: tdp_mw}

        logging.info("Initializing GpuTdpBalancer...")
        self._initialize_nvml()
        self._initial_tdp_setup()
        logging.info("Initialization complete.")

    def _initialize_nvml(self):
        """Initializes the NVML library and retrieves GPU handles."""
        try:
            nvmlInit()
            self.num_gpus = nvmlDeviceGetCount()
            if self.num_gpus == 0:
                logging.error("No NVIDIA GPUs found.")
                raise RuntimeError("No NVIDIA GPUs found.")

            logging.info(f"Found {self.num_gpus} GPUs.")

            for i in range(self.num_gpus):
                handle = nvmlDeviceGetHandleByIndex(i)
                self.handles.append(handle)
                # Enable persistence mode if possible (helps maintain settings) - requires root
                try:
                    nvmlDeviceSetPersistenceMode(handle, NVML_FEATURE_ENABLED)
                    logging.debug(f"Set persistence mode for GPU {i}.")
                except NVMLError as e:
                     if e.value == NVML_ERROR_NOT_SUPPORTED:
                         logging.warning(f"Persistence mode not supported for GPU {i}.")
                     elif e.value == NVML_ERROR_NO_PERMISSION:
                         logging.warning(f"No permission to set persistence mode for GPU {i}. Run as root/admin.")
                     else:
                         logging.warning(f"Could not set persistence mode for GPU {i}: {e}")

                 # Get and store the maximum possible power limit for this GPU
                try:
                    max_limit_mw = nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] # Max limit is the second element
                    self.gpu_max_tdp_limits_mw[i] = max_limit_mw
                    logging.info(f"GPU {i}: Max TDP Limit = {max_limit_mw / 1000:.1f} W")
                except NVMLError as e:
                     # Some cards might not support constraints query
                     if e.value == NVML_ERROR_NOT_SUPPORTED:
                        logging.warning(f"Could not query max power limit for GPU {i} (Not Supported). Falling back to default limit.")
                        try:
                            # Use the default limit as a fallback max
                            default_limit_mw = nvmlDeviceGetPowerManagementDefaultLimit(handle)
                            self.gpu_max_tdp_limits_mw[i] = default_limit_mw
                            logging.info(f"GPU {i}: Using Default TDP Limit as Max = {default_limit_mw / 1000:.1f} W")
                        except NVMLError as e_inner:
                             logging.error(f"Could not query default power limit for GPU {i}: {e_inner}. Cannot manage this GPU.")
                             raise RuntimeError(f"Failed to get power limits for GPU {i}") from e_inner
                     else:
                        logging.error(f"Could not query max power limit constraints for GPU {i}: {e}. Cannot manage this GPU.")
                        raise RuntimeError(f"Failed to get power limit constraints for GPU {i}") from e

            # Sanity check total max TDP vs sum of individual max TDPs
            total_individual_max_tdp_mw = sum(self.gpu_max_tdp_limits_mw.values())
            if self.gpu_max_tdp_total_mw > total_individual_max_tdp_mw:
                logging.warning(f"Configured GPU_MAX_TDP_TOTAL ({self.gpu_max_tdp_total_mw / 1000:.1f}W) is higher than the sum of individual GPU max TDPs ({total_individual_max_tdp_mw / 1000:.1f}W). Clamping to the sum.")
                self.gpu_max_tdp_total_mw = total_individual_max_tdp_mw

            # Sanity check min TDP vs individual max TDPs
            for i in range(self.num_gpus):
                if self.gpu_min_tdp_per_gpu_mw > self.gpu_max_tdp_limits_mw[i]:
                     logging.warning(f"Configured GPU_MIN_TDP_PER_GPU ({self.gpu_min_tdp_per_gpu_mw/1000:.1f}W) is higher than GPU {i}'s max TDP ({self.gpu_max_tdp_limits_mw[i]/1000:.1f}W). Clamping min TDP for this GPU to its max TDP.")
                     # This doesn't change the global minimum, but logic later will respect the GPU's max

        except NVMLError as error:
            logging.error(f"Failed to initialize NVML: {error}")
            raise RuntimeError("Failed to initialize NVML") from error

    def _initial_tdp_setup(self):
        """Sets the initial TDP limit evenly across all GPUs."""
        if self.num_gpus == 0:
            return

        logging.info("Performing initial TDP setup...")
        initial_tdp_per_gpu_mw = self.gpu_max_tdp_total_mw // self.num_gpus
        new_limits = {}

        # Calculate initial target, clamping between min_tdp and individual GPU max
        for i in range(self.num_gpus):
            target_tdp = max(self.gpu_min_tdp_per_gpu_mw, initial_tdp_per_gpu_mw)
            target_tdp = min(target_tdp, self.gpu_max_tdp_limits_mw[i])
            new_limits[i] = target_tdp

        # Adjust if initial allocation exceeded total budget due to min/max clamping
        current_total_mw = sum(new_limits.values())
        excess_mw = current_total_mw - self.gpu_max_tdp_total_mw

        if excess_mw > 0:
            logging.warning(f"Initial TDP clamping resulted in exceeding total budget by {excess_mw / 1000:.1f}W. Reducing limits.")
            # Reduce from GPUs that are currently above the minimum, starting with highest
            sorted_indices = sorted(new_limits, key=new_limits.get, reverse=True)
            for i in sorted_indices:
                if excess_mw <= 0:
                    break
                reduction = min(excess_mw, new_limits[i] - self.gpu_min_tdp_per_gpu_mw)
                if reduction > 0:
                    new_limits[i] -= reduction
                    excess_mw -= reduction

            # If still excess (means min limits sum > total budget), log error, but proceed.
            # The _set_tdp_limits logic should ultimately respect the total budget if implemented robustly,
            # or the user needs to adjust their config.
            if excess_mw > 0:
                 logging.error(f"Could not meet total TDP budget even after reducing to minimums where possible. Remaining excess: {excess_mw / 1000:.1f}W. Check configuration (GPU_MAX_TDP_TOTAL vs GPU_MIN_TDP_PER_GPU * num_gpus). Applying best effort.")


        logging.info(f"Setting initial TDP limits (mW): { {i: l/1000 for i, l in new_limits.items()} }")
        self._set_tdp_limits(new_limits)


    def _set_tdp_limits(self, target_limits_mw: dict):
        """
        Applies the target TDP limits (in mW) to the GPUs.

        Args:
            target_limits_mw: A dictionary {gpu_index: tdp_limit_mw}.
        """
        applied_something = False
        for i, handle in enumerate(self.handles):
            if i not in target_limits_mw:
                logging.warning(f"No target TDP provided for GPU {i}. Skipping.")
                continue

            # Clamp again just to be safe, respecting individual max and configured min
            limit_to_set = max(self.gpu_min_tdp_per_gpu_mw, target_limits_mw[i])
            limit_to_set = min(limit_to_set, self.gpu_max_tdp_limits_mw[i])
            limit_to_set = int(limit_to_set) # NVML expects integer mW

            # Only set if the limit has actually changed
            if i not in self.last_tdp_limits_mw or self.last_tdp_limits_mw[i] != limit_to_set:
                try:
                    nvmlDeviceSetPowerManagementLimit(handle, limit_to_set)
                    self.last_tdp_limits_mw[i] = limit_to_set
                    logging.debug(f"Set GPU {i} TDP limit to {limit_to_set / 1000:.1f} W")
                    applied_something = True
                except NVMLError as error:
                    # Handle common errors
                    if error.value == NVML_ERROR_NO_PERMISSION:
                        logging.error(f"Permission denied to set power limit for GPU {i}. Run script as root/administrator.")
                    elif error.value == NVML_ERROR_NOT_SUPPORTED:
                        logging.error(f"Power management limit is not supported for GPU {i}.")
                    else:
                        logging.error(f"Failed to set power limit for GPU {i}: {error}")
                    # Stop trying if we hit a permission error, likely applies to all
                    if error.value == NVML_ERROR_NO_PERMISSION:
                       raise RuntimeError("Permission denied to set power limits. Exiting.") from error
            else:
                logging.debug(f"GPU {i} TDP limit already at {limit_to_set / 1000:.1f} W. No change needed.")

        if applied_something:
            logging.info(f"Applied new TDP limits (W): { {k: v/1000 for k, v in self.last_tdp_limits_mw.items()} }")


    def get_gpu_status(self) -> list[tuple[int, int]]:
        """
        Gets the current utilization status for all GPUs.

        Returns:
            A list of tuples: [(gpu_index, utilization_percent), ...]
        """
        status = []
        for i, handle in enumerate(self.handles):
            try:
                util = nvmlDeviceGetUtilizationRates(handle)
                status.append((i, util.gpu)) # util.gpu is the GPU core utilization
            except NVMLError as error:
                logging.error(f"Failed to get utilization for GPU {i}: {error}")
                status.append((i, 0)) # Assume 0 utilization on error
        return status

    def run(self):
        """Starts the main balancing loop."""
        logging.info("Starting TDP balancing loop...")
        try:
            while True:
                gpu_statuses = self.get_gpu_status() # List of (index, utilization)
                logging.debug(f"Current GPU utilization: {gpu_statuses}")

                active_gpus = []
                all_passive = True
                for index, util in gpu_statuses:
                    if util >= self.gpu_active_level:
                        active_gpus.append((index, util))
                        all_passive = False # At least one is active
                    elif util >= self.gpu_passive_level:
                         all_passive = False # This one isn't passive enough for the "all passive" state

                new_limits_mw = {}

                if all_passive:
                    # Scenario: All GPUs are below passive threshold - distribute evenly
                    logging.debug("All GPUs are passive. Distributing TDP evenly.")
                    base_tdp_per_gpu_mw = self.gpu_max_tdp_total_mw // self.num_gpus
                    current_total_mw = 0
                    temp_limits = {}

                    # Initial pass: set base TDP, clamped by individual min/max
                    for i in range(self.num_gpus):
                        limit = max(self.gpu_min_tdp_per_gpu_mw, base_tdp_per_gpu_mw)
                        limit = min(limit, self.gpu_max_tdp_limits_mw[i])
                        temp_limits[i] = limit
                        current_total_mw += limit

                    # Adjust if total exceeds budget (due to clamping)
                    excess_mw = current_total_mw - self.gpu_max_tdp_total_mw
                    if excess_mw > 0:
                        logging.debug(f"Passive distribution exceeded budget by {excess_mw/1000:.1f}W. Reducing.")
                        # Reduce from GPUs above minimum, proportionally could be complex,
                        # simpler: reduce from highest clamped values first
                        sorted_indices = sorted(temp_limits, key=temp_limits.get, reverse=True)
                        for i in sorted_indices:
                            if excess_mw <= 0: break
                            reduction = min(excess_mw, temp_limits[i] - self.gpu_min_tdp_per_gpu_mw)
                            if reduction > 0:
                                temp_limits[i] -= reduction
                                excess_mw -= reduction

                    new_limits_mw = temp_limits # Use the potentially adjusted limits

                else:
                    # Scenario: Some GPUs are active - prioritize active GPUs
                    logging.debug(f"Active GPUs detected: {[g[0] for g in active_gpus]}. Prioritizing TDP.")
                    active_gpus.sort(key=lambda x: x[1], reverse=True) # Sort by utilization descending

                    # Start by assigning minimum TDP to all GPUs
                    for i in range(self.num_gpus):
                        # Ensure min TDP doesn't exceed the GPU's max capability
                        new_limits_mw[i] = min(self.gpu_min_tdp_per_gpu_mw, self.gpu_max_tdp_limits_mw[i])

                    current_allocated_mw = sum(new_limits_mw.values())
                    remaining_budget_mw = self.gpu_max_tdp_total_mw - current_allocated_mw

                    if remaining_budget_mw < 0:
                        # This implies sum of min TDPs > total budget. Log error.
                         logging.error(f"Sum of minimum TDPs ({current_allocated_mw/1000:.1f}W) exceeds total budget ({self.gpu_max_tdp_total_mw/1000:.1f}W). Cannot allocate more. Check config.")
                         remaining_budget_mw = 0 # Cannot allocate more

                    logging.debug(f"Initial min allocation: {current_allocated_mw / 1000:.1f} W. Remaining budget: {remaining_budget_mw / 1000:.1f} W.")

                    # Distribute remaining budget to active GPUs in order of utilization
                    for index, util in active_gpus:
                        if remaining_budget_mw <= 0:
                            break # No more budget to distribute

                        current_limit = new_limits_mw[index]
                        max_possible_for_gpu = self.gpu_max_tdp_limits_mw[index]
                        potential_increase = max_possible_for_gpu - current_limit

                        # How much can we actually give this GPU from the remaining budget?
                        increase_amount = min(remaining_budget_mw, potential_increase)

                        if increase_amount > 0:
                            new_limits_mw[index] += increase_amount
                            remaining_budget_mw -= increase_amount
                            logging.debug(f"Allocated {increase_amount / 1000:.1f} W extra to active GPU {index} (Util: {util}%). Remaining budget: {remaining_budget_mw / 1000:.1f} W.")


                # Apply the calculated limits (the function checks if changes are needed)
                self._set_tdp_limits(new_limits_mw)

                # Wait for the next cycle
                time.sleep(self.update_interval)

        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received.")
        except Exception as e:
            logging.exception(f"An unexpected error occurred in the main loop: {e}")
        finally:
            self.shutdown()

    def shutdown(self):
        """Shuts down the NVML library and optionally resets TDP."""
        logging.info("Shutting down GpuTdpBalancer...")
        # Optionally: Reset TDP to default or initial state here
        # For simplicity, we'll just log current state before shutdown
        logging.info(f"Final TDP limits (W): { {k: v/1000 for k, v in self.last_tdp_limits_mw.items()} }")
        try:
            # Attempt to disable persistence mode if it was enabled - requires root
            # This is often desired so settings don't stick after script exits
            for i, handle in enumerate(self.handles):
                 try:
                     # Check if persistence mode was supported and potentially enabled
                     # We didn't explicitly store if we succeeded per-GPU, so try disabling anyway
                     # It's safer to attempt disable than leave it enabled if user doesn't want it
                     mode = nvmlDeviceGetPersistenceMode(handle)
                     if mode == NVML_FEATURE_ENABLED:
                        nvmlDeviceSetPersistenceMode(handle, NVML_FEATURE_DISABLED)
                        logging.debug(f"Disabled persistence mode for GPU {i}.")
                 except NVMLError as e:
                     # Ignore errors if not supported or no permission, log others
                     if e.value not in [NVML_ERROR_NOT_SUPPORTED, NVML_ERROR_NO_PERMISSION]:
                          logging.warning(f"Could not disable persistence mode for GPU {i}: {e}")

            nvmlShutdown()
            logging.info("NVML shut down successfully.")
        except NVMLError as error:
            logging.error(f"Failed to shut down NVML: {error}")