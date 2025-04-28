import time
import logging
from typing import List, Tuple, Optional
from pynvml import (
    nvmlInit, nvmlShutdown, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetName, nvmlDeviceGetPowerManagementLimitConstraints,
    nvmlDeviceGetPowerManagementLimit, nvmlDeviceSetPowerManagementLimit,
    nvmlDeviceGetUtilizationRates,
    NVMLError
)

class GpuTdpBalancer:
    """
    Dynamically balances and sets GPU power (TDP) limits
    based on current device utilization, within a max cluster TDP budget.
    """

    def __init__(self,
            gpu_max_tdp_total_w: int = 800,
            gpu_min_tdp_per_gpu_w: int = 100,
            gpu_active_level_percent: int = 20,
            gpu_passive_level_percent: int = 10,
            update_interval_sec: float = 1.0,
        ) -> None:
        self.max_total_tdp_w = gpu_max_tdp_total_w
        self.min_tdp_w = gpu_min_tdp_per_gpu_w
        self.active_level = gpu_active_level_percent
        self.passive_level = gpu_passive_level_percent
        self.interval = update_interval_sec

        self.running = True
        self._nvml_initialized = False  # Track NVML state
        self._last_state = None # Track balancer state for logging
        self._last_target_limits = None # Track last *calculated* target limits for logging
        self._init_nvml()
        self._initialize_device_data()

    def _init_nvml(self) -> None:
        try:
            nvmlInit()
            self._nvml_initialized = True # Flag successful initialization
            logging.info("NVML initialized successfully.")
        except NVMLError as e:
            logging.error(f"Failed to initialize NVML: {e}")
            self._nvml_initialized = False
            raise # Re-raise to be caught by main

    def _initialize_device_data(self) -> None:
        if not self._nvml_initialized:
            raise RuntimeError("Cannot initialize device data: NVML not initialized.")
        try:
            self.gpu_count = nvmlDeviceGetCount()
            if self.gpu_count == 0:
                 raise RuntimeError("No NVIDIA GPUs detected by NVML.")
            self.handles = [nvmlDeviceGetHandleByIndex(i) for i in range(self.gpu_count)]
            self.device_names = [nvmlDeviceGetName(h) for h in self.handles]
            # TDP limits (Watts)
            self.tdp_limits: List[Tuple[int, int]] = []
            for i, h in enumerate(self.handles):
                try:
                    limits_mw = nvmlDeviceGetPowerManagementLimitConstraints(h)
                     # Ensure limits are valid (non-zero, min <= max)
                    if limits_mw[0] <= 0 or limits_mw[1] <= 0 or limits_mw[0] > limits_mw[1]:
                         logging.warning(f"GPU {i} ({self.device_names[i]}) reported invalid power limits: {limits_mw} mW. Check nvidia-smi.")
                         # Fallback or raise error - using reported max as min for now, but this is dubious
                         min_w = max(1, limits_mw[1] // 1000) # Use max as min if min is invalid/zero
                         max_w = limits_mw[1] // 1000
                         # Ensure min_tdp_w config isn't below this potentially odd hardware min
                         self.min_tdp_w = max(self.min_tdp_w, min_w)
                         logging.warning(f"Adjusted effective min TDP to {self.min_tdp_w}W due to hardware report.")
                    else:
                         min_w = limits_mw[0] // 1000
                         max_w = limits_mw[1] // 1000
                    self.tdp_limits.append( (min_w, max_w) )
                except NVMLError as e:
                     logging.error(f"Failed to get power constraints for GPU {i} ({self.device_names[i]}): {e}")
                     raise # Re-raise critical initialization error
            self.tdp_max: List[int] = [lim[1] for lim in self.tdp_limits]
            self.tdp_min: List[int] = [lim[0] for lim in self.tdp_limits]
            logging.info(f"Detected {self.gpu_count} GPUs: {self.device_names}")
            logging.info(f"Hardware TDP ranges (W): {self.tdp_limits}")
            logging.info(f"Configured Min TDP per GPU: {self.min_tdp_w}W")
            logging.info(f"Configured Max Total TDP: {self.max_total_tdp_w}W")

        except NVMLError as e:
            logging.error(f"NVML error during device data initialization: {e}")
            self.shutdown() # Attempt cleanup
            raise RuntimeError(f"NVML error initializing device data: {e}") from e

    def get_loads_and_limits(self) -> Tuple[List[int], List[int]]:
        """Gets current GPU utilization (%) and power limits (W)."""
        usages = []
        cur_limits = []
        for i, h in enumerate(self.handles):
            try:
                usages.append(nvmlDeviceGetUtilizationRates(h).gpu)
                cur_limits.append(nvmlDeviceGetPowerManagementLimit(h) // 1000)
            except NVMLError as e:
                logging.error(f"Failed to get data for GPU {i} ({self.device_names[i]}): {e}")
                # Re-raise to be caught by the run loop's handler
                raise
        return usages, cur_limits

    def set_tdp_limits(self, new_limits: List[int], cur_limits: List[int]) -> None:
        """
        Set new TDP limits for all GPUs, but only if the new value differs
        from the current value and is within hardware constraints.
        """
        for idx, (h, target, min_hw, max_hw, name, current) in enumerate(
            zip(self.handles, new_limits, self.tdp_min, self.tdp_max, self.device_names, cur_limits)
        ):
            # Clamp target value within hardware limits AND configured minimum
            w = max(min(target, max_hw), min_hw, self.min_tdp_w)

            if w == current:
                logging.debug(f"No changes {name} (GPU {idx}) TDP: {w}W (Target: {target}W, HW Range: {min_hw}-{max_hw}W)")
                continue  # No change needed for this GPU

            try:
                nvmlDeviceSetPowerManagementLimit(h, w * 1000)
                logging.info(f"Set {name} (GPU {idx}) TDP: {w}W (Target: {target}W, HW Range: {min_hw}-{max_hw}W)")
            except NVMLError as e:
                # Log error but continue trying to set for other GPUs
                logging.error(f"Failed to set TDP for {name} (GPU {idx}) to {w}W: {e}")

    def active_split(self, usages: List[int]) -> List[int]:
        """Assign TDP: min_tdp_w to low-usage GPUs, rest distributed proportionally by max TDP across active GPUs."""
        active_indices = [i for i, u in enumerate(usages) if u >= self.active_level]
        inactive_indices = [i for i, u in enumerate(usages) if u < self.active_level]

        limits = [0] * self.gpu_count # Initialize limits list

        # Assign minimum TDP to all inactive GPUs first
        for i in inactive_indices:
            limits[i] = self.min_tdp_w

        # Calculate budget for active and set inactive to min_tdp_w first
        total_inactive_min = len(inactive_indices) * self.min_tdp_w
        tdp_budget_for_active = max(0, self.max_total_tdp_w - total_inactive_min)

        if active_indices and tdp_budget_for_active > 0:
            # Distribute remaining budget proportionally among active GPUs based on their max TDP
            max_tdp_sum_active = sum(self.tdp_max[i] for i in active_indices)

            if max_tdp_sum_active <= 0:
                logging.warning("Sum of max TDP for active GPUs is zero. Assigning min TDP to all GPUs.")
                for i in range(self.gpu_count):
                    limits[i] = self.min_tdp_w
            else:
                # First set inactive GPUs to minimum
                temp_limits = {}
                for i in range(self.gpu_count):
                    if i not in active_indices:
                        limits[i] = self.min_tdp_w
                    else:
                        temp_limits[i] = 0

                # Calculate proportional share for active GPUs using corrected budget
                for i in active_indices:
                    prop_share = int(tdp_budget_for_active * self.tdp_max[i] / max_tdp_sum_active)
                    temp_limits[i] = max(self.min_tdp_w, prop_share)

                current_total = sum(temp_limits.values())
                if current_total > tdp_budget_for_active + total_inactive_min: # Check against original budget
                    overbudget = current_total - (tdp_budget_for_active + total_inactive_min)
                    reducible_sum = sum(max(0, temp_limits[i] - self.min_tdp_w) for i in active_indices)

                    if reducible_sum > 0:
                        # Reduce proportionally from active GPUs
                        for i in active_indices:
                            if temp_limits[i] > self.min_tdp_w:
                                reduction = int(overbudget * (temp_limits[i] - self.min_tdp_w) / reducible_sum)
                                limits[i] = max(self.min_tdp_w, temp_limits[i] - reduction)
                        # Inactive already set correctly
                    else:
                        logging.warning("Active split budget overrun could not be corrected. All active GPUs at min TDP.")
                        for i in range(self.gpu_count):
                            if i in active_indices:
                                limits[i] = self.min_tdp_w
                            else:
                                limits[i] = self.min_tdp_w # Ensure inactive also get min

                else:
                    for i in temp_limits:  # Use calculated limits (already includes inactive mins)
                        limits[i] = temp_limits[i]
        else:
            # No budget or no active GPUs - set everyone to min
            for i in range(self.gpu_count):
                limits[i] = self.min_tdp_w

        return limits


    def passive_split(self) -> List[int]:
        """Distribute total budget across all GPUs proportionally by max TDP, respecting min_tdp_w."""
        max_tdp_sum = sum(self.tdp_max)
        if max_tdp_sum <= 0: # Avoid division by zero
             logging.warning("Sum of max TDPs is zero. Returning min_tdp_w for all.")
             return [self.min_tdp_w] * self.gpu_count

        # Calculate initial proportional split
        calculated_limits = [
             int(self.max_total_tdp_w * self.tdp_max[i] / max_tdp_sum)
             for i in range(self.gpu_count)
        ]

        # Ensure minimum configured TDP is respected *after* proportional calculation
        final_limits = [max(self.min_tdp_w, limit) for limit in calculated_limits]

        # Adjust if applying min_tdp_w caused budget overrun
        current_total = sum(final_limits)
        if current_total > self.max_total_tdp_w:
            logging.debug(f"Passive split after applying min_tdp_w exceeds budget ({current_total}W > {self.max_total_tdp_w}W). Attempting reduction.")
            # Attempt to reduce those above min_tdp_w proportionally to fit budget
            overbudget = current_total - self.max_total_tdp_w
            reducible_sum = sum(final_limits[i] - self.min_tdp_w for i in range(self.gpu_count) if final_limits[i] > self.min_tdp_w)

            if reducible_sum > 0:
                temp_limits = list(final_limits) # Create copy to modify
                for i in range(self.gpu_count):
                    if final_limits[i] > self.min_tdp_w:
                        reduction = int(overbudget * (final_limits[i] - self.min_tdp_w) / reducible_sum)
                        temp_limits[i] = max(self.min_tdp_w, final_limits[i] - reduction)
                final_limits = temp_limits # Update final_limits with reduced values
                new_total = sum(final_limits)
                logging.debug(f"Passive split reduced to {new_total}W to meet budget.")
            else:
                # Cannot reduce further, accept budget overrun (set_tdp_limits will clamp individually)
                logging.warning(f"Passive split budget overrun ({current_total}W > {self.max_total_tdp_w}W) could not be fully corrected as all GPUs are at min TDP.")
                # Limits remain as they are (floored up)

        return final_limits

    def run(self) -> None:
        """Main control loop."""
        logging.info("Starting GPU TDP Balancer loop...")
        if not self._nvml_initialized or self.gpu_count == 0:
            logging.error("Balancer cannot run: NVML not initialized or no GPUs found.")
            self.running = False
            return # Exit run method early

        try:
            while self.running:
                try:
                    # --- Start of error-handled block for one update cycle ---
                    usages, cur_limits = self.get_loads_and_limits()
                    logging.debug(f"GPU usages: {usages} %, Current Power limits: {cur_limits} W")

                    is_any_active = any(u >= self.active_level for u in usages)
                    is_all_passive = all(u < self.passive_level for u in usages)

                    new_limits: List[int] = []
                    state: str = ""

                    if is_any_active:
                        # ACTIVE state: At least one GPU is heavily utilized
                        state = "ACTIVE"
                        new_limits = self.active_split(usages)
                    elif is_all_passive:
                        # PASSIVE state: All GPUs are below the passive threshold
                        state = "PASSIVE"
                        new_limits = self.passive_split()
                    else:
                        state = "TRANSITION"
                        new_limits = self.passive_split()

                    # Check if state or calculated target limits have changed since last cycle
                    target_limits_changed = new_limits != self._last_target_limits # Compare with previous *target*
                    state_changed = state != self._last_state

                    # Log state changes or if *calculated* target limits changed
                    if state_changed or target_limits_changed:
                         logging.info(f"Balancer state: {state}. Target limits: {new_limits} W")
                    # Log at DEBUG level if neither state nor calculated target limits changed
                    elif logging.getLogger().isEnabledFor(logging.DEBUG):
                         log_msg = f"Balancer state: {state}. Target limits: {new_limits} W (No change calculated)."
                         # Optionally add current limits info if they differ from target (useful for debug)
                         if new_limits != cur_limits:
                              log_msg += f" Current HW limits: {cur_limits} W."
                         logging.debug(log_msg)

                    self._last_state = state # Store state for next iteration comparison
                    self._last_target_limits = new_limits # Store calculated limits for next iteration

                    # Set limits, passing current limits for comparison inside the function
                    # set_tdp_limits handles clamping and only calls nvml if needed
                    self.set_tdp_limits(new_limits, cur_limits)

                except NVMLError as e:
                    logging.error(f"NVML error during update cycle: {e}. Skipping this update.")
                    # Optional: Add a small delay here if errors are frequent and persistent
                    # time.sleep(self.interval * 2)
                except Exception as e:
                     # Catch unexpected errors within the loop to prevent crashing the balancer
                     logging.exception(f"Unexpected error during update cycle: {e}. Skipping this update.")


                # Wait for the next interval regardless of success or error in this cycle
                if self.running: # Check running flag again before sleeping
                    time.sleep(self.interval)

        finally:
            logging.info("Exiting balancer loop.")
            self.running = False # Explicitly set running to False
            self.shutdown() # Ensure NVML resources are released

    def shutdown(self) -> None:
        """Shuts down NVML cleanly."""
        # Check if NVML was initialized successfully before trying to shut down
        if hasattr(self, '_nvml_initialized') and self._nvml_initialized:
            try:
                nvmlShutdown()
                logging.info("NVML shutdown complete.")
                self._nvml_initialized = False # Mark as shut down
            except NVMLError as e:
                # Log NVMLError specifically during shutdown
                logging.warning(f"NVML error during shutdown: {e}")
            except Exception as e:
                # Catch any other unexpected errors during shutdown
                logging.error(f"Unexpected error during NVML shutdown: {e}")
        else:
            logging.debug("NVML shutdown skipped (was not initialized or already shut down).")