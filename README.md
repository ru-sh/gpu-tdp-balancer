# GPU TDP Balancer

Dynamically balances and sets NVIDIA GPU power (TDP) limits based on current device utilization, working within a configurable maximum total power budget for the entire GPU cluster.

## Overview

This tool monitors the utilization of all NVIDIA GPUs detected on the system. Based on configurable thresholds, it categorizes the system state as either ACTIVE (at least one GPU is highly utilized) or PASSIVE (all GPUs are lightly utilized).

*   In the **ACTIVE** state, it assigns a minimum configured TDP to inactive GPUs and distributes the remaining power budget proportionally (based on max hardware TDP) among the active GPUs.
*   In the **PASSIVE** and **TRANSITION** state, it distributes the total power budget proportionally (based on max hardware TDP) across all GPUs, ensuring no GPU goes below the configured minimum TDP.

This allows for power savings when GPUs are idle while ensuring that active GPUs receive a larger share of the power budget when needed, without exceeding an overall system limit.

## Requirements

*   **Operating System:** Linux or Windows (tested primarily on Linux).
*   **NVIDIA Drivers:** Recent NVIDIA drivers must be installed.
*   **NVML:** NVIDIA Management Library (comes with the drivers).
*   **Python:** Python 3.6+
*   **pynvml:** Python bindings for NVML (`pip install pynvml`).
*   **Root/Administrator Privileges:** Required to set GPU power limits via NVML.

## Installation

1.  **Clone the repository (or download the source code):**
    ```bash
    git clone git@github.com:ru-sh/gpu-tdp-balancer.git
    cd gpu-tdp-balancer.git
    ```
2.  **Install Python dependencies:**
    ```bash
    pip install pynvml
    # Or if you create a requirements.txt:
    # pip install -r requirements.txt
    ```
    *(You might want to use a Python virtual environment)*

## Usage

Run the main script with root/administrator privileges.

```bash
sudo python3 src/main.py [OPTIONS]
```

or on Windows (in an Administrator terminal):

```
python src\main.py [OPTIONS]
```

## Command-Line Arguments

| Argument            | Type    | Default | Description                                                                      |
| :------------------ | :------ | :------ | :------------------------------------------------------------------------------- |
| `--max-total-tdp`   | `int`   | `800`   | Maximum total TDP allowed across all GPUs (Watts).                               |
| `--min-gpu-tdp`     | `int`   | `100`   | Minimum TDP limit for any single GPU (Watts). Will be clamped by hardware min.   |
| `--active-level`    | `int`   | `20`    | GPU utilization (%) threshold to consider a GPU 'active'.                        |
| `--passive-level`   | `int`   | `10`    | GPU utilization (%) threshold below which *all* GPUs must be for 'passive' state. |
| `--interval`        | `float` | `1.0`   | Update interval in seconds.                                                      |
| `-v`, `--verbose`   | `flag`  | `False` | Enable verbose (DEBUG level) logging.                                            |



## Example
Run the balancer with a total budget of 700W, a minimum GPU TDP of 150W, and verbose logging:

```bash
sudo python3 src/main.py --max-total-tdp 700 --min-gpu-tdp 150 --verbose
```

## Docker Usage
You can run the balancer inside a Docker container using the NVIDIA Container Toolkit.

### Docker Requirements
- Docker installed.
- NVIDIA Drivers installed on the host.
- NVIDIA Container Toolkit installed on the host. Follow the official guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

```bash
docker run --rm --gpus all --cap-add=SYS_ADMIN shakirovruslan/gpu-tdp-balancer [OPTIONS]
```

### Build the Docker Image
A Dockerfile is provided. Build the image using:

```bash
docker build -t gpu-tdp-balancer .
```

### Run the Container
Run the container, making sure to grant GPU access and necessary capabilities:

```bash
docker run --rm --gpus all --cap-add=SYS_ADMIN gpu-tdp-balancer [OPTIONS]
```

- --rm: Removes the container on exit.
- --gpus all: Grants access to all host GPUs (requires NVIDIA Container Toolkit). You can specify devices, e.g., --gpus '"device=0,1"'.
- --cap-add=SYS_ADMIN: Grants necessary privileges for NVML power management operations. Be aware of the security implications.
- [OPTIONS]: Add any command-line arguments for the script (e.g., --max-total-tdp 700).

### Example (Docker):
```bash
docker run --rm --gpus all --cap-add=SYS_ADMIN gpu-tdp-balancer --max-total-tdp 700 --min-gpu-tdp 30 --verbose
```

## How it Works

1.  **Initialization:** Detects GPUs, retrieves their hardware TDP limits (min/max), and verifies configuration parameters.
2.  **Monitoring Loop:** Periodically (defined by `--interval`):
    *   Gets current utilization (%) and power limits (W) for all GPUs.
    *   Determines the system state (ACTIVE, PASSIVE, TRANSITION) based on utilization and the `--active-level` / `--passive-level` thresholds.
    *   Calculates target TDP limits based on the state:
        *   **ACTIVE:** Assigns `--min-gpu-tdp` to GPUs below `--active-level`. The remaining budget from `--max-total-tdp` is distributed among active GPUs, proportional to their maximum hardware TDP. Adjustments are made to ensure no active GPU gets less than `--min-gpu-tdp` and to try and stay within the budget.
        *   **PASSIVE:** Distributes `--max-total-tdp` proportionally across *all* GPUs based on their maximum hardware TDP. Adjustments are made to ensure no GPU gets less than `--min-gpu-tdp` and to try and stay within the budget.
        *   **TRANSITION:** Treats as if no GPUs are active, assigning `--min-gpu-tdp` to all (via `active_split` with no active GPUs).
    *   Applies the calculated target TDP limits. Limits are clamped between the GPU's hardware min/max and the configured `--min-gpu-tdp`. The `nvmlDeviceSetPowerManagementLimit` call is only made if the calculated (and clamped) target differs from the current limit.
3.  **Logging:** Reports state changes, target limit calculations, and actual TDP setting actions. Verbose mode provides per-cycle utilization and limit details.
4.  **Shutdown:** Catches `SIGINT`/`SIGTERM` (Ctrl+C, kill) for graceful shutdown, ensuring NVML resources are released.


# Important Notes
- **Root/Administrator Privileges**: Setting GPU power limits requires elevated privileges. The script includes checks and warnings if not run as root/admin.
- **Hardware Limits**: The balancer respects the minimum and maximum TDP limits reported by the GPU hardware. The configured --min-gpu-tdp cannot override the hardware minimum if the hardware minimum is higher. Similarly, calculated TDPs are capped at the hardware maximum.
- **NVML Errors**: The script includes error handling for common NVML issues (e.g., initialization failures, communication errors during updates). If frequent NVML errors occur, check the system's NVIDIA driver status (nvidia-smi).
- **Budget Exceeded Warnings**: In some scenarios (e.g., when enforcing --min-gpu-tdp for many GPUs pushes the total above --max-total-tdp), the script might log warnings if it cannot perfectly adhere to the budget after applying minimums. The final `set_tdp_limits` step still clamps individual GPUs within their hardware limits.

## Disclaimer / Limitation of Liability

This software is provided 'AS IS', without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.

Use this tool responsibly and at your own risk. Modifying hardware settings improperly can potentially lead to system instability or damage. 

# License
This project is licensed under the MIT License.