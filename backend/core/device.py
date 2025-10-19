from typing import Optional, Dict, Any
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML = True
except Exception:
    _NVML = False

def gpu_info() -> Optional[Dict[str, Any]]:
    if not _NVML:
        return None
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(h).decode()
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        return {
            "name": name,
            "mem_total_gb": round(mem.total/1024**3, 2),
            "mem_used_gb": round(mem.used/1024**3, 2),
            "util": util.gpu,
            "temp_c": temp
        }
    except Exception:
        return None
