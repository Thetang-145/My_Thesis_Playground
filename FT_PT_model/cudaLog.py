import logging
import torch
import time
from pynvml import *
from pathlib import Path


def main():
    log_dir = "log"
    if not(Path(log_dir).exists()): os.system(f"mkdir -p {log_dir}")
    logging.basicConfig(
        filename=f'{log_dir}/cuda.log',
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s: %(message)s',
    )
    
    nvmlInit()
    cuda_num = 0
    cuda_name = torch.cuda.get_device_name(cuda_num)
    gb = pow(1024,3)
    
    while True:
        print(f"Log")
        h = nvmlDeviceGetHandleByIndex(cuda_num)
        info = nvmlDeviceGetMemoryInfo(h)
        logging.info(f"{cuda_name} | Used: {info.used/gb:.4f}/{info.total/gb:.4f} GB, Free: {info.free/gb:.4f} GB")
        time.sleep(30)
        
    
        
        
if __name__ == "__main__":
    main()