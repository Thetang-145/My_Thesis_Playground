import logging
import torch
import time
from pynvml import *
from pathlib import Path
import datetime
import pytz
import argparse

def logCuda(args):
    nvmlInit()
    cuda_name = torch.cuda.get_device_name(args.cuda)
    gb = pow(1024,3)
    tokyo_tz = pytz.timezone('Asia/Tokyo')

    while True:
        h = nvmlDeviceGetHandleByIndex(args.cuda)
        info = nvmlDeviceGetMemoryInfo(h)
        log_msg = f"{cuda_name} | Used: {info.used/gb:.4f}/{info.total/gb:.4f} GB, Free: {info.free/gb:.4f} GB"
        logging.info(log_msg)
        dt = datetime.datetime.now()
        dt = dt.astimezone(tokyo_tz)
        # dt = datetime.strptime(dt, '%Y-%m-%d %H:%M:%S |')
        print(f"{dt} | {log_msg}")
        time.sleep(args.delay)
        
def convertM2S(m):
    s = int(m*60)
    return min([10, s])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--delay', default=1 , type=float, help="delay logging (min)")
    parser.add_argument('--cuda', default=0 , type=int, help="cuda:n")
    args = parser.parse_args()
    
    log_dir = "log"
    if not(Path(log_dir).exists()): os.system(f"mkdir -p {log_dir}")
    logging.basicConfig(
        filename=f'{log_dir}/cuda.log',
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s: %(message)s',
    )
    args.delay = convertM2S(args.delay)
    logCuda(args)   
        
        
if __name__ == "__main__":
    main()