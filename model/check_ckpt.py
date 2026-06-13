import torch
from pathlib import Path

ckpt_path = 'runs/detect/yolov8_mobilenetv34/weights/last.pt'
if Path(ckpt_path).exists():
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        print(f"Checkpoint keys: {ckpt.keys()}")
        
        if 'ema' in ckpt:
            ema = ckpt['ema']
            print(f"EMA type: {type(ema)}")
            if hasattr(ema, 'nc'):
                print(f"EMA nc: {ema.nc}")
            elif hasattr(ema, 'model'):
                 if hasattr(ema.model, 'nc'):
                     print(f"EMA model nc: {ema.model.nc}")
            
    except Exception as e:
        print(f"Error: {e}")
