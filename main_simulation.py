from config import get_config
from train_simulation import train_deepspeed
from generate import generate_samples

if __name__ == "__main__":
    cfg = get_config()
    
    if cfg.mode == "train":
        train_deepspeed(cfg)
    elif cfg.mode == "generate":
        if not cfg.model_path:
            raise ValueError("Model path must be specified for generation mode")
        generate_samples(cfg, cfg.model_path, cfg.num_images)