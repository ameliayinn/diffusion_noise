from config import get_config

if __name__ == "__main__":
    cfg = get_config()
    
    if cfg.use_different_noise:
        from generate_with_different_noise import generate_samples
        from train_with_different_noise import train_deepspeed
    else:
        from generate import generate_samples
        from train import train_deepspeed

    if cfg.mode == "train":
        train_deepspeed(cfg)
    elif cfg.mode == "generate":
        if not cfg.model_path:
            raise ValueError("Model path must be specified for generation mode")
        generate_samples(cfg, cfg.model_path, cfg.num_images)