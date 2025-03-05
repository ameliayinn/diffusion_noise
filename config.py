import argparse

def get_config():
    parser = argparse.ArgumentParser(description="Diffusion Model Training with DeepSpeed")
    
    # 训练参数
    parser.add_argument("--image_size", type=int, default=64, help="Input image size")
    parser.add_argument("--batch_size", type=int, default=256, help="Global batch size")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--dataset_name", type=str, default="cats_vs_dogs", help="Dataset name")
    
    # 路径参数
    parser.add_argument("--samples_dir", type=str, default="./samples", help="Directory to save samples")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    
    # 精度设置
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision training (default: True)")
    parser.add_argument("--no-fp16", action="store_false", dest="fp16", help="Disable mixed precision training")
    
    # 运行模式
    parser.add_argument("--mode", choices=["train", "generate"], default="train", help="Run mode")
    parser.add_argument("--model_path", type=str, help="Model path for generation mode")
    parser.add_argument("--num_images", type=int, default=16, help="Number of images to generate")
    
    parser.set_defaults(fp16=True)
    return parser.parse_args()