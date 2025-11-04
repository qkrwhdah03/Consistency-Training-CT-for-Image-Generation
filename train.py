#!/usr/bin/env python3
"""
Skeleton training script for generative models
Students need to implement their own model classes and modify this script accordingly.
"""

import json
import argparse
import torch
import torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm

from src.network import UNet
from dataset import SimpsonsDataModule, get_data_iterator
from src.utils import tensor_to_pil_image, get_current_time, save_model, seed_everything
from src.base_model import BaseScheduler, BaseGenerativeModel
from custom_model import create_custom_model


def train_model(
    model,
    train_iterator,
    num_iterations=100000,
    lr=1e-4,
    save_dir="./results",
    device="cpu",
    log_interval=500,
    save_interval=5000,
    model_config=None
):
    """
    Train a generative model.
    
    Args:
        model: Generative model to train
        train_iterator: Training data iterator
        num_iterations: Number of training iterations
        lr: Learning rate
        save_dir: Directory to save checkpoints
        device: Device to run training on
        log_interval: Interval for logging
        save_interval: Interval for saving checkpoints and samples
        model_config: Model configuration dictionary to save with checkpoints
    """
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Create optimizer
    optimizer = optim.Adam(model.network.parameters(), lr=lr)
    
    # Save training configuration
    config = {
        'num_iterations': num_iterations,
        'lr': lr,
        'log_interval': log_interval,
        'save_interval': save_interval,
        'device': str(device),
    }
    with open(save_dir / 'training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Training config saved to: {save_dir / 'training_config.json'}")
    
    # Training loop
    train_losses = []
    
    print(f"Starting training for {num_iterations} iterations...")
    print(f"Model: {type(model).__name__}")
    print(f"Device: {device}")
    print(f"Save directory: {save_dir}")
    
    model.train()
    
    pbar = tqdm(range(num_iterations), desc="Training")
    
    for iteration in pbar:
        # Get batch from infinite iterator
        data = next(train_iterator)
        data = data.to(device)
        
        # Generate noise - nothing to do for CM
        
        # Compute loss
        try:
            loss = model.compute_loss(data, iteration)
        except NotImplementedError:
            print("Error: compute_loss method not implemented!")
            print("Please implement the compute_loss method in your model class.")
            return
        except Exception as e:
            print(f"Error computing loss: {e}")
            return
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())

        model.update_teacher()
        
        # Update progress bar
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        # Logging and save loss curve
        if (iteration + 1) % log_interval == 0:
            avg_loss = sum(train_losses[-log_interval:]) / min(log_interval, len(train_losses))
            tqdm.write(f"Iteration {iteration+1}/{num_iterations}, Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}")
            
            # Save training loss curve
            try:
                plt.figure(figsize=(10, 6))
                plt.plot(train_losses)
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.title('Training Loss')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(save_dir / "training_curves.png", dpi=150, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Warning: Could not save training curve: {e}")
        
        if (iteration + 1) % save_interval == 0:
            # Save checkpoint
            checkpoint_path = save_dir / f"checkpoint_iter_{iteration+1}.pt"
            save_model(model, str(checkpoint_path), model_config)
            print(f"\n  Checkpoint saved: {checkpoint_path}")
            print(f"  Config saved: {checkpoint_path.parent / 'model_config.json'}")
        
            
            # Generate samples
            print("\n  Generating samples...")
            model.eval()
            with torch.no_grad():
                shape = (1, 3, 64, 64)
                samples = model.sample(shape, 
                                        num_inference_timesteps=20)
                model.train()
                
                # Save samples
                pil_images = tensor_to_pil_image(samples)
                for i, img in enumerate(pil_images):
                    img.save(save_dir / f"iter={iteration+1}_sample_{i}.png")
            
                
    # Save final model
    final_path = save_dir / "final_model.pt"
    try:
        save_model(model, str(final_path), model_config)
        print(f"Final model saved: {final_path}")
        print(f"Config saved: {final_path.parent / 'model_config.json'}")
    except Exception as e:
        print(f"Error saving final model: {e}")
    
    print(f"\nTraining completed! Results saved to: {save_dir}")
    print("Check the training_curves.png for loss visualization.")


def main(args):
    # Set seed for reproducibility
    seed_everything(args.seed)
    print(f"Seed set to: {args.seed}")
    
    # Add timestamp if save directory is not specified
    if args.save_dir == "./results":
        args.save_dir = f"./results/{get_current_time()}"
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    print("Creating model...")

    # Prepare kwargs for create_custom_model from args
    model_kwargs = {
    }
    # Pass all custom arguments except training-specific ones
    # Network hyperparameters (ch, ch_mult, attn, num_res_blocks, dropout) are FIXED
    # Students can add their own custom arguments for scheduler/model configuration
    excluded_keys = ['device', 'batch_size', 
                     'lr', 'save_dir', 'log_interval', 'save_interval', 'seed']
    for key, value in args.__dict__.items():
        if key not in excluded_keys and value is not None:
            model_kwargs[key] = value

    try:
        model = create_custom_model(
            device=device,
            **model_kwargs
        )
        print(f"Model created: {type(model).__name__}")
        print(f"Scheduler: {type(model.scheduler).__name__}")
        print(f"Parameters: {sum(p.numel() for p in model.network.parameters()):,}")
    except NotImplementedError as e:
        print(f"Error: {e}")
        print("Please implement the CustomScheduler and CustomGenerativeModel classes in custom_model.py")
        return
    except Exception as e:
        print(f"Error creating model: {e}")
        return
    
    # Prepare model configuration for reproducibility
    # This will be saved together with each checkpoint
    model_config = {
        'model_type': type(model).__name__,
        'scheduler_type': type(model.scheduler).__name__,
        **model_kwargs  # Include all custom model arguments
    }
    
    # Load dataset
    print("Loading dataset...")
    try:
        data_module = SimpsonsDataModule(
            batch_size=args.batch_size,
            num_workers=4
        )
        
        train_loader = data_module.train_dataloader()
        train_iterator = get_data_iterator(train_loader)
        
        print(f"Total iterations: {args.num_iterations}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Train model
    train_model(
        model=model,
        train_iterator=train_iterator,
        num_iterations=args.num_iterations,
        lr=args.lr,
        save_dir=args.save_dir,
        device=device,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        model_config=model_config
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train generative model")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--num_iterations", type=int, default=100000,
                       help="Number of training iterations")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="./results",
                       help="Directory to save")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run training on")
    parser.add_argument("--log_interval", type=int, default=100,
                       help="Interval for logging")
    parser.add_argument("--save_interval", type=int, default=5000,
                       help="Interval for saving checkpoints and samples")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")

    # Model-specific arguments (students can add more)
    # DO NOT MODIFY THE PROVIDED NETWORK HYPERPARAMETERS 
    # (ch=128, ch_mult=[1,2,2,2], attn=[1], num_res_blocks=4, dropout=0.1)
    parser.add_argument("--use_additional_condition", action="store_true",
                       help="Use additional condition embedding in U-Net (e.g., step size for Shortcut Models or end timestep for Consistency Trajectory Models)")
    parser.add_argument("--num_train_timesteps", type=int, default=1000,
                       help="Number of training timesteps for scheduler")
    parser.add_argument("--t_min", type=float, default= 0.002)
    parser.add_argument("--t_max", type=float, default= 80)
    parser.add_argument("--rho", type=float, default= 7.0)
    parser.add_argument("--s0", type= int, default= 10)
    parser.add_argument("--s1", type= int, default= 1280)
    parser.add_argument("--p_mean", type= float, default= -1.1)
    parser.add_argument("--p_std", type= float, default= 2.0)
    parser.add_argument("--d", type= int, default= 64*64*3)
    parser.add_argument("--sigma_data", type= float, default= 0.5)
    
    # Students can add their own custom arguments below for their implementation
    # For example:
    # parser.add_argument("--sigma_min", type=float, default=0.001, help="Minimum noise level")
    # parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay rate")
    # etc.
    
    args = parser.parse_args()

    main(args)