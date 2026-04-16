"""
Train PPO agent trên môi trường rau củ
Thời gian: khoảng 15 phút trên laptop CPU
"""
import sys
from pathlib import Path

# Add parent directory to path để import được VegetableMarketEnv
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from env.vegetable_env import VegetableMarketEnv


def make_env():
    """Factory function để tạo environment"""
    return Monitor(VegetableMarketEnv())


def train_ppo():
    """Train PPO model"""
    print("Khởi tạo environment...\n")
    
    # Tạo vectorized environment
    env = DummyVecEnv([make_env])
    
    print("Khởi tạo PPO model...\n")
    
    # Khởi tạo PPO model
    model = PPO(
        "MlpPolicy",                    # Policy type (MLP cho observation liên tục)
        env,
        learning_rate=3e-4,             # Learning rate
        n_steps=2048,                   # Steps per rollout
        batch_size=64,                  # Batch size
        n_epochs=10,                    # Number of epochs
        gamma=0.99,                     # Discount factor
        gae_lambda=0.95,                # Generalized Advantage Estimate
        clip_range=0.2,                 # PPO clip range
        verbose=1,                      # In log mỗi update
        tensorboard_log="./results/tb_logs/",
    )
    
    print("Bắt đầu training 200,000 timesteps...")
    print("Dự kiến thời gian: ~15 phút\n")
    
    # Train
    model.learn(
        total_timesteps=200_000,
        log_interval=10,  # In log mỗi 10 update
       # progress_bar=True,  # Hiển thị progress bar
    )
    
    print("\nTraining hoàn thành!\n")
    
    # Lưu model
    model_path = Path(__file__).parent.parent / "models" / "ppo_vegetable"
    model.save(str(model_path))
    
    print(f"Model đã lưu: {model_path}.zip\n")
    
    print("Xem training progress với TensorBoard:")
    print("tensorboard --logdir ./results/tb_logs/\n")
    
    env.close()


if __name__ == "__main__":
    train_ppo()