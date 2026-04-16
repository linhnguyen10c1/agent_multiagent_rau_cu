"""
Đánh giá 3 agents trên 100 episodes
"""
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from env.vegetable_env import VegetableMarketEnv
from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from stable_baselines3 import PPO


def evaluate(agent, env, n_episodes=100, name="agent"):
    """
    Đánh giá agent
    
    Args:
        agent: Agent cần đánh giá
        env: Environment
        n_episodes: Số episodes để chạy
        name: Tên agent (để lưu file)
    
    Returns:
        df: DataFrame kết quả
    """
    results = []
    
    print(f"\n{'='*60}")
    print(f"Đánh giá {name.upper()} ({n_episodes} episodes)...")
    print(f"{'='*60}\n")
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, _ = agent.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
        
        results.append({
            "episode": ep,
            "total_reward": total_reward,
            "revenue": env.cum_revenue,
            "waste_kg": env.cum_waste,
            "customers_bought": env.customers_bought,
            "price_changes": env.price_change_count,
            "sell_through_rate": (80 - env.cum_waste) / 80 * 100,
        })
        
        if (ep + 1) % 10 == 0:
            print(f"   ✓ {ep + 1:3d}/{n_episodes} episodes done")
    
    df = pd.DataFrame(results)
    
    # Lưu CSV
    output_path = Path(__file__).parent.parent / "results" / f"{name}_results.csv"
    df.to_csv(output_path, index=False)
    
    # In thống kê
    print(f"\n{'='*60}")
    print(f"KẾT QUẢ: {name.upper()}")
    print(f"{'='*60}")
    print(f"\nTotal Reward:")
    print(f"   Trung bình: {df['total_reward'].mean():12.0f}đ")
    print(f"   Std dev:   {df['total_reward'].std():12.0f}đ")
    print(f"   Min:       {df['total_reward'].min():12.0f}đ")
    print(f"   Max:       {df['total_reward'].max():12.0f}đ")
    
    print(f"\nRevenue:")
    print(f"   Trung bình: {df['revenue'].mean():12.0f}đ")
    print(f"   Std dev:   {df['revenue'].std():12.0f}đ")
    
    print(f"\nWaste:")
    print(f"   Trung bình: {df['waste_kg'].mean():12.1f}kg")
    print(f"   Std dev:   {df['waste_kg'].std():12.1f}kg")
    
    print(f"\nSell-through rate:")
    print(f"   Trung bình: {df['sell_through_rate'].mean():11.1f}%")
    
    print(f"\nCustomer satisfaction:")
    satisfaction = df['customers_bought'] / (df['customers_bought'] + 1e-6)
    print(f"   Trung bình: {satisfaction.mean()*100:11.1f}%")
    
    print(f"\nKết quả lưu: {output_path}\n")
    
    return df


def main():
    """Đánh giá tất cả agents"""
    
    print("\n" + "="*60)
    print("ĐÁNH GIÁ 3 AGENTS TRÊN MÔI TRƯỜNG RẦU CỦ")
    print("="*60)
    
    # Tạo environment
    env = VegetableMarketEnv()
    
    # Đánh giá Random Agent
    print("\n1️⃣ RANDOM AGENT (Baseline cận dưới)")
    random_df = evaluate(RandomAgent(env), env, n_episodes=100, name="random")
    
    # Đánh giá Rule-based Agent
    print("\n2️⃣  RULE-BASED AGENT (Baseline thực tế)")
    rule_df = evaluate(RuleBasedAgent(), env, n_episodes=100, name="rule_based")
    
    # Đánh giá PPO Agent
    print("\n3️⃣  PPO AGENT (Học từ RL)")
    model_path = Path(__file__).parent.parent / "models" / "ppo_vegetable"
    ppo_model = PPO.load(str(model_path))
    ppo_df = evaluate(ppo_model, env, n_episodes=100, name="ppo")
    
    # So sánh
    print("\n" + "="*60)
    print("📊 SO SÁNH TỔNG QUÁT")
    print("="*60)
    
    comparison = pd.DataFrame({
        "Agent": ["Random", "Rule-based", "PPO"],
        "Avg Reward": [
            random_df['total_reward'].mean(),
            rule_df['total_reward'].mean(),
            ppo_df['total_reward'].mean(),
        ],
        "Avg Waste": [
            random_df['waste_kg'].mean(),
            rule_df['waste_kg'].mean(),
            ppo_df['waste_kg'].mean(),
        ],
        "Avg Revenue": [
            random_df['revenue'].mean(),
            rule_df['revenue'].mean(),
            ppo_df['revenue'].mean(),
        ],
    })
    
    print("\n" + comparison.to_string(index=False))
    
    # Kết luận
    print("\n" + "="*60)
    print("🎯 KẾT LUẬN")
    print("="*60)
    
    ppo_vs_random = ppo_df['total_reward'].mean() / random_df['total_reward'].mean()
    ppo_vs_rule = ppo_df['total_reward'].mean() / rule_df['total_reward'].mean()
    
    print(f"\n✓ PPO reward / Random reward: {ppo_vs_random:.2f}x")
    print(f"✓ PPO reward / Rule-based reward: {ppo_vs_rule:.2f}x")
    
    if ppo_vs_rule >= 1.0:
        print(f"\n✅ PPO VƯỢT QUA RULE-BASED! Demo thành công! 🎉\n")
    else:
        print(f"\n⚠️  PPO chưa vượt qua Rule-based. Cần debug hoặc train lâu hơn.\n")


if __name__ == "__main__":
    main()