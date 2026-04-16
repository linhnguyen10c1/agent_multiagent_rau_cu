"""
Vẽ biểu đồ so sánh 3 agents
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setup style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 5)
plt.rcParams['font.size'] = 10


def plot_results():
    """Vẽ 3 biểu đồ so sánh"""
    
    # Đọc dữ liệu
    results_dir = Path(__file__).parent.parent / "results"
    
    random_df = pd.read_csv(results_dir / "random_results.csv")
    rule_df = pd.read_csv(results_dir / "rule_based_results.csv")
    ppo_df = pd.read_csv(results_dir / "ppo_results.csv")
    
    print("📊 Vẽ biểu đồ so sánh...\n")
    
    # Tạo figure với 3 subplot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # ===== BIỂU ĐỒ 1: Distribution Total Reward =====
    for df, name, color in [
        (random_df, "Random", "#FF6B6B"),
        (rule_df, "Rule-based", "#4ECDC4"),
        (ppo_df, "PPO", "#45B7D1"),
    ]:
        axes[0].hist(
            df["total_reward"] / 1e6,
            bins=20,
            alpha=0.6,
            label=name,
            color=color,
        )
    
    axes[0].set_xlabel("Total Reward (Triệu đ)", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Số Episodes", fontsize=11, fontweight='bold')
    axes[0].set_title("Phân bố Total Reward", fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # ===== BIỂU ĐỒ 2: Box plot Waste =====
    waste_data = [random_df["waste_kg"], rule_df["waste_kg"], ppo_df["waste_kg"]]
    bp = axes[1].boxplot(
        waste_data,
        labels=["Random", "Rule-based", "PPO"],
        patch_artist=True,
    )
    
    # Tô màu cho box plot
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    axes[1].set_ylabel("Lãng phí (kg)", fontsize=11, fontweight='bold')
    axes[1].set_title("So sánh Lãng phí Hàng", fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')
    
    # ===== BIỂU ĐỒ 3: Scatter Revenue vs Waste =====
    for df, name, color in [
        (random_df, "Random", "#FF6B6B"),
        (rule_df, "Rule-based", "#4ECDC4"),
        (ppo_df, "PPO", "#45B7D1"),
    ]:
        axes[2].scatter(
            df["waste_kg"],
            df["revenue"] / 1e6,
            alpha=0.6,
            label=name,
            c=color,
            s=80,
            edgecolors='black',
            linewidth=0.5,
        )
    
    axes[2].set_xlabel("Lãng phí (kg)", fontsize=11, fontweight='bold')
    axes[2].set_ylabel("Doanh thu (Triệu đ)", fontsize=11, fontweight='bold')
    axes[2].set_title("Trade-off: Revenue vs Waste", fontsize=12, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Lưu figure
    output_path = results_dir / "comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Biểu đồ đã lưu: {output_path}\n")
    
    plt.show()
    
    # In thống kê
    print("="*60)
    print("📈 THỐNG KÊ")
    print("="*60)
    
    print("\nTotal Reward (Triệu đ):")
    print(f"  Random:      {random_df['total_reward'].mean()/1e6:6.2f} ± {random_df['total_reward'].std()/1e6:6.2f}")
    print(f"  Rule-based:  {rule_df['total_reward'].mean()/1e6:6.2f} ± {rule_df['total_reward'].std()/1e6:6.2f}")
    print(f"  PPO:         {ppo_df['total_reward'].mean()/1e6:6.2f} ± {ppo_df['total_reward'].std()/1e6:6.2f}")
    
    print("\nWaste (kg):")
    print(f"  Random:      {random_df['waste_kg'].mean():6.2f} ± {random_df['waste_kg'].std():6.2f}")
    print(f"  Rule-based:  {rule_df['waste_kg'].mean():6.2f} ± {rule_df['waste_kg'].std():6.2f}")
    print(f"  PPO:         {ppo_df['waste_kg'].mean():6.2f} ± {ppo_df['waste_kg'].std():6.2f}")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    plot_results()