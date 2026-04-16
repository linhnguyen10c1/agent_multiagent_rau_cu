"""
Environment cho bài toán quản lý rau củ siêu thị
Sử dụng Gymnasium API
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from pathlib import Path


class VegetableMarketEnv(gym.Env):
    """
    Môi trường mô phỏng siêu thị bán rau 1 ngày
    
    State: 14 chiều (tồn kho, giá, thời gian, thời tiết, ...)
    Action: 5 mức giảm giá (0%, 10%, 20%, 30%, 50%)
    Reward: Lợi nhuận hôm nay (doanh thu - phạt)
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, data_path=None):
        """
        Args:
            data_path: Đường dẫn đến file CSV dữ liệu. 
                      Nếu None, tìm ở thư mục data/
        """
        super().__init__()
        
        # Tìm file dữ liệu
        if data_path is None:
            # Tìm file trong thư mục data
            current_dir = Path(__file__).parent.parent
            data_path = current_dir / "data" / "vegetable_90days.csv"
        
        self.df = pd.read_csv(data_path)
        
        # Định nghĩa action space
        self.action_space = spaces.Discrete(5)
        
        # Định nghĩa observation space (14 chiều)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(14,), 
            dtype=np.float32
        )
        
        # Mapping hành động -> hệ số giảm giá
        self.action_to_discount = [1.0, 0.9, 0.8, 0.7, 0.5]
        
        # Khởi tạo state
        self.reset()
    
    def reset(self, seed=None, options=None):
        """
        Reset environment cho episode mới
        
        Returns:
            obs: Observation (state vector)
            info: Dict rỗng
        """
        super().reset(seed=seed)
        
        # Chọn ngẫu nhiên 1 ngày trong 90 ngày
        self.day_data = self.df.sample(1).iloc[0]
        self.step_idx = 0
        
        # Khởi tạo biến trạng thái
        self.base_price = 25000  # Giá gốc cà chua (đ/kg)
        self.current_price = self.base_price
        
        self.stock = 80.0  # Nhập 80 kg đầu ngày
        self.initial_stock = self.stock
        
        self.cum_revenue = 0.0  # Doanh thu tích lũy
        self.cum_waste = 0.0    # Lãng phí tích lũy
        self.customers_arrived = 0.0
        self.customers_bought = 0.0
        self.price_change_count = 0
        
        # Tuổi rau (0: mới, 1: hôm qua)
        self.age_days = np.random.choice([0, 1])
        
        return self._get_obs(), {}
    
    def step(self, action):
        """
        Thực hiện 1 bước thời gian (1 giờ)
        
        Args:
            action: 0-4 (mức giảm giá)
        
        Returns:
            obs: Observation mới
            reward: Reward nhận được
            terminated: Có kết thúc episode?
            truncated: Có bị cut off?
            info: Dict info thêm
        """
        # 1. Áp dụng hành động (điều chỉnh giá)
        new_price = self.base_price * self.action_to_discount[action]
        price_change = abs(new_price - self.current_price) / 1000  # Convert to 1000đ
        
        if new_price != self.current_price:
            self.price_change_count += 1
        
        self.current_price = new_price
        
        # 2. Tính cầu khách hàng dựa trên giá mới
        demand = self._compute_demand()
        
        # 3. Tính số lượng bán được
        sold = min(demand, self.stock)
        
        # 4. Tính số khách mất (không mua được)
        lost_customers = max(0, demand - sold) / 0.3  # ~0.3 kg/khách
        
        # 5. Cập nhật tồn kho
        self.stock -= sold
        
        # 6. Tính doanh thu
        revenue = sold * self.current_price
        self.cum_revenue += revenue
        
        # 7. Cập nhật thống kê khách
        self.customers_arrived += demand / 0.3
        self.customers_bought += sold / 0.3
        
        # 8. Tính reward
        reward = (
            revenue     
            - 15000 * 0                          # Doanh thu
            - 20000 * lost_customers             # Phạt khách mất (20k/khách)
            - 500 * price_change                 # Phạt biến động giá
        )
        
        # 9. Chuyển sang step tiếp theo
        self.step_idx += 1
        
        # 10. Kiểm tra kết thúc episode (14 bước = 14 giờ = 1 ngày)
        terminated = self.step_idx >= 14
        
        if terminated:
            # Phạt lãng phí cuối ngày
            self.cum_waste = self.stock
            reward -= 15000 * self.cum_waste  # Phạt 15k/kg hỏng
        
        return self._get_obs(), reward, terminated, False, {}
    
    def _compute_demand(self) -> float:
        """
        Tính nhu cầu khách hàng dựa trên:
        - Giá hiện tại (elasticity)
        - Độ tươi của rau (age_days)
        - Thời tiết (mưa)
        - Nhu cầu dự kiến hôm nay
        
        Returns:
            float: Lượng khách mua (kg)
        """
        # Elasticity giá: giá cao → cầu thấp
        price_ratio = self.current_price / self.base_price
        elasticity = np.exp(-2.0 * (price_ratio - 1.0))
        
        # Rau cũ → ít người muốn
        freshness = max(0, 1 - 0.3 * self.age_days)
        
        # Nhu cầu dự kiến từ CSV
        expected = self.day_data["demand_ca_chua"]
        hourly = expected / 14  # Chia đều cho 14 giờ
        
        # Điều chỉnh theo thời tiết
        if self.day_data["is_rainy"]:
            hourly *= 0.75  # Mưa → ít khách
        
        # Nhu cầu cuối cùng = hourly × elasticity × tươi
        demand = hourly * elasticity * freshness
        
        # Thêm nhiễu ngẫu nhiên
        demand += np.random.normal(0, max(demand * 0.1, 0.1))
        
        return max(0, demand)
    
    def _get_obs(self) -> np.ndarray:
        """
        Lấy observation state vector (14 chiều)
        
        Returns:
            np.ndarray: State vector
        """
        return np.array([
            self.stock,                           # 0: tồn kho (kg)
            self.current_price / 1000,            # 1: giá hiện tại (1000đ)
            14 - self.step_idx,                   # 2: giờ còn lại
            self.day_data["weekday"],             # 3: ngày tuần (0-6)
            self.day_data["is_weekend"],          # 4: cuối tuần (0/1)
            self.day_data["is_rainy"],            # 5: mưa (0/1)
            self.day_data["temperature"],         # 6: nhiệt độ (°C)
            self.age_days,                        # 7: tuổi rau (0-2)
            self.cum_revenue / 1000,              # 8: doanh thu (1000đ)
            self.cum_waste,                       # 9: lãng phí (kg)
            self.customers_arrived,               # 10: khách đến
            self.customers_bought,                # 11: khách mua
            (self.cum_revenue / max(self.customers_bought, 1)) / 1000,  # 12: giá trung bình
            self.price_change_count,              # 13: số lần đổi giá
        ], dtype=np.float32)
    
    def render(self):
        """In thông tin state hiện tại"""
        print(f"Step {self.step_idx:2d} | "
              f"Stock: {self.stock:5.1f}kg | "
              f"Price: {self.current_price:7.0f}đ | "
              f"Revenue: {self.cum_revenue:10.0f}đ")


def test_environment():
    """Test environment chạy được"""
    print("Testing VegetableMarketEnv...\n")
    
    env = VegetableMarketEnv()
    obs, _ = env.reset()
    
    print("Environment initialized!")
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Initial obs: {obs}\n")
    
    print("Running 14 steps (1 full day):")
    total_reward = 0
    
    for step in range(14):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        
        discount_str = f"{env.action_to_discount[action]:.0%}"
        print(f"Step {step+1:2d}: Action {action} ({discount_str}) → "
              f"Reward {reward:10.0f}đ | "
              f"Stock {env.stock:5.1f}kg")
        
        if done:
            break
    
    print(f"\nEpisode Summary:")
    print(f"   Total Revenue:  {env.cum_revenue:10.0f}đ")
    print(f"   Waste:          {env.cum_waste:10.1f}kg")
    print(f"   Total Reward:   {total_reward:10.0f}đ")
    print(f"   Price changes:  {env.price_change_count:3d}")
    print(f"\nTest passed!\n")


if __name__ == "__main__":
    test_environment()