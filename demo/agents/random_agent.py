"""
Random Agent - Chọn hành động ngẫu nhiên
Dùng làm cận dưới (baseline mấu)
"""


class RandomAgent:
    """Agent chọn hành động ngẫu nhiên"""
    
    def __init__(self, env):
        """
        Args:
            env: Environment (cần có action_space.sample())
        """
        self.env = env
    
    def predict(self, obs):
        """
        Dự đoán hành động (chọn ngẫu nhiên)
        
        Args:
            obs: Observation (không dùng)
        
        Returns:
            action: Hành động ngẫu nhiên (0-4)
            None: Placeholder (để giống PPO interface)
        """
        action = self.env.action_space.sample()
        return action, None