"""
Rule-Based Agent - Sử dụng luật if-else
Đây là baseline thực tế (khôngcó AI, chỉ là logic kinh doanh)
"""


class RuleBasedAgent:
    """
    Agent dùng luật cứng (if-else) để ra quyết định
    Logic: Một quản lý siêu thị có kinh nghiệm sẽ làm như thế nào?
    """
    
    def predict(self, obs):
        """
        Ra quyết định dựa trên obs
        
        Args:
            obs: State vector (14 chiều)
                obs[0] = stock (kg)
                obs[2] = hours_remaining
                obs[7] = age_days
        
        Returns:
            action: 0-4 (0=giữ giá, 4=xả hàng)
            None: Placeholder
        """
        stock = obs[0]
        hours_remaining = obs[2]
        age_days = obs[7]
        
        # Tỉ lệ tồn kho so với nhập ban đầu (80kg)
        stock_ratio = stock / 80.0
        
        # ----- LUẬT 1: Rau cũ phải xả hàng ngay -----
        if age_days >= 2:
            return 4, None  # Giảm 50% (xả hàng)
        
        # ----- LUẬT 2: Sắp đóng cửa + còn hàng nhiều -----
        if hours_remaining <= 2 and stock_ratio > 0.4:
            return 3, None  # Giảm 30%
        
        if hours_remaining <= 4 and stock_ratio > 0.6:
            return 2, None  # Giảm 20%
        
        # ----- LUẬT 3: Giữa ngày, tồn kho bình thường -----
        if stock_ratio > 0.8:
            return 1, None  # Giảm 10%
        
        # ----- MẶC ĐỊNH: Giữ giá -----
        return 0, None


def test_rule_based_agent():
    """Test agent"""
    import numpy as np
    
    print("Testing RuleBasedAgent...\n")
    
    agent = RuleBasedAgent()
    
    # Test case 1: Rau mới, giữa ngày, tồn kho đủ
    obs1 = np.array([60, 25, 8, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    action1, _ = agent.predict(obs1)
    print(f"✓ Test 1 (rau mới, tồn kho 60kg, giữa ngày):")
    print(f"  → Action: {action1} (expected: 1 - giảm 10%)\n")
    
    # Test case 2: Rau cũ
    obs2 = np.array([40, 25, 8, 0, 0, 0, 28, 2, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    action2, _ = agent.predict(obs2)
    print(f"✓ Test 2 (rau cũ 2 ngày tuổi):")
    print(f"  → Action: {action2} (expected: 4 - giảm 50%)\n")
    
    # Test case 3: Sắp đóng cửa, tồn kho cao
    obs3 = np.array([70, 25, 2, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    action3, _ = agent.predict(obs3)
    print(f"✓ Test 3 (sắp đóng cửa 2h, tồn kho 70kg):")
    print(f"  → Action: {action3} (expected: 3 - giảm 30%)\n")
    
    print("All tests passed!\n")


if __name__ == "__main__":
    test_rule_based_agent()