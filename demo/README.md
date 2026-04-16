1. Mở terminal (Ctrl + `)

2. tạo virtual environment
- lệnh:  python -m venv venv

3. Kích hoạt môi trường ảo
- lệnh: venv\Scripts\activate
- Kiểm tra kích hoạt:
Terminal sẽ hiển thị (venv) ở đầu dòng
Ví dụ: (venv) C:\Users\YourName\Documents\agent-multiagent-for-jp>
4. Setup VS dùng môi trường ảo này

-  Mở Command Palette (Ctrl + Shift + P)
- Gõ "Python: Select Interpreter"
- Chọn "./venv/bin/python" (hoặc ".\venv\Scripts\python.exe" trên Windows)
- VSCode sẽ dùng Python từ venv này

5. Cài đặt dependencies
-  Nâng cấp pip 
python -m pip install --upgrade pip
- Cài các thư viện cần thiết
pip install -r requirements.txt
- Kiểm tra cài đặt
python -c "import gymnasium, stable_baselines3, numpy, pandas; print('✅ All libraries installed successfully!')"