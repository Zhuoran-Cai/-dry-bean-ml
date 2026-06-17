@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在启动干豆多分类展示界面...
echo 使用端口 8505（避免与 8501 上其他项目冲突）
echo.
python -m streamlit run app.py --server.port 8505
if errorlevel 1 (
    echo.
    echo 启动失败。请确认已安装依赖: pip install -r requirements.txt
    pause
)
