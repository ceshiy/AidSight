#!/bin/bash
# 将 AidSight 安装为系统服务（systemd）

set -e

# 检查是否以 root 运行
if [ "$EUID" -ne 0 ]; then
    echo "错误: 此脚本需要 root 权限"
    echo "请使用: sudo $0"
    exit 1
fi

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "================================"
echo "  安装 AidSight 系统服务"
echo "================================"

# 获取当前用户（实际运行 sudo 的用户）
ACTUAL_USER="${SUDO_USER:-$USER}"
ACTUAL_HOME=$(eval echo ~"$ACTUAL_USER")

echo "项目目录: $PROJECT_DIR"
echo "运行用户: $ACTUAL_USER"

# 创建 systemd 服务文件
SERVICE_FILE="/etc/systemd/system/aidsight.service"

echo "创建服务文件: $SERVICE_FILE"

cat > "$SERVICE_FILE" << EOF
[Unit]
Description=AidSight - 视障辅助导航系统
After=network.target

[Service]
Type=simple
User=$ACTUAL_USER
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$PROJECT_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$PROJECT_DIR/venv/bin/python $PROJECT_DIR/src/main.py --config $PROJECT_DIR/config.yaml
ExecStop=/bin/kill -TERM \$MAINPID
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# 设置文件权限
chmod 644 "$SERVICE_FILE"

# 重新加载 systemd
echo "重新加载 systemd..."
systemctl daemon-reload

# 启用服务（开机自启）
echo "启用服务（开机自启）..."
systemctl enable aidsight.service

echo ""
echo "✓ AidSight 服务安装成功！"
echo ""
echo "常用命令:"
echo "  启动服务:   sudo systemctl start aidsight"
echo "  停止服务:   sudo systemctl stop aidsight"
echo "  重启服务:   sudo systemctl restart aidsight"
echo "  查看状态:   sudo systemctl status aidsight"
echo "  查看日志:   sudo journalctl -u aidsight -f"
echo "  禁用自启:   sudo systemctl disable aidsight"
echo ""
echo "================================"
