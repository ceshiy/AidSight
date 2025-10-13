#!/bin/bash
# AidSight 系统停止脚本

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "================================"
echo "  AidSight 系统停止"
echo "================================"

# 切换到项目目录
cd "$PROJECT_DIR"

PID_FILE="logs/aidsight.pid"

# 检查 PID 文件是否存在
if [ ! -f "$PID_FILE" ]; then
    echo "未找到运行中的 AidSight 实例"
    exit 0
fi

# 读取 PID
PID=$(cat "$PID_FILE")

# 检查进程是否存在
if ! ps -p "$PID" > /dev/null 2>&1; then
    echo "进程 $PID 不存在，清理 PID 文件"
    rm -f "$PID_FILE"
    exit 0
fi

# 停止进程
echo "正在停止 AidSight (PID: $PID)..."

# 首先尝试优雅停止 (SIGTERM)
kill -TERM "$PID" 2>/dev/null || true

# 等待进程退出（最多 10 秒）
for i in {1..10}; do
    if ! ps -p "$PID" > /dev/null 2>&1; then
        echo "✓ AidSight 已停止"
        rm -f "$PID_FILE"
        exit 0
    fi
    sleep 1
    echo -n "."
done

echo ""
echo "进程未响应 SIGTERM，尝试强制停止 (SIGKILL)..."

# 强制停止 (SIGKILL)
kill -KILL "$PID" 2>/dev/null || true

# 再次等待
sleep 2

if ! ps -p "$PID" > /dev/null 2>&1; then
    echo "✓ AidSight 已强制停止"
    rm -f "$PID_FILE"
else
    echo "✗ 无法停止进程 $PID"
    exit 1
fi

echo "================================"
