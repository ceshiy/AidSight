#!/bin/bash
# AidSight 系统启动脚本

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "================================"
echo "  AidSight 系统启动"
echo "================================"

# 切换到项目目录
cd "$PROJECT_DIR"

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "错误: 虚拟环境不存在，请先运行 setup.sh 安装依赖"
    exit 1
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 检查昇腾 NPU（可选）
echo "检查硬件环境..."
if command -v npu-smi &> /dev/null; then
    if npu-smi info &> /dev/null; then
        echo "✓ 昇腾 NPU 已就绪"
    else
        echo "⚠ 未检测到昇腾 NPU，将使用 CPU 模式"
    fi
else
    echo "⚠ npu-smi 未安装，跳过 NPU 检查"
fi

# 检查配置文件
if [ ! -f "config.yaml" ]; then
    echo "错误: 配置文件 config.yaml 不存在"
    exit 1
fi

# 确保日志目录存在
mkdir -p logs

# 检查是否已在运行
PID_FILE="logs/aidsight.pid"
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "错误: AidSight 已在运行 (PID: $OLD_PID)"
        echo "如需重启，请先运行: ./scripts/stop_aidsight.sh"
        exit 1
    else
        # PID 文件存在但进程不在，清理旧文件
        rm -f "$PID_FILE"
    fi
fi

# 解析命令行参数
DAEMON=false
LOG_LEVEL="INFO"

while [[ $# -gt 0 ]]; do
    case $1 in
        --daemon)
            DAEMON=true
            shift
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --daemon         后台运行"
            echo "  --log-level LEVEL 日志级别 (DEBUG/INFO/WARNING/ERROR/CRITICAL)"
            echo "  --help           显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 启动系统
echo "正在启动 AidSight..."

if [ "$DAEMON" = true ]; then
    # 后台运行
    nohup python src/main.py \
        --config config.yaml \
        --log-level "$LOG_LEVEL" \
        > logs/stdout.log 2>&1 &
    
    PID=$!
    echo $PID > "$PID_FILE"
    echo "✓ AidSight 已在后台启动 (PID: $PID)"
    echo "  日志文件: logs/aidsight.log"
    echo "  标准输出: logs/stdout.log"
    echo "  停止服务: ./scripts/stop_aidsight.sh"
else
    # 前台运行
    python src/main.py \
        --config config.yaml \
        --log-level "$LOG_LEVEL"
fi

echo "================================"
