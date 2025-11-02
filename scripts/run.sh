#!/bin/bash
set -e

# 简化配置：直接在脚本内指定核心路径（无需yq解析YAML）
CONFIG_PATH="../configs/config.yaml"
TEST_MODE=false
# 手动指定结果目录（与config.yaml中的best_model_path一致）
RESULTS_DIR="../results/results1"

# 解析命令行参数
while [ $# -gt 0 ]; do
  case $1 in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --test)
      TEST_MODE=true
      shift 1
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

# 检查配置文件是否存在
if [ ! -f "$CONFIG_PATH" ]; then
  echo "配置文件不存在: $CONFIG_PATH"
  exit 1
fi

# 创建结果目录（如果不存在）
mkdir -p "$RESULTS_DIR"

# 执行脚本
if [ "$TEST_MODE" = true ]; then
  echo "=== 执行测试模式（使用配置：$CONFIG_PATH）==="
  python main.py --config "$CONFIG_PATH" --test
else
  echo "=== 执行训练模式（使用配置：$CONFIG_PATH）==="
  python main.py --config "$CONFIG_PATH"
fi