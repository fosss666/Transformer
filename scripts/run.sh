#!/bin/bash
set -e

CONFIG_PATH="../configs/base.yaml"
TEST_MODE=false
# 手动指定结果目录
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

# 创建结果目录
mkdir -p "$RESULTS_DIR"

# 先运行 data.py 生成/更新词汇表
echo "=== 第一步：运行 data.py 准备数据并生成词汇表 ==="
python data.py --config "$CONFIG_PATH"

# 执行 main.py
if [ "$TEST_MODE" = true ]; then
  echo "=== 第二步：执行测试模式==="
  python main.py --config "$CONFIG_PATH" --test
else
  echo "=== 第二步：执行训练模式==="
  python main.py --config "$CONFIG_PATH"
fi