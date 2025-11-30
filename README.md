# Qwen-8B LoRA Fine-tuning

遵循标准化实验日志管理规范的 Qwen 模型 LoRA 微调代码。

## 🎯 实验日志规范

本项目严格遵循以下规范：

### 1️⃣ 配置解耦 (Config Decoupling)
- 所有超参数通过 `config.yaml` 配置
- 支持命令行参数覆盖
- 每次实验自动保存配置副本

### 2️⃣ 代码指纹 (Git Commit ID)
- 自动记录 Git Commit Hash
- 检测未提交的代码修改
- 日志开头打印代码版本

### 3️⃣ 标准化目录结构 (Structure)
```
work_dirs/
└── 20251130_qwen257binstruct_lora_r16_lr0.0001/
    ├── config.yaml          # 配置副本
    ├── log.txt              # 完整日志
    ├── train.py             # 代码备份
    ├── best_model/          # 最优权重
    │   ├── adapter_config.json
    │   └── adapter_model.safetensors
    ├── checkpoints/         # 训练检查点
    └── tb_logs/             # TensorBoard 日志
```

### 4️⃣ 日志内容核对表 (Checklist)
`log.txt` 开头包含：
1. **Command**: 完整运行指令
2. **Environment**: PyTorch/CUDA/GPU 信息
3. **Seed**: 随机种子
4. **Git Hash**: 代码版本号
5. **Config**: 所有超参数

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 准备数据
数据格式 (JSONL):
```json
{"instruction": "问题", "input": "可选输入", "output": "期望输出"}
```

### 3. 修改配置
编辑 `config.yaml` 设置模型路径、数据路径等。

### 4. 开始训练
```bash
# 使用默认配置
python train_qwen8b_lora.py --config config.yaml

# 命令行覆盖参数
python train_qwen8b_lora.py --config config.yaml --lr 5e-5 --epochs 3

# 自定义实验名称
python train_qwen8b_lora.py --config config.yaml --exp_name my_experiment
```

## 📊 SwanLab 监控

训练过程会自动记录到 SwanLab：
- 训练 Loss
- 学习率变化
- 验证指标

查看训练过程：
```bash
swanlab watch
```

## ⚙️ 配置说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| batch_size | 8 | 批次大小 |
| learning_rate | 1e-4 | 学习率 |
| n_epochs | 2 | 训练轮数 |
| lora_rank | 16 | LoRA 秩 |
| lora_alpha | 32 | LoRA alpha |
| lora_dropout | 0.15 | LoRA dropout |
| max_length | 8192 | 最大序列长度 |
| seed | 42 | 随机种子 |

## 📝 复现实验

找到你想复现的实验目录，例如 `work_dirs/20251130_xxx/`:

```bash
# 1. 恢复代码版本
git checkout <log.txt 中的 Git Hash>

# 2. 使用保存的配置
python train_qwen8b_lora.py --config work_dirs/20251130_xxx/config.yaml
```

## 🔄 多组超参数实验

### 配置超参数搜索

编辑 `sweep_config.yaml`：

```yaml
# 网格搜索模式
search_mode: "grid_search"

grid_search:
  learning_rate: [1.0e-4, 5.0e-5, 1.0e-5]
  lora_rank: [8, 16, 32]
  batch_size: [4, 8]
  n_epochs: [2, 3]
```

或者手动指定实验组：

```yaml
search_mode: "manual"

manual_experiments:
  - name: "baseline"
    learning_rate: 1.0e-4
    lora_rank: 16
    batch_size: 8
    n_epochs: 2

  - name: "high_rank"
    learning_rate: 5.0e-5
    lora_rank: 64
    batch_size: 4
    n_epochs: 3
```

### 运行多组实验

```bash
# 预览实验计划（不实际运行）
python run_sweep.py --config sweep_config.yaml --dry-run

# 开始运行所有实验
python run_sweep.py --config sweep_config.yaml

# 从第 5 个实验开始（断点续跑）
python run_sweep.py --config sweep_config.yaml --start-from 5
```

### 实验结果汇总

运行完成后，自动生成：
- `work_dirs/sweep_summary.md` - Markdown 汇总报告
- `work_dirs/sweep_results.json` - JSON 格式完整结果

报告包含：
- 所有实验的状态、耗时、Final Loss
- 最佳实验的配置详情
- 失败实验的错误信息

## 📁 项目结构

```
qwen8b_lora/
├── config.yaml              # 单次实验配置
├── sweep_config.yaml        # 多实验超参数搜索配置
├── train_qwen8b_lora.py     # 训练脚本
├── run_sweep.py             # 多实验运行器
├── requirements.txt         # 依赖列表
├── data/
│   └── train.jsonl          # 训练数据
└── work_dirs/               # 实验输出目录
    ├── sweep_summary.md     # 汇总报告
    ├── sweep_results.json   # 完整结果
    └── 20251130_xxx/        # 各实验目录
```
