#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
/media/hairmore/LaCie/qwen8b_loraExperiment_RuleExe 硬盘上的代码路径
conda activate qwen8b_lora 进入虚拟环境
Qwen-8B LoRA Fine-tuning Script
================================
遵循实验日志管理规范：
1. 配置解耦 (Config Decoupling)
2. 代码指纹 (Git Commit ID)
3. 标准化目录结构 (Structure)
4. 日志内容核对表 (Checklist)
"""


import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
import sys
import yaml
import json
import shutil
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,   # ✅ 改成 LM collator
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import transformers
from transformers import TrainingArguments

print("[DEBUG] transformers version:", transformers.__version__)
print("[DEBUG] TrainingArguments from:", TrainingArguments.__module__, TrainingArguments.__qualname__)

# ============================================
# 第二步：代码指纹 (Git Commit ID)
# ============================================
def get_git_info():
    """获取当前 Git 信息"""
    git_info = {
        "commit_hash": "unknown",
        "branch": "unknown",
        "is_dirty": False,
    }
    try:
        # 获取 commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            git_info["commit_hash"] = result.stdout.strip()
        
        # 获取分支名
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            git_info["branch"] = result.stdout.strip()
        
        # 检查是否有未提交的修改
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            git_info["is_dirty"] = len(result.stdout.strip()) > 0
            
    except Exception as e:
        print(f"Warning: Failed to get git info: {e}")
    
    return git_info


def get_environment_info():
    """获取运行环境信息"""
    env_info = {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": "N/A",
        "gpu_name": "N/A",
        "gpu_count": 0,
    }
    
    if torch.cuda.is_available():
        env_info["cuda_version"] = torch.version.cuda
        env_info["gpu_count"] = torch.cuda.device_count()
        env_info["gpu_name"] = torch.cuda.get_device_name(0)
    
    return env_info


# ============================================
# 第三步：标准化目录结构 (Structure)
# ============================================
def setup_experiment_dir(config, args):
    """
    创建实验目录，命名规范：时间戳_模型名称_关键改动
    
    目录结构：
    work_dirs/
    └── 20251126_qwen8b_lora_lr1e-4/
        ├── config.yaml          # 当时的配置副本
        ├── log.txt              # 完整的训练日志
        ├── best_model/          # 最优权重
        └── train.py             # 主代码备份
    """
    work_dir = Path(config["output"]["work_dir"])
    
    # 生成实验名称
    if config["output"].get("experiment_name"):
        exp_name = config["output"]["experiment_name"]
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = config["model"]["name"].split("/")[-1].lower().replace("-", "")
        lr_str = f"lr{config['training']['learning_rate']}"
        exp_name = f"{timestamp}_{model_short}_lora_r{config['lora']['rank']}_{lr_str}"
    
    exp_dir = work_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置副本
    config_copy_path = exp_dir / "config.yaml"
    with open(config_copy_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    # 备份训练脚本
    script_path = Path(__file__).resolve()
    if script_path.exists():
        shutil.copy(script_path, exp_dir / "train.py")
    
    return exp_dir, exp_name


# ============================================
# 第四步：日志内容核对表 (Checklist)
# ============================================
def write_experiment_header(log_file, config, args, git_info, env_info):
    """
    log.txt 开头必须包含以下信息，缺一不可：
    1. Command: 运行这行代码的完整指令
    2. Environment: PyTorch 版本、CUDA 版本、显卡型号
    3. Seed: 当前使用的随机种子
    4. Git Hash: 代码版本号
    5. Config: 所有超参数的列表
    """
    header = []
    header.append("=" * 60)
    header.append("EXPERIMENT LOG")
    header.append("=" * 60)
    header.append(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    header.append("")
    
    # 1. Command
    header.append("-" * 40)
    header.append("1. COMMAND")
    header.append("-" * 40)
    header.append(f"   {' '.join(sys.argv)}")
    header.append("")
    
    # 2. Environment
    header.append("-" * 40)
    header.append("2. ENVIRONMENT")
    header.append("-" * 40)
    header.append(f"   Python Version:       {env_info['python_version'].split()[0]}")
    header.append(f"   PyTorch Version:      {env_info['pytorch_version']}")
    header.append(f"   Transformers Version: {env_info['transformers_version']}")
    header.append(f"   CUDA Version:         {env_info['cuda_version']}")
    header.append(f"   GPU Name:             {env_info['gpu_name']}")
    header.append(f"   GPU Count:            {env_info['gpu_count']}")
    header.append("")
    
    # 3. Seed
    header.append("-" * 40)
    header.append("3. SEED")
    header.append("-" * 40)
    header.append(f"   Random Seed: {config['seed']}")
    header.append("")
    
    # 4. Git Hash
    header.append("-" * 40)
    header.append("4. GIT HASH (Code Version)")
    header.append("-" * 40)
    dirty_flag = " (DIRTY - uncommitted changes!)" if git_info["is_dirty"] else ""
    header.append(f"   Commit: {git_info['commit_hash']}{dirty_flag}")
    header.append(f"   Branch: {git_info['branch']}")
    header.append("")
    
    # 5. Config
    header.append("-" * 40)
    header.append("5. CONFIG (All Hyperparameters)")
    header.append("-" * 40)
    header.append(yaml.dump(config, allow_unicode=True, default_flow_style=False))
    header.append("")
    header.append("=" * 60)
    header.append("TRAINING LOG")
    header.append("=" * 60)
    header.append("")
    
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(header))
    
    # 同时打印到控制台
    print("\n".join(header))


# ============================================
# 数据处理
# ============================================
def load_and_preprocess_data(config, tokenizer):
    """加载并预处理训练数据，支持自动切分验证集"""
    from datasets import DatasetDict
    
    # 加载数据
    data_files = {"train": config["data"]["train_file"]}
    
    # 检查是否有独立的验证集文件
    eval_file = config["data"].get("eval_file")
    has_separate_eval = eval_file and Path(eval_file).exists()
    
    if has_separate_eval:
        data_files["eval"] = eval_file
    
    dataset = load_dataset("json", data_files=data_files)
    
    # 如果没有独立验证集，且配置了切分
    split_config = config["data"].get("split", {})
    if not has_separate_eval and split_config.get("enabled", False):
        eval_ratio = split_config.get("eval_ratio", 0.1)
        split_seed = split_config.get("seed", 42)
        
        # 切分训练集
        split_result = dataset["train"].train_test_split(
            test_size=eval_ratio,
            seed=split_seed,
            shuffle=True
        )
        
        # 重新构建 DatasetDict，将 "test" 重命名为 "eval"
        dataset = DatasetDict({
            "train": split_result["train"],
            "eval": split_result["test"]
        })
        
        print(f"📊 Data split: {len(dataset['train'])} train / {len(dataset['eval'])} eval "
              f"(ratio={eval_ratio}, seed={split_seed})")
    
    max_length = config["data"]["max_length"]
    
    def preprocess_function(examples):
        """
        支持两种格式：
        1) {"instruction": "...", "input": "...", "output": "..."}
        2) {"messages": [{"role": "system/user/assistant", "content": "..."}, ...]}
        """
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        # 明确 batch size
        if "instruction" in examples:
            batch_size = len(examples["instruction"])
        elif "messages" in examples:
            batch_size = len(examples["messages"])
        else:
            return model_inputs
        
        for i in range(batch_size):
            # 格式1: instruction/input/output
            if "instruction" in examples:
                instruction = examples["instruction"][i]
                input_list = examples.get("input")
                if input_list is not None:
                    input_text = input_list[i]
                else:
                    input_text = ""
                output_text = examples["output"][i]
                
                if input_text:
                    prompt = (
                        f"<|im_start|>user\n{instruction}\n{input_text}"
                        f"<|im_end|>\n<|im_start|>assistant\n"
                    )
                else:
                    prompt = (
                        f"<|im_start|>user\n{instruction}"
                        f"<|im_end|>\n<|im_start|>assistant\n"
                    )
                response = f"{output_text}<|im_end|>"
            
            # 格式2: messages（支持 system/user/assistant）
            elif "messages" in examples:
                messages = examples["messages"][i]
                prompt = ""
                response = ""
                
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    
                    if role == "system":
                        # System prompt 放在最前面
                        prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
                    elif role == "user":
                        prompt += f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
                    elif role == "assistant":
                        # 只取最后一条 assistant 作为 supervised target
                        response = f"{content}<|im_end|>"
                
                if not response:
                    # 没有 assistant，跳过
                    continue
            else:
                continue
            
            # Tokenize
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            response_ids = tokenizer.encode(response, add_special_tokens=False)
            
            input_ids = prompt_ids + response_ids
            labels = [-100] * len(prompt_ids) + response_ids  # 只计算 response 的 loss
            
            # 截断
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]
            
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
        
        return model_inputs
    
    # 处理数据集
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Preprocessing dataset",
    )
    
    return processed_dataset


# ============================================
# SwanLab 回调
# ============================================
class SwanLabCallback(transformers.TrainerCallback):
    """SwanLab 训练监控回调"""
    
    def __init__(self, swanlab_run):
        self.run = swanlab_run
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        log_dict = {}
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                log_dict[key] = value
        
        if log_dict and self.run is not None:
            self.run.log(log_dict, step=state.global_step)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and self.run is not None:
            eval_metrics = {
                f"eval/{k}": v for k, v in metrics.items()
                if isinstance(v, (int, float))
            }
            self.run.log(eval_metrics, step=state.global_step)


# ============================================
# 主训练函数
# ============================================
def train(config, args):
    """主训练流程"""
    
    # 获取环境和 Git 信息
    git_info = get_git_info()
    env_info = get_environment_info()
    
    # 打印代码版本（规范要求）
    print(f"\n{'='*60}")
    print(f"Current Code Version: {git_info['commit_hash']}")
    print(f"{'='*60}\n")
    
    # 设置实验目录
    exp_dir, exp_name = setup_experiment_dir(config, args)
    print(f"Experiment directory: {exp_dir}")
    
    # 设置日志文件
    log_file = exp_dir / "log.txt"
    write_experiment_header(log_file, config, args, git_info, env_info)
    
    # 设置随机种子
    transformers.set_seed(config["seed"])
    
    # 初始化 SwanLab
    swanlab_run = None
    swanlab_callback = None
    if config["swanlab"]["enabled"]:
        try:
            import swanlab
            swanlab_run = swanlab.init(
                project=config["swanlab"]["project"],
                experiment_name=config["swanlab"].get("experiment_name") or exp_name,
                config=config,
            )
            swanlab_callback = SwanLabCallback(swanlab_run)
            print(f"SwanLab initialized: {config['swanlab']['project']}/{exp_name}")
        except ImportError:
            print("Warning: swanlab not installed. Run: pip install swanlab")
        except Exception as e:
            print(f"Warning: Failed to initialize SwanLab: {e}")
    
    # 加载模型和 tokenizer
    print(f"Loading model: {config['model']['name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"],
        trust_remote_code=config["model"]["trust_remote_code"],
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        trust_remote_code=config["model"]["trust_remote_code"],
        torch_dtype=torch.float16 if config["training"]["fp16"] else torch.float32,
        device_map="auto",
    )
    
    # 配置 LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["lora"]["rank"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 启用 gradient checkpointing
    if config["training"]["gradient_checkpointing"]:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    
    # 加载数据
    print("Loading and preprocessing data...")
    dataset = load_and_preprocess_data(config, tokenizer)
    
    # 配置训练参数（⚠ eval_strategy → evaluation_strategy）
    has_eval = "eval" in dataset

    training_args = TrainingArguments(
        output_dir=str(exp_dir / "checkpoints"),
        num_train_epochs=config["training"]["n_epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        warmup_ratio=config["training"]["warmup_ratio"],
        weight_decay=config["training"]["weight_decay"],
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        eval_steps=config["training"]["eval_steps"] if has_eval else None,
        evaluation_strategy="steps" if has_eval else "no",   # ✅ 正确参数名
        save_total_limit=3,
        load_best_model_at_end=True if has_eval else False,
        fp16=config["training"]["fp16"],
        report_to=[],  # 禁用默认的 wandb 等
        logging_dir=str(exp_dir / "tb_logs"),
        seed=config["seed"],
    )
    
    # Data collator：Causal LM 标准写法
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 设置回调
    callbacks = []
    if swanlab_callback:
        callbacks.append(swanlab_callback)
    
    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("eval") if has_eval else None,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    # 开始训练
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    trainer.train()
    
    # 保存最终模型 / 最优模型
    best_model_path = exp_dir / "best_model"
    trainer.save_model(str(best_model_path))
    tokenizer.save_pretrained(str(best_model_path))
    print(f"\nBest model saved to: {best_model_path}")
    
    # 结束 SwanLab
    if swanlab_run:
        swanlab_run.finish()
    
    # 写入训练完成日志
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Best model saved to: {best_model_path}\n")
        f.write(f"{'='*60}\n")
    
    print(f"\n✅ Training completed! Check {exp_dir} for all outputs.")


# ============================================
# 主入口
# ============================================
def main():
    parser = argparse.ArgumentParser(description="Qwen-8B LoRA Fine-tuning")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    # 允许命令行覆盖配置
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--exp_name", type=str, help="Custom experiment name")
    
    args = parser.parse_args()
    
    # 加载配置文件
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 命令行参数覆盖
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.epochs:
        config["training"]["n_epochs"] = args.epochs
    if args.seed:
        config["seed"] = args.seed
    if args.exp_name:
        config["output"]["experiment_name"] = args.exp_name
    
    # 开始训练
    train(config, args)


if __name__ == "__main__":
    main()
