#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多组超参数实验运行器
====================
支持网格搜索和手动指定实验组，自动运行并汇总结果。

遵循实验日志规范：
- 每组实验独立目录
- 自动记录所有配置
- 生成汇总对比报告
"""

import os
import sys
import yaml
import json
import time
import itertools
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import copy


def load_sweep_config(config_path: str) -> Dict:
    """加载超参数搜索配置"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_grid_experiments(sweep_config: Dict) -> List[Dict]:
    """
    网格搜索：生成所有参数组合
    """
    grid_params = sweep_config["grid_search"]
    fixed_params = sweep_config.get("fixed_params", {})
    
    # 获取所有参数名和值列表
    param_names = list(grid_params.keys())
    param_values = [grid_params[name] for name in param_names]
    
    # 生成所有组合
    experiments = []
    for combo in itertools.product(*param_values):
        exp_params = dict(zip(param_names, combo))
        
        # 应用固定参数
        if "lora_dropout" not in exp_params and "lora_dropout" in fixed_params:
            exp_params["lora_dropout"] = fixed_params["lora_dropout"]
        if "batch_size" not in exp_params and "batch_size" in fixed_params:
            exp_params["batch_size"] = fixed_params["batch_size"]
        if "seed" not in exp_params and "seed" in fixed_params:
            exp_params["seed"] = fixed_params["seed"]
        if "weight_decay" not in exp_params and "weight_decay" in fixed_params:
            exp_params["weight_decay"] = fixed_params["weight_decay"]
        
        # 自动计算 lora_alpha (如果配置了 ratio)
        if "lora_alpha" not in exp_params and "lora_alpha_ratio" in fixed_params:
            exp_params["lora_alpha"] = exp_params.get("lora_rank", 16) * fixed_params["lora_alpha_ratio"]
        
        # 生成实验名称
        name_parts = []
        if "learning_rate" in exp_params:
            name_parts.append(f"lr{exp_params['learning_rate']}")
        if "lora_rank" in exp_params:
            name_parts.append(f"r{exp_params['lora_rank']}")
        if "lora_alpha" in exp_params:
            name_parts.append(f"a{exp_params['lora_alpha']}")
        if "batch_size" in exp_params:
            name_parts.append(f"bs{exp_params['batch_size']}")
        if "n_epochs" in exp_params:
            name_parts.append(f"ep{exp_params['n_epochs']}")
        
        exp_params["name"] = "_".join(name_parts)
        experiments.append(exp_params)
    
    return experiments


def generate_manual_experiments(sweep_config: Dict) -> List[Dict]:
    """
    手动模式：直接使用指定的实验组
    """
    return sweep_config["manual_experiments"]


def merge_config(base_config: Dict, exp_params: Dict) -> Dict:
    """
    合并基础配置和实验参数
    """
    config = copy.deepcopy(base_config)
    
    # 映射实验参数到配置结构
    param_mapping = {
        "learning_rate": ("training", "learning_rate"),
        "batch_size": ("training", "batch_size"),
        "n_epochs": ("training", "n_epochs"),
        "weight_decay": ("training", "weight_decay"),
        "lora_rank": ("lora", "rank"),
        "lora_alpha": ("lora", "alpha"),
        "lora_dropout": ("lora", "dropout"),
        "seed": (None, "seed"),
    }
    
    for param_name, value in exp_params.items():
        if param_name == "name":
            continue
        if param_name in param_mapping:
            section, key = param_mapping[param_name]
            if section is None:
                config[key] = value
            else:
                if section not in config:
                    config[section] = {}
                config[section][key] = value
    
    # 设置实验名称
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["output"]["experiment_name"] = f"{timestamp}_{exp_params['name']}"
    
    # SwanLab 实验名称
    if config.get("swanlab", {}).get("enabled"):
        config["swanlab"]["experiment_name"] = exp_params["name"]
    
    return config


def run_single_experiment(config: Dict, exp_index: int, total: int, script_path: str) -> Dict:
    """
    运行单个实验
    """
    exp_name = config["output"]["experiment_name"]
    print(f"\n{'='*60}")
    print(f"🚀 Running Experiment [{exp_index + 1}/{total}]: {exp_name}")
    print(f"{'='*60}")
    
    # 保存临时配置文件
    temp_config_path = Path(script_path).parent / f".temp_config_{exp_index}.yaml"
    with open(temp_config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    # 记录开始时间
    start_time = time.time()
    
    # 运行训练脚本
    result = {
        "name": exp_name,
        "config": config,
        "status": "unknown",
        "duration": 0,
        "error": None,
        "metrics": {},
    }
    
    try:
        process = subprocess.run(
            [sys.executable, script_path, "--config", str(temp_config_path)],
            capture_output=False,  # 直接输出到终端
            text=True,
        )
        
        if process.returncode == 0:
            result["status"] = "success"
            print(f"✅ Experiment {exp_name} completed successfully!")
        else:
            result["status"] = "failed"
            result["error"] = f"Return code: {process.returncode}"
            print(f"❌ Experiment {exp_name} failed!")
            
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        print(f"❌ Experiment {exp_name} error: {e}")
    
    # 记录耗时
    result["duration"] = time.time() - start_time
    
    # 清理临时配置
    if temp_config_path.exists():
        temp_config_path.unlink()
    
    # 尝试读取训练结果
    exp_dir = Path(config["output"]["work_dir"]) / exp_name
    trainer_state_path = exp_dir / "checkpoints" / "trainer_state.json"
    if trainer_state_path.exists():
        try:
            with open(trainer_state_path, "r") as f:
                trainer_state = json.load(f)
                if trainer_state.get("best_metric"):
                    result["metrics"]["best_metric"] = trainer_state["best_metric"]
                if trainer_state.get("log_history"):
                    # 获取最后的训练 loss
                    for log in reversed(trainer_state["log_history"]):
                        if "loss" in log:
                            result["metrics"]["final_loss"] = log["loss"]
                            break
        except Exception:
            pass
    
    return result


def generate_summary_report(results: List[Dict], output_dir: Path):
    """
    生成实验汇总报告
    """
    report_path = output_dir / "sweep_summary.md"
    
    lines = []
    lines.append("# 超参数搜索实验汇总报告")
    lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 统计
    total = len(results)
    success = sum(1 for r in results if r["status"] == "success")
    failed = total - success
    
    lines.append(f"## 📊 总体统计\n")
    lines.append(f"- 总实验数: {total}")
    lines.append(f"- 成功: {success}")
    lines.append(f"- 失败: {failed}")
    lines.append(f"- 总耗时: {sum(r['duration'] for r in results) / 3600:.2f} 小时\n")
    
    # 结果表格
    lines.append("## 📋 实验结果\n")
    lines.append("| 实验名称 | 状态 | 耗时 | Final Loss | LR | Rank | Dropout | WD | Epochs |")
    lines.append("|----------|------|------|------------|-----|------|---------|-----|--------|")
    
    for r in results:
        status_icon = "✅" if r["status"] == "success" else "❌"
        duration = f"{r['duration']/60:.1f}min"
        final_loss = r["metrics"].get("final_loss", "N/A")
        if isinstance(final_loss, float):
            final_loss = f"{final_loss:.4f}"
        
        # 提取参数
        lr = r["config"]["training"].get("learning_rate", "N/A")
        rank = r["config"]["lora"].get("rank", "N/A")
        dropout = r["config"]["lora"].get("dropout", "N/A")
        wd = r["config"]["training"].get("weight_decay", "N/A")
        epochs = r["config"]["training"].get("n_epochs", "N/A")
        
        lines.append(f"| {r['name'][:25]} | {status_icon} | {duration} | {final_loss} | {lr} | {rank} | {dropout} | {wd} | {epochs} |")
    
    # 最佳实验
    successful_results = [r for r in results if r["status"] == "success" and r["metrics"].get("final_loss")]
    if successful_results:
        best = min(successful_results, key=lambda x: x["metrics"]["final_loss"])
        lines.append(f"\n## 🏆 最佳实验\n")
        lines.append(f"**{best['name']}**")
        lines.append(f"- Final Loss: {best['metrics']['final_loss']:.4f}")
        lines.append(f"- 配置:")
        lines.append(f"  - Learning Rate: {best['config']['training']['learning_rate']}")
        lines.append(f"  - LoRA Rank: {best['config']['lora']['rank']}")
        lines.append(f"  - LoRA Alpha: {best['config']['lora']['alpha']}")
        lines.append(f"  - LoRA Dropout: {best['config']['lora'].get('dropout', 'N/A')}")
        lines.append(f"  - Weight Decay: {best['config']['training'].get('weight_decay', 'N/A')}")
        lines.append(f"  - Batch Size: {best['config']['training']['batch_size']}")
        lines.append(f"  - Epochs: {best['config']['training']['n_epochs']}")
    
    # 失败实验
    failed_results = [r for r in results if r["status"] != "success"]
    if failed_results:
        lines.append(f"\n## ⚠️ 失败实验\n")
        for r in failed_results:
            lines.append(f"- **{r['name']}**: {r.get('error', 'Unknown error')}")
    
    # 写入报告
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"\n📝 Summary report saved to: {report_path}")
    
    # 同时保存 JSON 格式结果
    json_path = output_dir / "sweep_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    return report_path


def check_completed_experiments(work_dir: Path, experiments: List[Dict]) -> List[Dict]:
    """
    检查已完成的实验，返回未完成的实验列表
    """
    pending = []
    for exp in experiments:
        # 检查是否有匹配的实验目录
        exp_pattern = f"*_{exp['name']}"
        matching_dirs = list(work_dir.glob(exp_pattern))
        
        # 检查是否有 best_model 目录（表示训练完成）
        completed = False
        for d in matching_dirs:
            if (d / "best_model").exists():
                completed = True
                print(f"⏭️  Skipping completed experiment: {exp['name']}")
                break
        
        if not completed:
            pending.append(exp)
    
    return pending


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="多组超参数实验运行器")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="sweep_config.yaml",
        help="超参数搜索配置文件"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印实验计划，不实际运行"
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="从第几个实验开始（用于断点续跑）"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    sweep_config = load_sweep_config(args.config)
    base_config = sweep_config["base_config"]
    run_control = sweep_config["run_control"]
    
    # 生成实验列表
    if sweep_config["search_mode"] == "grid_search":
        experiments = generate_grid_experiments(sweep_config)
        print(f"📊 Grid Search Mode: {len(experiments)} experiments generated")
    else:
        experiments = generate_manual_experiments(sweep_config)
        print(f"📋 Manual Mode: {len(experiments)} experiments specified")
    
    # 限制实验数量
    max_exp = run_control.get("max_experiments", 100)
    if len(experiments) > max_exp:
        print(f"⚠️  Limiting to first {max_exp} experiments (from {len(experiments)})")
        experiments = experiments[:max_exp]
    
    # 跳过已完成的实验
    work_dir = Path(base_config["output"]["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    
    if run_control.get("skip_completed", True):
        experiments = check_completed_experiments(work_dir, experiments)
        print(f"📝 {len(experiments)} experiments remaining")
    
    # 从指定位置开始
    if args.start_from > 0:
        experiments = experiments[args.start_from:]
        print(f"⏩ Starting from experiment {args.start_from}")
    
    if not experiments:
        print("✅ All experiments completed!")
        return
    
    # 打印实验计划
    print(f"\n{'='*60}")
    print("📋 Experiment Plan")
    print(f"{'='*60}")
    for i, exp in enumerate(experiments):
        print(f"  [{i+1}] {exp['name']}")
        print(f"      lr={exp.get('learning_rate', 'default')}, "
              f"rank={exp.get('lora_rank', 'default')}, "
              f"bs={exp.get('batch_size', 'default')}, "
              f"epochs={exp.get('n_epochs', 'default')}")
    
    if args.dry_run:
        print(f"\n🔍 Dry run complete. {len(experiments)} experiments would be run.")
        return
    
    # 确认运行
    print(f"\n{'='*60}")
    print(f"🚀 Ready to run {len(experiments)} experiments")
    print(f"{'='*60}")
    
    response = input("Continue? [y/N]: ").strip().lower()
    if response != "y":
        print("Aborted.")
        return
    
    # 运行实验
    script_path = Path(__file__).parent / "train_qwen8b_lora.py"
    results = []
    
    for i, exp_params in enumerate(experiments):
        # 合并配置
        config = merge_config(base_config, exp_params)
        
        # 运行实验
        result = run_single_experiment(config, i, len(experiments), str(script_path))
        results.append(result)
        
        # 检查是否继续
        if result["status"] != "success" and not run_control.get("continue_on_failure", True):
            print("❌ Stopping due to failure (continue_on_failure=false)")
            break
        
        # 实验间隔
        if i < len(experiments) - 1:
            interval = run_control.get("interval_seconds", 30)
            print(f"⏳ Waiting {interval}s before next experiment...")
            time.sleep(interval)
    
    # 生成汇总报告
    generate_summary_report(results, work_dir)
    
    print(f"\n{'='*60}")
    print("🎉 All experiments completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
