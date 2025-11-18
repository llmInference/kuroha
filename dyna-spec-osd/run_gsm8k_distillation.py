#!/usr/bin/env python3
"""
SGLang GSM8K在线蒸馏测试脚本

基于OSD原始脚本的超参数配置

使用方法:
# 基本运行（使用默认OSD参数）
python run_gsm8k_distillation.py --model_path Qwen/Qwen3-8B --data_path data/gsm8k_test.jsonl --output_dir results/

# 完全匹配OSD online.sh参数
python run_gsm8k_distillation.py \
    --model_path Qwen/Qwen3-8B \
    --speculative_draft_model_path lkeab/infinitebench-eagle-3-8b \
    --data_path data/gsm8k_test.jsonl \
    --output_dir results/ \
    --max_propose_num 5 \
    --online_update_interval 8 \
    --online_eval_interval 1 \
    --distillation_lr 1e-4 \
    --kl_method forward \
    --sample_source student \
    --focal_gamma 0.0 \
    --online_distillation_enable True \
    --max_samples 1000

# OSD参数对应关系:
# max_propose_num 5           → --max_propose_num 5
# online_update_interval 8    → --online_update_interval 8  
# online_eval_interval 1      → --online_eval_interval 1
# learning_rate 1e-4          → --distillation_lr 1e-4
# mode online                 → --online_distillation_enable True
# kl_method forward           → --kl_method forward
# sample_source student       → --sample_source student
# focal_loss_gamma 0.0        → --focal_gamma 0.0

SGLang在线蒸馏参数:
--online_distillation_enable: 启用在线蒸馏
--online_distillation_buffer_size: 缓冲区大小
--online_distillation_update_interval: 更新间隔
--online_distillation_learning_rate: 学习率
--online_distillation_temperature: 蒸馏温度
--online_distillation_kl_weight: KL散度权重
--online_distillation_focal_gamma: Focal Loss gamma参数
--online_distillation_topk_alignment: TopK对齐参数
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 添加SGLang路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

try:
    import sglang as sgl
    from sglang.srt.server_args import ServerArgs
    print('SGLang导入成功')
except ImportError as e:
    print(f'SGLang导入失败: {e}')
    print('请确保SGLang已正确安装')
    sys.exit(1)


def load_gsm8k_data(data_path, max_samples=None):
    """加载GSM8K数据"""
    print(f"加载GSM8K数据: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"错误: 数据文件不存在 {data_path}")
        return []
        
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    print(f"加载了 {len(data)} 个样本")
    return data


def prepare_gsm8k_prompts(data):
    """准备GSM8K提示词"""
    prompts = []
    for item in data:
        if "conversation" in item and len(item["conversation"]) > 0:
            user_content = item["conversation"][0]["content"]
            # 添加数学问题求解的提示
            prompt = f"请一步步解决这个数学问题，并给出最终答案：\n\n{user_content}\n\n解题步骤："
            prompts.append(prompt)
    return prompts


def print_osd_config(args, server_args):
    """打印OSD相关配置信息"""
    print("\n=== OSD在线蒸馏配置 ===")
    print(f"目标模型: {args.model_path}")
    print(f"草稿模型: {args.speculative_draft_model_path}")
    print(f"推测算法: {args.speculative_algorithm}")
    print(f"最大提议token数: {server_args.speculative_num_draft_tokens}")
    print(f"在线蒸馏启用: {server_args.online_distillation_enable}")
    print(f"缓冲区大小: {server_args.online_distillation_buffer_size}")
    print(f"更新间隔: {server_args.online_distillation_update_interval}")
    print(f"学习率: {server_args.online_distillation_learning_rate}")
    print(f"KL散度权重: {server_args.online_distillation_kl_weight}")
    print(f"Focal Loss gamma: {server_args.online_distillation_focal_gamma}")
    print(f"KL计算方法: {args.kl_method}")
    print(f"采样来源: {args.sample_source}")
    print()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="GSM8K在线蒸馏测试")
    
    # 添加SGLang服务器参数
    ServerArgs.add_cli_args(parser)
    
    # 添加自定义参数
    parser.add_argument("--data-path", type=str, default="OSD/data/raw_data/gsm8k_train.json",
                       help="GSM8K数据路径")
    parser.add_argument("--output-dir", type=str, default="./output",
                       help="输出目录")
    parser.add_argument("--max-samples", type=int, default=100,
                       help="最大测试样本数")
    
    # OSD相关的额外参数
    parser.add_argument("--max-propose-num", type=int, default=5,
                       help="草稿模型每次提议的token数量（对应OSD的max_propose_num）")
    parser.add_argument("--online-update-interval", type=int, default=8,
                       help="在线蒸馏更新间隔（对应OSD的online_update_interval）")
    parser.add_argument("--online-eval-interval", type=int, default=1,
                       help="在线评估间隔（对应OSD的online_eval_interval）")
    parser.add_argument("--distillation-lr", type=float, default=1e-4,
                       help="在线蒸馏学习率（对应OSD的learning_rate）")
    parser.add_argument("--kl-method", type=str, default="forward", choices=["forward", "reverse", "jsd"],
                       help="KL散度计算方法（对应OSD的kl_method）")
    parser.add_argument("--sample-source", type=str, default="student", 
                       choices=["student", "teacher", "mix_request", "mix_token"],
                       help="采样来源（对应OSD的sample_source）")
    parser.add_argument("--focal-gamma", type=float, default=0.0,
                       help="Focal Loss gamma参数（对应OSD的focal_loss_gamma）")
    
    # 设置在线蒸馏相关的默认参数（基于OSD原始脚本调整）
    parser.set_defaults(
        model_path="Qwen/Qwen3-8B",
        speculative_algorithm="EAGLE",
        speculative_draft_model_path="Tengyunw/qwen3_8b_eagle3",
        # OSD对应参数：max_propose_num 5
        speculative_num_draft_tokens=5,
        # 启用在线蒸馏（对应OSD的mode online）
        online_distillation_enable=True,
        # 缓冲区大小，可以适当增加以获得更稳定的训练
        online_distillation_buffer_size=1000,
        # OSD对应参数：online_update_interval 8
        online_distillation_update_interval=8,
        # OSD对应参数：learning_rate 1e-4
        online_distillation_learning_rate=1e-4,
        # 梯度裁剪
        online_distillation_max_grad_norm=1.0,
        # 蒸馏温度，保持默认值
        online_distillation_temperature=1.0,
        # KL散度权重
        online_distillation_kl_weight=1.0,
        # Focal Loss gamma，OSD中默认为0.0
        online_distillation_focal_gamma=0.0,
        # TopK对齐，保持默认值
        online_distillation_topk_alignment=3,
        # Qwen模型使用chatml模板
        chat_template="chatml",
        # 并发请求数，可以适当增加
        max_running_requests=1000,
        # 张量并行度
        tp=1,
        # 内存分配比例
        mem_fraction_static=0.8,
        # 其他SGLang特定参数
        speculative_num_steps=5,  # 对应max_propose_num
        speculative_accept_threshold=0.5,  # 接受阈值
    )
    
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    
    # 应用OSD参数覆盖
    if hasattr(args, 'max_propose_num') and args.max_propose_num:
        server_args.speculative_num_draft_tokens = args.max_propose_num
        server_args.speculative_num_steps = args.max_propose_num
    
    if hasattr(args, 'online_update_interval') and args.online_update_interval:
        server_args.online_distillation_update_interval = args.online_update_interval
    
    if hasattr(args, 'distillation_lr') and args.distillation_lr:
        server_args.online_distillation_learning_rate = args.distillation_lr
    
    if hasattr(args, 'focal_gamma') and args.focal_gamma:
        server_args.online_distillation_focal_gamma = args.focal_gamma
    
    # 打印配置信息
    print_osd_config(args, server_args)
    
    print("开始运行GSM8K在线蒸馏测试...")
    print(f"数据路径: {args.data_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"最大样本数: {args.max_samples}")
    print()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载GSM8K数据
    data = load_gsm8k_data(args.data_path, args.max_samples)
    if not data:
        print("没有可用的测试数据")
        return
    
    # 准备提示词
    prompts = prepare_gsm8k_prompts(data)
    print(f"准备了 {len(prompts)} 个问题进行测试")
    
    # 创建采样参数
    sampling_params = {
        "temperature": 0.0,  # 数学问题使用确定性生成
        "max_new_tokens": 512,
        "top_p": 0.95,
    }
    
    # 创建SGLang引擎
    print("启动SGLang引擎进行在线蒸馏...")
    try:
        llm = sgl.Engine(
            model_path=server_args.model_path,
            speculative_algorithm=server_args.speculative_algorithm,
            speculative_draft_model_path=server_args.speculative_draft_model_path,
            online_distillation_enable=server_args.online_distillation_enable,
            online_distillation_buffer_size=server_args.online_distillation_buffer_size,
            online_distillation_update_interval=server_args.online_distillation_update_interval,
            online_distillation_learning_rate=server_args.online_distillation_learning_rate,
            online_distillation_max_grad_norm=server_args.online_distillation_max_grad_norm,
            online_distillation_temperature=server_args.online_distillation_temperature,
            online_distillation_kl_weight=server_args.online_distillation_kl_weight,
            online_distillation_focal_gamma=server_args.online_distillation_focal_gamma,
            online_distillation_topk_alignment=server_args.online_distillation_topk_alignment,
            chat_template=server_args.chat_template,
            max_running_requests=server_args.max_running_requests,
            mem_fraction_static=args.mem_fraction_static,
        )
        print("引擎启动成功！")
    except Exception as e:
        print(f"引擎启动失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 运行推理
    results = []
    start_time = time.time()
    
    try:
        for i, prompt in enumerate(prompts):
            print(f"\n处理第 {i+1}/{len(prompts)} 个问题...")
            print(f"问题: {prompt[:100]}...")
            
            try:
                # 生成回答
                output = llm.generate([prompt], sampling_params)
                generated_text = output[0]['text'] if output else ""
                
                # 保存结果
                result = {
                    "id": data[i]["id"] if "id" in data[i] else i,
                    "question": data[i]["conversation"][0]["content"],
                    "prompt": prompt,
                    "generated_answer": generated_text,
                    "timestamp": time.time()
                }
                results.append(result)
                
                print(f"生成回答: {generated_text[:200]}...")
                
            except Exception as e:
                print(f"生成回答时出错: {e}")
                error_result = {
                    "id": data[i]["id"] if "id" in data[i] else i,
                    "question": data[i]["conversation"][0]["content"],
                    "prompt": prompt,
                    "generated_answer": f"错误: {str(e)}",
                    "timestamp": time.time()
                }
                results.append(error_result)
        
        # 计算统计信息
        end_time = time.time()
        total_time = end_time - start_time
        successful_requests = len([r for r in results if not r["generated_answer"].startswith("错误")])
        
        # 保存结果
        output_file = os.path.join(args.output_dir, "gsm8k_online_results.json")
        
        result_data = {
            "config": {
                "model_path": server_args.model_path,
                "draft_model_path": server_args.speculative_draft_model_path,
                "speculative_algorithm": server_args.speculative_algorithm,
                "online_distillation_enable": server_args.online_distillation_enable,
                "online_distillation_buffer_size": server_args.online_distillation_buffer_size,
                "online_distillation_update_interval": server_args.online_distillation_update_interval,
                "online_distillation_learning_rate": server_args.online_distillation_learning_rate,
                "data_path": args.data_path,
                "max_samples": args.max_samples,
                "sampling_params": sampling_params,
            },
            "results": results,
            "statistics": {
                "total_samples": len(results),
                "successful_requests": successful_requests,
                "success_rate": successful_requests / len(results) if results else 0,
                "total_time": total_time,
                "avg_time_per_request": total_time / len(results) if results else 0,
                "requests_per_second": len(results) / total_time if total_time > 0 else 0,
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        # 打印结果摘要
        print("\n" + "=" * 60)
        print("测试完成！")
        print(f"结果保存在: {output_file}")
        print(f"总样本数: {len(results)}")
        print(f"成功请求: {successful_requests}")
        print(f"成功率: {successful_requests / len(results) * 100:.2f}%")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"平均每请求耗时: {total_time / len(results):.2f}秒")
        print(f"请求速率: {len(results) / total_time:.2f} 请求/秒")
        print("=" * 60)
        
    except Exception as e:
        print(f"推理过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        if 'llm' in locals():
            llm.shutdown()


if __name__ == "__main__":
    main()