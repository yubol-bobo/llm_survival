#!/usr/bin/env python3
"""
LLM生存分析特征构成全面分析报告
分析Cox回归模型中的15个基础特征以及高级交互项和时间变化协变量
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def comprehensive_feature_analysis():
    """全面分析LLM生存分析中的特征构成"""
    
    print("🔍 LLM生存分析特征构成全面分析")
    print("="*80)
    
    # 1. 基础15个特征分析
    print("\n📊 **1. 基础15个特征分析**")
    print("-"*50)
    
    basic_features = {
        "漂移特征 (4个)": [
            "prompt_to_prompt_drift",    # 提示间漂移
            "context_to_prompt_drift",   # 上下文到提示漂移  
            "cumulative_drift",          # 累积漂移
            "prompt_complexity"          # 提示复杂度
        ],
        "模型虚拟变量 (11个)": [
            "model_deepseek_r1",         # DeepSeek R1
            "model_gemini_25",           # Gemini 2.5
            "model_gpt_4o",             # GPT-4o
            "model_gpt_5",              # GPT-5
            "model_gpt_oss",            # GPT OSS
            "model_llama_33",           # LLaMA 3.3
            "model_llama_4_maverick",   # LLaMA 4 Maverick
            "model_llama_4_scout",      # LLaMA 4 Scout
            "model_mistral_large",      # Mistral Large
            "model_qwen_3",             # Qwen 3.0
            "model_qwen_max"            # Qwen Max
        ]
    }
    
    for category, features in basic_features.items():
        print(f"\n**{category}:**")
        for i, feature in enumerate(features, 1):
            print(f"   {i}. {feature}")
    
    print(f"\n**总计:** {sum(len(features) for features in basic_features.values())} 个基础特征")
    
    # 2. 高级交互项分析
    print("\n⚡ **2. 高级交互项分析**")
    print("-"*50)
    
    try:
        interaction_df = pd.read_csv('results/outputs/advanced/interaction_effects.csv')
        print(f"**发现 {len(interaction_df)} 个交互项！**")
        
        # 按显著性分组
        significant_interactions = interaction_df[interaction_df['significant'] == True]
        print(f"   - 显著交互项: {len(significant_interactions)} 个")
        print(f"   - 非显著交互项: {len(interaction_df) - len(significant_interactions)} 个")
        
        print(f"\n**显著交互项前5个:**")
        top_interactions = significant_interactions.nlargest(5, 'hazard_ratio')
        for idx, row in top_interactions.iterrows():
            print(f"   • {row['interaction_term']}")
            print(f"     HR = {row['hazard_ratio']:.2e}, p = {row['p_value']:.6f}")
        
        print(f"\n**交互项类型分布:**")
        interaction_types = {}
        for term in interaction_df['interaction_term']:
            if 'prompt_to_prompt_drift_x_model' in term:
                interaction_types['P2P漂移×模型'] = interaction_types.get('P2P漂移×模型', 0) + 1
            elif 'context_to_prompt_drift_x_model' in term:
                interaction_types['C2P漂移×模型'] = interaction_types.get('C2P漂移×模型', 0) + 1
            elif 'cumulative_drift_x_model' in term:
                interaction_types['累积漂移×模型'] = interaction_types.get('累积漂移×模型', 0) + 1
            elif 'prompt_complexity_x_model' in term:
                interaction_types['复杂度×模型'] = interaction_types.get('复杂度×模型', 0) + 1
        
        for itype, count in interaction_types.items():
            print(f"   • {itype}: {count} 个")
            
    except FileNotFoundError:
        print("   ❌ 交互项数据文件未找到")
    
    # 3. 时间变化协变量分析  
    print("\n⏰ **3. 时间变化协变量分析**")
    print("-"*50)
    
    try:
        time_varying_df = pd.read_csv('results/outputs/time_varying/time_varying_models.csv')
        print(f"**发现 {len(time_varying_df)} 个模型的时间变化分析！**")
        
        # 分析时间变化特征
        tv_models = time_varying_df['model'].tolist()
        print(f"   - 包含模型: {', '.join(tv_models)}")
        
        # C-index性能
        avg_cindex = time_varying_df['c_index'].mean()
        print(f"   - 平均C-index: {avg_cindex:.4f}")
        
        print(f"\n**时间变化特征效应:**")
        for col in ['prompt_to_prompt_drift', 'context_to_prompt_drift', 'cumulative_drift', 'prompt_complexity']:
            coef_col = f"{col}_coef"
            pval_col = f"{col}_pval"
            if coef_col in time_varying_df.columns:
                avg_coef = time_varying_df[coef_col].mean()
                avg_pval = time_varying_df[pval_col].mean()
                print(f"   • {col}: 平均系数 = {avg_coef:.6f}, 平均p值 = {avg_pval:.4f}")
        
        # 检查交互时间变化
        interaction_tv_df = pd.read_csv('results/outputs/time_varying/interaction_time_varying_results.csv')
        print(f"\n**交互×时间变化分析:**")
        print(f"   - 总协变量数: {len(interaction_tv_df)}")
        
        unique_models = interaction_tv_df['Model'].nunique()
        print(f"   - 涉及模型数: {unique_models}")
        
        significant_tv = interaction_tv_df[interaction_tv_df['p'] < 0.05]
        print(f"   - 显著时间变化效应: {len(significant_tv)} 个")
        
    except FileNotFoundError:
        print("   ❌ 时间变化数据文件未找到")
    
    # 4. 特征复杂度总结
    print("\n🎯 **4. 特征复杂度层次总结**")
    print("-"*50)
    
    feature_hierarchy = {
        "第1层 - 基础特征": {
            "数量": 15,
            "类型": "4个漂移特征 + 11个模型虚拟变量",
            "说明": "标准Cox回归的基础协变量"
        },
        "第2层 - 交互效应": {
            "数量": "44个",
            "类型": "漂移特征 × 模型交互项",
            "说明": "捕捉不同模型对漂移的差异化响应"
        },
        "第3层 - 时间变化": {
            "数量": "270+个",
            "类型": "基础特征的时间相关变化",
            "说明": "允许协变量效应随时间变化"
        },
        "第4层 - 交互×时间": {
            "数量": "复合型",
            "类型": "交互项的时间变化效应",
            "说明": "最复杂的动态交互模式"
        }
    }
    
    for layer, info in feature_hierarchy.items():
        print(f"\n**{layer}:**")
        print(f"   • 特征数量: {info['数量']}")
        print(f"   • 特征类型: {info['类型']}")
        print(f"   • 功能说明: {info['说明']}")
    
    # 5. 建议的扩展特征
    print("\n💡 **5. 建议的扩展特征**")
    print("-"*50)
    
    extensions = {
        "时间多项式特征": [
            "round^2, round^3 (非线性时间效应)",
            "log(round), sqrt(round) (时间变换)"
        ],
        "漂移间交互": [
            "prompt_to_prompt_drift × cumulative_drift",
            "context_to_prompt_drift × prompt_complexity"
        ],
        "分层时间效应": [
            "early_phase (round 1-2) × features",
            "late_phase (round 3-4) × features"
        ],
        "动态阈值特征": [
            "drift_threshold_crossed (漂移超过阈值的二元变量)",
            "complexity_percentile (复杂度在对话中的相对位置)"
        ],
        "序列依赖": [
            "lagged_drift (前一轮的漂移值)",
            "drift_acceleration (漂移变化率)"
        ]
    }
    
    for ext_type, features in extensions.items():
        print(f"\n**{ext_type}:**")
        for feature in features:
            print(f"   • {feature}")
    
    print("\n" + "="*80)
    print("🎉 **特征分析完成！当前模型已从15个基础特征扩展到300+个复合特征**")
    print("="*80)

if __name__ == "__main__":
    comprehensive_feature_analysis()