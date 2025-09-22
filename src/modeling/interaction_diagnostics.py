#!/usr/bin/env python3
"""
交互项Cox回归假设检验
专门检验包含交互项的Cox模型是否满足比例风险假设
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 导入现有的分析模块
import sys
import os
sys.path.append(os.path.dirname(__file__))
from src.modeling.baseline import BaselineModeling

class InteractionCoxDiagnostics:
    """交互项Cox回归模型诊断类"""
    
    def __init__(self):
        self.data = None
        self.baseline_model = None
        self.interaction_model = None
        self.fitted_baseline = False
        self.fitted_interaction = False
        
    def load_and_prepare_data(self):
        """加载和准备数据"""
        print("📊 加载包含交互项的Cox模型数据...")
        
        try:
            baseline = BaselineModeling()
            baseline.load_data()
            
            # 合并所有模型数据
            combined_data = []
            for model_name, model_data in baseline.models_data.items():
                long_df = model_data['long'].copy()
                long_df['model'] = model_name
                
                required_cols = ['round', 'failure', 'conversation_id', 'model', 
                               'prompt_to_prompt_drift', 'context_to_prompt_drift', 
                               'cumulative_drift', 'prompt_complexity']
                
                available_cols = [col for col in required_cols if col in long_df.columns]
                model_subset = long_df[available_cols].copy()
                model_subset = model_subset.dropna()
                
                if len(model_subset) > 0:
                    combined_data.append(model_subset)
            
            self.data = pd.concat(combined_data, ignore_index=True)
            
            # 创建模型虚拟变量
            model_dummies = pd.get_dummies(self.data['model'], prefix='model', drop_first=True)
            self.data = pd.concat([self.data, model_dummies], axis=1)
            
            # 创建交互项
            self.create_interaction_terms()
            
            print(f"✅ 数据加载完成: {len(self.data)} 观测值, {len(self.data['model'].unique())} 个模型")
            return True
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
    
    def create_interaction_terms(self):
        """创建交互项"""
        print("⚡ 创建交互项...")
        
        drift_vars = ['prompt_to_prompt_drift', 'context_to_prompt_drift', 
                     'cumulative_drift', 'prompt_complexity']
        model_vars = [col for col in self.data.columns if col.startswith('model_')]
        
        interaction_count = 0
        for drift_var in drift_vars:
            for model_var in model_vars:
                interaction_name = f"{drift_var}_x_{model_var}"
                self.data[interaction_name] = self.data[drift_var] * self.data[model_var]
                interaction_count += 1
        
        print(f"✅ 创建了 {interaction_count} 个交互项")
    
    def fit_baseline_model(self):
        """拟合基础模型（无交互项）"""
        print("\n🔧 拟合基础Cox回归模型（无交互项）...")
        
        drift_vars = ['prompt_to_prompt_drift', 'context_to_prompt_drift', 
                     'cumulative_drift', 'prompt_complexity']
        model_vars = [col for col in self.data.columns if col.startswith('model_')]
        
        baseline_vars = drift_vars + model_vars + ['round', 'failure']
        baseline_data = self.data[baseline_vars].copy()
        
        self.baseline_model = CoxPHFitter(penalizer=0.01)
        self.baseline_model.fit(baseline_data, duration_col='round', event_col='failure', show_progress=False)
        
        self.fitted_baseline = True
        print(f"✅ 基础模型拟合完成 (C-index: {self.baseline_model.concordance_index_:.4f})")
        
        return self.baseline_model
    
    def fit_interaction_model(self):
        """拟合交互项模型"""
        print("\n🔧 拟合交互项Cox回归模型...")
        
        # 获取所有特征（包括交互项）
        drift_vars = ['prompt_to_prompt_drift', 'context_to_prompt_drift', 
                     'cumulative_drift', 'prompt_complexity']
        model_vars = [col for col in self.data.columns if col.startswith('model_')]
        interaction_vars = [col for col in self.data.columns if '_x_model_' in col]
        
        all_vars = drift_vars + model_vars + interaction_vars + ['round', 'failure']
        interaction_data = self.data[all_vars].copy()
        
        print(f"   包含特征: {len(drift_vars)} 漂移 + {len(model_vars)} 模型 + {len(interaction_vars)} 交互项")
        
        self.interaction_model = CoxPHFitter(penalizer=0.05)  # 增加正则化
        self.interaction_model.fit(interaction_data, duration_col='round', event_col='failure', show_progress=False)
        
        self.fitted_interaction = True
        print(f"✅ 交互模型拟合完成 (C-index: {self.interaction_model.concordance_index_:.4f})")
        
        return self.interaction_model
    
    def test_interaction_proportional_hazards(self):
        """检验交互项模型的比例风险假设"""
        print("\n🧪 检验交互项模型的比例风险假设")
        print("=" * 60)
        
        if not self.fitted_interaction:
            print("❌ 请先拟合交互项模型")
            return
        
        try:
            # 准备测试数据
            drift_vars = ['prompt_to_prompt_drift', 'context_to_prompt_drift', 
                         'cumulative_drift', 'prompt_complexity']
            model_vars = [col for col in self.data.columns if col.startswith('model_')]
            interaction_vars = [col for col in self.data.columns if '_x_model_' in col]
            
            all_vars = drift_vars + model_vars + interaction_vars + ['round', 'failure']
            test_data = self.data[all_vars].copy()
            
            # 由于交互项模型复杂，使用手动Schoenfeld残差检验
            print("🔄 使用Schoenfeld残差进行比例风险检验...")
            self.manual_interaction_ph_test(test_data)
            
        except Exception as e:
            print(f"❌ 交互项比例风险检验失败: {e}")
            print("🔄 尝试分组检验...")
            self.grouped_ph_test()
    
    def manual_interaction_ph_test(self, test_data):
        """基于Schoenfeld残差的交互项比例风险检验"""
        try:
            from scipy.stats import pearsonr
            
            # 计算Schoenfeld残差
            schoenfeld_resid = self.interaction_model.compute_residuals(test_data, kind='schoenfeld')
            
            print("交互项Schoenfeld残差检验结果:")
            
            # 分类检验不同类型的特征
            feature_categories = {
                '基础漂移特征': ['prompt_to_prompt_drift', 'context_to_prompt_drift', 
                              'cumulative_drift', 'prompt_complexity'],
                '模型虚拟变量': [col for col in schoenfeld_resid.columns if col.startswith('model_') and '_x_' not in col],
                '交互项特征': [col for col in schoenfeld_resid.columns if '_x_model_' in col]
            }
            
            times = test_data['round']
            category_violations = {}
            
            for category, features in feature_categories.items():
                print(f"\n【{category}】:")
                violations = []
                
                for feature in features:
                    if feature in schoenfeld_resid.columns:
                        residuals = schoenfeld_resid[feature]
                        
                        # 移除NaN值
                        mask = ~(np.isnan(residuals) | np.isnan(times))
                        if mask.sum() < 10:
                            continue
                        
                        clean_resid = residuals[mask]
                        clean_times = times[mask]
                        
                        try:
                            corr, p_val = pearsonr(clean_times, clean_resid)
                            status = "❌ 违反" if p_val < 0.05 else "✅ 符合"
                            print(f"  {feature}: r={corr:.4f}, p={p_val:.6f} {status}")
                            
                            if p_val < 0.05:
                                violations.append(feature)
                        except:
                            continue
                
                category_violations[category] = violations
            
            # 总结各类别的违反情况
            print(f"\n=== 比例风险假设违反总结 ===")
            for category, violations in category_violations.items():
                total_features = len(feature_categories[category])
                violation_count = len(violations)
                violation_rate = violation_count / total_features * 100 if total_features > 0 else 0
                
                print(f"{category}: {violation_count}/{total_features} ({violation_rate:.1f}%) 违反假设")
                
                if violations:
                    print(f"  违反的特征: {', '.join(violations[:3])}{'...' if len(violations) > 3 else ''}")
            
            # 特别关注交互项
            interaction_violations = category_violations.get('交互项特征', [])
            if interaction_violations:
                print(f"\n⚠️  {len(interaction_violations)} 个交互项违反比例风险假设")
                print("这可能意味着:")
                print("  1. 交互效应本身随时间变化")
                print("  2. 需要考虑三阶交互（特征×模型×时间）")
                print("  3. 模型可能过度复杂化")
            else:
                print(f"\n✅ 交互项基本符合比例风险假设")
                
        except Exception as e:
            print(f"❌ 手动交互项检验失败: {e}")
    
    def grouped_ph_test(self):
        """分组比例风险检验"""
        print("\n🔄 分组比例风险检验...")
        
        try:
            # 只检验最重要的交互项
            significant_interactions = [
                'prompt_to_prompt_drift_x_model_deepseek_r1',
                'prompt_to_prompt_drift_x_model_mistral_large',
                'prompt_to_prompt_drift_x_model_qwen_max'
            ]
            
            for interaction in significant_interactions:
                if interaction in self.data.columns:
                    print(f"\n检验 {interaction}:")
                    
                    # 创建简化数据集
                    simple_vars = ['prompt_to_prompt_drift', interaction, 'round', 'failure']
                    simple_data = self.data[simple_vars].copy().dropna()
                    
                    if len(simple_data) > 100:
                        # 拟合简化模型
                        simple_model = CoxPHFitter()
                        simple_model.fit(simple_data, duration_col='round', event_col='failure', show_progress=False)
                        
                        # 计算残差
                        residuals = simple_model.compute_residuals(simple_data, kind='schoenfeld')
                        times = simple_data['round']
                        
                        for col in residuals.columns:
                            if col == interaction:
                                mask = ~(np.isnan(residuals[col]) | np.isnan(times))
                                if mask.sum() > 10:
                                    corr, p_val = pearsonr(times[mask], residuals[col][mask])
                                    status = "❌ 违反" if p_val < 0.05 else "✅ 符合"
                                    print(f"  {col}: p={p_val:.6f} {status}")
        
        except Exception as e:
            print(f"分组检验失败: {e}")
    
    def compare_models(self):
        """比较基础模型和交互项模型"""
        print("\n📈 模型比较分析")
        print("=" * 50)
        
        if not (self.fitted_baseline and self.fitted_interaction):
            print("❌ 请先拟合两个模型")
            return
        
        print("基础模型 vs 交互项模型:")
        print(f"  基础模型 C-index: {self.baseline_model.concordance_index_:.4f}")
        print(f"  交互模型 C-index: {self.interaction_model.concordance_index_:.4f}")
        print(f"  改进幅度: {(self.interaction_model.concordance_index_ - self.baseline_model.concordance_index_)*100:.2f}%")
        
        print(f"\n  基础模型 AIC: {self.baseline_model.AIC_partial_:.2f}")
        print(f"  交互模型 AIC: {self.interaction_model.AIC_partial_:.2f}")
        
        aic_improvement = self.baseline_model.AIC_partial_ - self.interaction_model.AIC_partial_
        print(f"  AIC改进: {aic_improvement:.2f} ({'更好' if aic_improvement > 0 else '更差'})")
        
        # 似然比检验
        try:
            lr_stat = -2 * (self.baseline_model.log_likelihood_ - self.interaction_model.log_likelihood_)
            # 交互项数量作为自由度
            interaction_count = len([col for col in self.data.columns if '_x_model_' in col])
            
            from scipy.stats import chi2
            p_value = 1 - chi2.cdf(lr_stat, interaction_count)
            
            print(f"\n似然比检验:")
            print(f"  LR统计量: {lr_stat:.4f}")
            print(f"  自由度: {interaction_count}")
            print(f"  p值: {p_value:.2e}")
            print(f"  结论: {'交互项显著改善模型' if p_value < 0.05 else '交互项改善不显著'}")
            
        except Exception as e:
            print(f"似然比检验失败: {e}")
    
    def comprehensive_interaction_diagnostics(self):
        """运行完整的交互项诊断"""
        print("🧪 交互项Cox回归模型完整诊断")
        print("=" * 80)
        
        # 加载数据
        if not self.load_and_prepare_data():
            return
        
        # 拟合两个模型
        self.fit_baseline_model()
        self.fit_interaction_model()
        
        print("=" * 80)
        
        # 比例风险假设检验
        self.test_interaction_proportional_hazards()
        
        print("=" * 80)
        
        # 模型比较
        self.compare_models()
        
        print("=" * 80)
        print("🎉 交互项诊断完成！")

def main():
    """主函数"""
    diagnostics = InteractionCoxDiagnostics()
    diagnostics.comprehensive_interaction_diagnostics()

if __name__ == "__main__":
    main()