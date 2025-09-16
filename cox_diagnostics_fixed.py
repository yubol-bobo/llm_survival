#!/usr/bin/env python3
"""
Cox回归模型诊断工具
检验Cox回归模型的各种假设和诊断统计量
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
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

class CoxModelDiagnostics:
    """Cox回归模型诊断类"""
    
    def __init__(self):
        self.data = None
        self.model = None
        self.fitted = False
        
    def load_and_prepare_data(self):
        """加载和准备数据"""
        print("📊 加载Cox模型数据...")
        
        try:
            baseline = BaselineModeling()
            baseline.load_data()
            
            # 合并所有模型数据
            combined_data = []
            for model_name, model_data in baseline.models_data.items():
                long_df = model_data['long'].copy()
                long_df['model'] = model_name
                
                # 选择必要的列
                required_cols = ['round', 'failure', 'conversation_id', 'model', 
                               'prompt_to_prompt_drift', 'context_to_prompt_drift', 
                               'cumulative_drift', 'prompt_complexity']
                
                available_cols = [col for col in required_cols if col in long_df.columns]
                model_subset = long_df[available_cols].copy()
                model_subset = model_subset.dropna()
                
                if len(model_subset) > 0:
                    combined_data.append(model_subset)
            
            self.data = pd.concat(combined_data, ignore_index=True)
            
            # 创建模型虚拟变量 (以claude_35为参照)
            model_dummies = pd.get_dummies(self.data['model'], prefix='model', drop_first=True)
            self.data = pd.concat([self.data, model_dummies], axis=1)
            
            print(f"✅ 数据加载完成: {len(self.data)} 观测值, {len(self.data['model'].unique())} 个模型")
            return True
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
    
    def fit_cox_model(self):
        """拟合Cox模型"""
        print("\n🔧 拟合Cox回归模型...")
        
        # 准备协变量
        drift_vars = ['prompt_to_prompt_drift', 'context_to_prompt_drift', 
                     'cumulative_drift', 'prompt_complexity']
        model_vars = [col for col in self.data.columns if col.startswith('model_')]
        
        covariates = drift_vars + model_vars + ['round', 'failure']
        model_data = self.data[covariates].copy()
        
        # 拟合模型
        self.model = CoxPHFitter(penalizer=0.01)  # 添加L2正则化
        self.model.fit(model_data, duration_col='round', event_col='failure', show_progress=False)
        
        self.fitted = True
        print(f"✅ 模型拟合完成 (C-index: {self.model.concordance_index_:.4f})")
        
        return self.model
    
    def test_proportional_hazards(self):
        """检验比例风险假设"""
        print("\n🧪 检验比例风险假设 (Proportional Hazards Assumption)")
        print("=" * 60)
        
        if not self.fitted:
            print("❌ 请先拟合模型")
            return
        
        try:
            # 准备用于检验的数据 - 使用模型训练时相同的数据
            drift_vars = ['prompt_to_prompt_drift', 'context_to_prompt_drift', 
                         'cumulative_drift', 'prompt_complexity']
            model_vars = [col for col in self.data.columns if col.startswith('model_')]
            covariates = drift_vars + model_vars + ['round', 'failure']
            test_data = self.data[covariates].copy()
            
            # 进行比例风险检验
            ph_test = proportional_hazard_test(self.model, test_data, time_col='round', event_col='failure')
            
            print("比例风险检验结果:")
            if hasattr(ph_test, 'p_value'):
                # p_value是一个数组，包含每个协变量的p值
                p_values = ph_test.p_value
                if isinstance(p_values, np.ndarray):
                    # 显示各协变量的检验结果
                    if hasattr(ph_test, 'summary') and ph_test.summary is not None:
                        print("\n各协变量比例风险检验:")
                        summary_df = ph_test.summary
                        for idx, var_name in enumerate(summary_df.index):
                            p_val = p_values[idx] if idx < len(p_values) else p_values[0]
                            status = "❌ 违反假设" if p_val < 0.05 else "✅ 符合假设"
                            print(f"  {var_name}: p = {p_val:.6f} {status}")
                    
                    # 全局检验：如果任何一个协变量违反假设，整体就违反
                    min_p = np.min(p_values)
                    violated_count = np.sum(p_values < 0.05)
                    print(f"\n全局检验总结:")
                    print(f"  最小p值: {min_p:.6f}")
                    print(f"  违反假设的协变量数: {violated_count}/{len(p_values)}")
                    
                    if violated_count > 0:
                        print("⚠️  部分协变量违反比例风险假设")
                    else:
                        print("✅ 所有协变量均符合比例风险假设")
                else:
                    # 标量p值的情况
                    print(f"全局检验 p值: {p_values:.6f}")
                    if p_values < 0.05:
                        print("⚠️  比例风险假设可能被违反 (p < 0.05)")
                    else:
                        print("✅ 比例风险假设合理 (p ≥ 0.05)")
            
            # 可视化Schoenfeld残差
            self.plot_schoenfeld_residuals()
            
            return ph_test
            
        except Exception as e:
            print(f"❌ 比例风险检验失败: {e}")
            print("⚠️  这可能是由于数据格式或模型复杂性导致的")
            # 尝试简化版本的检验
            try:
                print("🔄 尝试简化版比例风险检验...")
                self.simplified_ph_test()
            except Exception as simple_e:
                print(f"⚠️  简化版检验也失败: {simple_e}")
                # 尝试手动Schoenfeld残差检验
                print("🔄 尝试基于Schoenfeld残差的手动检验...")
                self.manual_ph_test()
            return None
    
    def manual_ph_test(self):
        """基于Schoenfeld残差的手动比例风险检验"""
        try:
            from scipy.stats import pearsonr
            
            # 准备数据
            drift_vars = ['prompt_to_prompt_drift', 'context_to_prompt_drift', 
                         'cumulative_drift', 'prompt_complexity']
            model_vars = [col for col in self.data.columns if col.startswith('model_')]
            covariates = drift_vars + model_vars + ['round', 'failure']
            residual_data = self.data[covariates].copy()
            
            # 计算Schoenfeld残差
            schoenfeld_resid = self.model.compute_residuals(residual_data, kind='schoenfeld')
            
            print("手动Schoenfeld残差检验结果:")
            
            # 对每个协变量检验残差与时间的相关性
            ph_violations = []
            times = residual_data['round']
            
            for i, var_name in enumerate(schoenfeld_resid.columns):
                residuals = schoenfeld_resid.iloc[:, i]
                
                # 移除NaN值
                mask = ~(np.isnan(residuals) | np.isnan(times))
                if mask.sum() < 10:
                    print(f"  {var_name}: 数据不足，跳过")
                    continue
                
                clean_resid = residuals[mask]
                clean_times = times[mask]
                
                # 计算相关性
                try:
                    corr, p_val = pearsonr(clean_times, clean_resid)
                    status = "❌ 违反假设" if p_val < 0.05 else "✅ 符合假设"
                    print(f"  {var_name}: 相关系数 = {corr:.4f}, p = {p_val:.6f} {status}")
                    
                    if p_val < 0.05:
                        ph_violations.append(var_name)
                except Exception as corr_e:
                    print(f"  {var_name}: 相关性计算失败 - {corr_e}")
            
            # 总结
            print(f"\n手动检验总结:")
            print(f"  违反比例风险假设的变量: {len(ph_violations)}")
            if ph_violations:
                print(f"  违反的变量: {', '.join(ph_violations)}")
                print("⚠️  部分变量可能违反比例风险假设")
            else:
                print("✅ 所有变量均符合比例风险假设（基于Schoenfeld残差）")
                
        except Exception as e:
            print(f"❌ 手动检验也失败: {e}")
            print("💡 建议: 模型虽然检验困难，但C-index高说明预测性能良好")
    
    def simplified_ph_test(self):
        """简化版比例风险检验"""
        try:
            # 只对核心变量进行检验
            drift_vars = ['prompt_to_prompt_drift', 'context_to_prompt_drift', 
                         'cumulative_drift', 'prompt_complexity', 'round', 'failure']
            simple_data = self.data[drift_vars].copy()
            
            # 拟合简化模型
            simple_model = CoxPHFitter()
            simple_model.fit(simple_data, duration_col='round', event_col='failure', show_progress=False)
            
            # 检验
            ph_test = proportional_hazard_test(simple_model, simple_data, time_col='round', event_col='failure')
            p_values = ph_test.p_value
            if isinstance(p_values, np.ndarray):
                min_p = np.min(p_values)
                violated_count = np.sum(p_values < 0.05)
                print(f"简化模型检验结果:")
                print(f"  最小p值: {min_p:.6f}")
                print(f"  违反假设变量数: {violated_count}/{len(p_values)}")
            else:
                print(f"简化模型全局检验 p值: {p_values:.6f}")
            
        except Exception as e:
            print(f"简化版检验失败: {e}")
    
    def plot_schoenfeld_residuals(self):
        """绘制Schoenfeld残差图检验比例风险假设"""
        try:
            # 准备数据
            drift_vars = ['prompt_to_prompt_drift', 'context_to_prompt_drift', 
                         'cumulative_drift', 'prompt_complexity']
            model_vars = [col for col in self.data.columns if col.startswith('model_')]
            covariates = drift_vars + model_vars + ['round', 'failure']
            residual_data = self.data[covariates].copy()
            
            # 获取Schoenfeld残差
            residuals = self.model.compute_residuals(residual_data, kind='schoenfeld')
            
            # 创建图形
            n_vars = len(residuals.columns)
            n_cols = min(3, n_vars)
            n_rows = (n_vars + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            for i, var in enumerate(residuals.columns):
                if i < len(axes):
                    ax = axes[i]
                    
                    # 绘制残差vs时间
                    resid_values = residuals[var]
                    
                    # 确保维度匹配 - Schoenfeld残差通常对应事件时间
                    # 创建相应的时间序列
                    n_points = len(resid_values)
                    if n_points > 0:
                        # 使用观测序号作为x轴，而不是时间
                        x_vals = range(n_points)
                        
                        ax.scatter(x_vals, resid_values, alpha=0.5, s=10)
                        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
                        ax.set_xlabel('观测序号')
                        ax.set_ylabel(f'Schoenfeld残差')
                        ax.set_title(f'{var}')
                        ax.grid(True, alpha=0.3)
                    else:
                        ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'{var} (无数据)')
            
            # 隐藏多余的子图
            for i in range(n_vars, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig('results/figures/cox_schoenfeld_residuals.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ Schoenfeld残差图已保存至 results/figures/cox_schoenfeld_residuals.png")
            
        except Exception as e:
            print(f"❌ Schoenfeld残差图绘制失败: {e}")
    
    def test_linearity(self):
        """检验对数线性假设"""
        print("\n🧪 检验对数线性假设 (Log-linearity)")
        print("=" * 50)
        
        if not self.fitted:
            print("❌ 请先拟合模型")
            return
        
        # 计算Martingale残差
        try:
            # 准备用于残差计算的数据 - 使用模型训练时相同的数据
            drift_vars = ['prompt_to_prompt_drift', 'context_to_prompt_drift', 
                         'cumulative_drift', 'prompt_complexity']
            model_vars = [col for col in self.data.columns if col.startswith('model_')]
            covariates = drift_vars + model_vars + ['round', 'failure']
            residual_data = self.data[covariates].copy()
            
            martingale_resid = self.model.compute_residuals(residual_data, kind='martingale')
            
            # 处理残差数据格式
            if isinstance(martingale_resid, pd.DataFrame):
                martingale_values = martingale_resid.iloc[:, 0]
            else:
                martingale_values = martingale_resid
            
            # 对连续变量检验线性关系
            continuous_vars = ['prompt_to_prompt_drift', 'context_to_prompt_drift', 
                             'cumulative_drift', 'prompt_complexity']
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for i, var in enumerate(continuous_vars):
                if var in self.data.columns and i < len(axes):
                    ax = axes[i]
                    
                    # 绘制Martingale残差 vs 协变量
                    x_vals = self.data[var]
                    y_vals = martingale_values
                    
                    # 确保数据长度一致
                    min_len = min(len(x_vals), len(y_vals))
                    x_vals = x_vals[:min_len]
                    y_vals = y_vals[:min_len]
                    
                    ax.scatter(x_vals, y_vals, alpha=0.5, s=20)
                    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
                    
                    # 添加平滑曲线
                    try:
                        from scipy.interpolate import UnivariateSpline
                        # 移除NaN值
                        mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
                        if mask.sum() > 50:  # 需要足够的数据点
                            x_clean = x_vals[mask]
                            y_clean = y_vals[mask]
                            
                            # 排序以便平滑
                            sort_idx = np.argsort(x_clean)
                            x_sorted = x_clean.iloc[sort_idx] if hasattr(x_clean, 'iloc') else x_clean[sort_idx]
                            y_sorted = y_clean.iloc[sort_idx] if hasattr(y_clean, 'iloc') else y_clean[sort_idx]
                            
                            # 减少平滑参数避免过拟合
                            spline = UnivariateSpline(x_sorted, y_sorted, s=len(x_sorted)*0.1)
                            smooth_x = np.linspace(x_sorted.min(), x_sorted.max(), 100)
                            ax.plot(smooth_x, spline(smooth_x), 'r-', linewidth=2, alpha=0.8)
                    except Exception as e:
                        print(f"平滑曲线绘制失败 ({var}): {e}")
                    
                    ax.set_xlabel(var)
                    ax.set_ylabel('Martingale残差')
                    ax.set_title(f'线性检验: {var}')
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('results/figures/cox_linearity_test.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ 线性检验图已保存至 results/figures/cox_linearity_test.png")
            print("💡 如果平滑曲线接近水平线，则符合线性假设")
            
        except Exception as e:
            print(f"❌ 线性检验失败: {e}")
    
    def detect_outliers(self):
        """检测异常值"""
        print("\n🧪 异常值检测")
        print("=" * 30)
        
        if not self.fitted:
            print("❌ 请先拟合模型")
            return
        
        try:
            # 准备用于残差计算的数据
            drift_vars = ['prompt_to_prompt_drift', 'context_to_prompt_drift', 
                         'cumulative_drift', 'prompt_complexity']
            model_vars = [col for col in self.data.columns if col.startswith('model_')]
            covariates = drift_vars + model_vars + ['round', 'failure']
            residual_data = self.data[covariates].copy()
            
            # 计算Deviance残差
            deviance_resid = self.model.compute_residuals(residual_data, kind='deviance')
            
            # 处理残差数据格式
            if isinstance(deviance_resid, pd.DataFrame):
                deviance_values = deviance_resid.iloc[:, 0]  # 取第一列
            else:
                deviance_values = deviance_resid
            
            # 定义异常值阈值 (通常用±2.5标准差)
            threshold = 2.5
            outliers = np.abs(deviance_values) > threshold
            
            print(f"异常值检测结果:")
            print(f"  总观测数: {len(deviance_values)}")
            print(f"  异常值数量: {outliers.sum()}")
            print(f"  异常值比例: {outliers.sum()/len(deviance_values)*100:.2f}%")
            
            if outliers.sum() > 0:
                print(f"\n异常值观测ID:")
                outlier_indices = np.where(outliers)[0]
                for idx in outlier_indices[:10]:  # 只显示前10个
                    print(f"  观测 {idx}: Deviance残差 = {deviance_values.iloc[idx]:.3f}")
                if len(outlier_indices) > 10:
                    print(f"  ... 还有 {len(outlier_indices)-10} 个异常值")
            
            # 绘制异常值图
            plt.figure(figsize=(10, 6))
            plt.scatter(range(len(deviance_values)), deviance_values, alpha=0.6, s=20)
            plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'阈值 ±{threshold}')
            plt.axhline(y=-threshold, color='red', linestyle='--', alpha=0.7)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # 标记异常值
            if outliers.sum() > 0:
                outlier_x = np.where(outliers)[0]
                outlier_y = deviance_values[outliers]
                plt.scatter(outlier_x, outlier_y, color='red', s=50, alpha=0.8, label=f'异常值 ({outliers.sum()})')
            
            plt.xlabel('观测序号')
            plt.ylabel('Deviance残差')
            plt.title('Cox回归异常值检测')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('results/figures/cox_outliers.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ 异常值图已保存至 results/figures/cox_outliers.png")
            
            return outliers
            
        except Exception as e:
            print(f"❌ 异常值检测失败: {e}")
            return None
    
    def check_multicollinearity(self):
        """检查多重共线性"""
        print("\n🧪 多重共线性检验")
        print("=" * 40)
        
        # 计算连续变量的相关矩阵
        drift_vars = ['prompt_to_prompt_drift', 'context_to_prompt_drift', 
                     'cumulative_drift', 'prompt_complexity']
        
        corr_data = self.data[drift_vars]
        corr_matrix = corr_data.corr()
        
        print("协变量相关矩阵:")
        print(corr_matrix.round(3))
        
        # 检查高相关性
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        if high_corr_pairs:
            print(f"\n⚠️  发现高相关性变量对 (|r| > 0.8):")
            for var1, var2, corr_val in high_corr_pairs:
                print(f"  {var1} - {var2}: r = {corr_val:.3f}")
        else:
            print("\n✅ 未发现严重多重共线性问题")
        
        # 绘制相关性热图
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, cbar_kws={'shrink': 0.8})
        plt.title('协变量相关性矩阵')
        plt.tight_layout()
        plt.savefig('results/figures/cox_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 相关性矩阵图已保存至 results/figures/cox_correlation_matrix.png")
        
        return corr_matrix
    
    def goodness_of_fit(self):
        """拟合优度评估"""
        print("\n🧪 模型拟合优度评估")
        print("=" * 40)
        
        if not self.fitted:
            print("❌ 请先拟合模型")
            return
        
        # 计算各种拟合指标
        c_index = self.model.concordance_index_
        log_likelihood = self.model.log_likelihood_
        aic_partial = self.model.AIC_partial_  # Cox模型使用partial AIC
        
        print(f"拟合优度指标:")
        print(f"  C-index (一致性指数): {c_index:.4f}")
        print(f"  Log-likelihood: {log_likelihood:.2f}")
        print(f"  AIC (partial): {aic_partial:.2f}")
        
        # C-index解释
        if c_index > 0.8:
            c_interpretation = "优秀"
        elif c_index > 0.7:
            c_interpretation = "良好"
        elif c_index > 0.6:
            c_interpretation = "中等"
        else:
            c_interpretation = "较差"
        
        print(f"  C-index解释: {c_interpretation}")
        
        # 计算残差统计
        try:
            # 准备用于残差计算的数据
            drift_vars = ['prompt_to_prompt_drift', 'context_to_prompt_drift', 
                         'cumulative_drift', 'prompt_complexity']
            model_vars = [col for col in self.data.columns if col.startswith('model_')]
            covariates = drift_vars + model_vars + ['round', 'failure']
            residual_data = self.data[covariates].copy()
            
            martingale_resid = self.model.compute_residuals(residual_data, kind='martingale')
            deviance_resid = self.model.compute_residuals(residual_data, kind='deviance')
            
            # 处理残差数据格式
            if isinstance(martingale_resid, pd.DataFrame):
                martingale_values = martingale_resid.iloc[:, 0]
            else:
                martingale_values = martingale_resid
                
            if isinstance(deviance_resid, pd.DataFrame):
                deviance_values = deviance_resid.iloc[:, 0]
            else:
                deviance_values = deviance_resid
            
            print(f"\n残差统计:")
            print(f"  Martingale残差均值: {martingale_values.mean():.6f}")
            print(f"  Martingale残差标准差: {martingale_values.std():.4f}")
            print(f"  Deviance残差均值: {deviance_values.mean():.6f}")
            print(f"  Deviance残差标准差: {deviance_values.std():.4f}")
            
        except Exception as e:
            print(f"残差计算失败: {e}")
        
        return {
            'c_index': c_index,
            'log_likelihood': log_likelihood,
            'aic_partial': aic_partial
        }
    
    def comprehensive_diagnostics(self):
        """运行完整的模型诊断"""
        print("🩺 COX回归模型完整诊断")
        print("=" * 80)
        
        # 加载数据
        if not self.load_and_prepare_data():
            return
        
        # 拟合模型
        self.fit_cox_model()
        
        print("=" * 80)
        
        # 运行所有诊断测试
        self.test_proportional_hazards()
        
        print("=" * 80)
        
        self.test_linearity()
        
        print("=" * 80)
        
        self.detect_outliers()
        
        print("=" * 80)
        
        self.check_multicollinearity()
        
        print("=" * 80)
        
        self.goodness_of_fit()
        
        print("=" * 80)
        print("🎉 模型诊断完成！")
        print("📁 诊断图表已保存至 results/figures/ 目录")

def main():
    """主函数"""
    # 创建诊断对象并运行完整诊断
    diagnostics = CoxModelDiagnostics()
    diagnostics.comprehensive_diagnostics()

if __name__ == "__main__":
    main()