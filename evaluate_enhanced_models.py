#!/usr/bin/env python3
"""
Evaluation script for enhanced aviation visibility prediction models.

This script evaluates the trained models against acceptance criteria and generates
comprehensive performance reports.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
import logging
from datetime import datetime

# Add src to path
sys.path.append('src')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedModelEvaluator:
    """
    Evaluator for enhanced visibility prediction models.
    """
    
    def __init__(self, config_path: str = "config/enhanced_model.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Find latest model directory
        self.model_dir = self._find_latest_model_dir()
        self.metrics_path = os.path.join(self.model_dir, 'metrics.pkl')
        self.time_splits_path = os.path.join(self.model_dir.replace('enhanced_model_trainer', 'enhanced_data_transformation'), 'time_splits.pkl')
        
        # Load metrics if available
        self.metrics = None
        if os.path.exists(self.metrics_path):
            self.metrics = joblib.load(self.metrics_path)
        
        # Evaluation criteria
        self.target_mae_1mi = self.config['evaluation']['target_mae_1mi']
        self.target_mae_3mi = self.config['evaluation']['target_mae_3mi']
        self.target_auc = self.config['evaluation']['target_auc']
        self.target_recall = self.config['evaluation']['target_recall']
        
        self.visibility_bins = self.config['evaluation']['visibility_bins']
        self.bin_labels = self.config['evaluation']['bin_labels']
    
    def _find_latest_model_dir(self) -> str:
        """Find the latest model directory."""
        artifacts_dir = "artifacts"
        if os.path.exists(artifacts_dir):
            timestamp_dirs = [d for d in os.listdir(artifacts_dir) 
                            if os.path.isdir(os.path.join(artifacts_dir, d))]
            if timestamp_dirs:
                latest_timestamp = max(timestamp_dirs)
                return os.path.join(artifacts_dir, latest_timestamp, 'enhanced_model_trainer')
        
        return os.path.join(artifacts_dir, 'enhanced_model_trainer')
    
    def evaluate_acceptance_criteria(self) -> Dict[str, bool]:
        """
        Evaluate models against acceptance criteria.
        
        Returns:
            Dictionary of acceptance criteria results
        """
        if self.metrics is None:
            logger.warning("No metrics found. Run training first.")
            return {}
        
        criteria = {}
        
        # Check two-stage model criteria
        if 'two_stage' in self.metrics and 'binned_metrics' in self.metrics['two_stage']:
            binned_metrics = self.metrics['two_stage']['binned_metrics']
            
            # MAE criteria for <1mi bin
            if '<1mi' in binned_metrics:
                mae_1mi = binned_metrics['<1mi']['mae']
                criteria['mae_1mi_two_stage'] = mae_1mi <= self.target_mae_1mi
                criteria['mae_1mi_value'] = mae_1mi
            
            # MAE criteria for 1-3mi bin
            if '1-3mi' in binned_metrics:
                mae_3mi = binned_metrics['1-3mi']['mae']
                criteria['mae_3mi_two_stage'] = mae_3mi <= self.target_mae_3mi
                criteria['mae_3mi_value'] = mae_3mi
            
            # AUC criteria for classifier
            if 'classifier_auc' in self.metrics['two_stage']:
                auc = self.metrics['two_stage']['classifier_auc']
                criteria['auc_two_stage'] = auc >= self.target_auc
                criteria['auc_value'] = auc
        
        # Check weighted model criteria
        if 'weighted' in self.metrics and 'binned_metrics' in self.metrics['weighted']:
            binned_metrics = self.metrics['weighted']['binned_metrics']
            
            # MAE criteria for <1mi bin
            if '<1mi' in binned_metrics:
                mae_1mi = binned_metrics['<1mi']['mae']
                criteria['mae_1mi_weighted'] = mae_1mi <= self.target_mae_1mi
                if 'mae_1mi_value' not in criteria:
                    criteria['mae_1mi_value'] = mae_1mi
            
            # MAE criteria for 1-3mi bin
            if '1-3mi' in binned_metrics:
                mae_3mi = binned_metrics['1-3mi']['mae']
                criteria['mae_3mi_weighted'] = mae_3mi <= self.target_mae_3mi
                if 'mae_3mi_value' not in criteria:
                    criteria['mae_3mi_value'] = mae_3mi
        
        return criteria
    
    def generate_performance_report(self) -> str:
        """
        Generate comprehensive performance report.
        
        Returns:
            Report string
        """
        if self.metrics is None:
            return "No metrics available. Please run training first."
        
        report = []
        report.append("="*80)
        report.append("ENHANCED AVIATION VISIBILITY PREDICTION - PERFORMANCE REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall performance
        report.append("OVERALL PERFORMANCE")
        report.append("-" * 40)
        
        for model_type in ['two_stage', 'weighted']:
            if model_type in self.metrics:
                metrics = self.metrics[model_type]
                report.append(f"\n{model_type.upper()} MODEL:")
                report.append(f"  MAE: {metrics['mae']:.3f} miles")
                report.append(f"  RMSE: {metrics['rmse']:.3f} miles")
                report.append(f"  R²: {metrics['r2']:.3f}")
                
                if 'classifier_auc' in metrics:
                    report.append(f"  Classifier AUC: {metrics['classifier_auc']:.3f}")
        
        # Binned performance
        report.append("\nBINNED PERFORMANCE")
        report.append("-" * 40)
        
        for model_type in ['two_stage', 'weighted']:
            if model_type in self.metrics and 'binned_metrics' in self.metrics[model_type]:
                report.append(f"\n{model_type.upper()} MODEL:")
                binned_metrics = self.metrics[model_type]['binned_metrics']
                
                for bin_name in self.bin_labels:
                    if bin_name in binned_metrics:
                        bin_metrics = binned_metrics[bin_name]
                        report.append(f"  {bin_name}:")
                        report.append(f"    MAE: {bin_metrics['mae']:.3f} miles")
                        report.append(f"    RMSE: {bin_metrics['rmse']:.3f} miles")
                        report.append(f"    R²: {bin_metrics['r2']:.3f}")
                        report.append(f"    Samples: {bin_metrics['samples']}")
        
        # Uncertainty quantification
        if 'quantile' in self.metrics:
            report.append("\nUNCERTAINTY QUANTIFICATION")
            report.append("-" * 40)
            quantile_metrics = self.metrics['quantile']
            report.append(f"  Mean PI Width: {quantile_metrics['mean_pi_width']:.3f} miles")
            report.append(f"  95% Coverage: {quantile_metrics['coverage_95']:.3f}")
        
        # Acceptance criteria
        criteria = self.evaluate_acceptance_criteria()
        if criteria:
            report.append("\nACCEPTANCE CRITERIA")
            report.append("-" * 40)
            
            for criterion, result in criteria.items():
                if isinstance(result, bool):
                    status = "✓ PASS" if result else "✗ FAIL"
                    report.append(f"  {criterion}: {status}")
                else:
                    report.append(f"  {criterion}: {result:.3f}")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)
    
    def create_visualizations(self, save_path: str = "evaluation_plots"):
        """
        Create performance visualizations.
        
        Args:
            save_path: Directory to save plots
        """
        if self.metrics is None:
            logger.warning("No metrics found. Cannot create visualizations.")
            return
        
        os.makedirs(save_path, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Model comparison plot
        self._create_model_comparison_plot(save_path)
        
        # 2. Binned performance plot
        self._create_binned_performance_plot(save_path)
        
        # 3. Acceptance criteria plot
        self._create_acceptance_criteria_plot(save_path)
        
        logger.info(f"Visualizations saved to {save_path}")
    
    def _create_model_comparison_plot(self, save_path: str):
        """Create model comparison visualization."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        models = []
        mae_values = []
        rmse_values = []
        r2_values = []
        
        for model_type in ['two_stage', 'weighted']:
            if model_type in self.metrics:
                models.append(model_type.replace('_', ' ').title())
                mae_values.append(self.metrics[model_type]['mae'])
                rmse_values.append(self.metrics[model_type]['rmse'])
                r2_values.append(self.metrics[model_type]['r2'])
        
        # MAE comparison
        axes[0].bar(models, mae_values, color=['#1f77b4', '#ff7f0e'])
        axes[0].set_title('Mean Absolute Error Comparison')
        axes[0].set_ylabel('MAE (miles)')
        axes[0].tick_params(axis='x', rotation=45)
        
        # RMSE comparison
        axes[1].bar(models, rmse_values, color=['#1f77b4', '#ff7f0e'])
        axes[1].set_title('Root Mean Square Error Comparison')
        axes[1].set_ylabel('RMSE (miles)')
        axes[1].tick_params(axis='x', rotation=45)
        
        # R² comparison
        axes[2].bar(models, r2_values, color=['#1f77b4', '#ff7f0e'])
        axes[2].set_title('R² Score Comparison')
        axes[2].set_ylabel('R²')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_binned_performance_plot(self, save_path: str):
        """Create binned performance visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for i, model_type in enumerate(['two_stage', 'weighted']):
            if model_type in self.metrics and 'binned_metrics' in self.metrics[model_type]:
                binned_metrics = self.metrics[model_type]['binned_metrics']
                
                bins = []
                mae_values = []
                sample_counts = []
                
                for bin_name in self.bin_labels:
                    if bin_name in binned_metrics:
                        bins.append(bin_name)
                        mae_values.append(binned_metrics[bin_name]['mae'])
                        sample_counts.append(binned_metrics[bin_name]['samples'])
                
                # MAE by bin
                axes[i].bar(bins, mae_values, color=['#d62728', '#ff7f0e', '#2ca02c'])
                axes[i].set_title(f'{model_type.replace("_", " ").title()} - MAE by Visibility Bin')
                axes[i].set_ylabel('MAE (miles)')
                axes[i].tick_params(axis='x', rotation=45)
                
                # Add sample count annotations
                for j, (bin_name, count) in enumerate(zip(bins, sample_counts)):
                    axes[i].text(j, mae_values[j] + 0.05, f'n={count}', 
                               ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'binned_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_acceptance_criteria_plot(self, save_path: str):
        """Create acceptance criteria visualization."""
        criteria = self.evaluate_acceptance_criteria()
        
        if not criteria:
            return
        
        # Prepare data for plotting
        criterion_names = []
        values = []
        targets = []
        passed = []
        
        for criterion, result in criteria.items():
            if isinstance(result, bool):
                continue
            
            criterion_names.append(criterion.replace('_', ' ').title())
            values.append(result)
            
            if 'mae_1mi' in criterion:
                targets.append(self.target_mae_1mi)
                passed.append(result <= self.target_mae_1mi)
            elif 'mae_3mi' in criterion:
                targets.append(self.target_mae_3mi)
                passed.append(result <= self.target_mae_3mi)
            elif 'auc' in criterion:
                targets.append(self.target_auc)
                passed.append(result >= self.target_auc)
            else:
                targets.append(0)
                passed.append(True)
        
        if not criterion_names:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(criterion_names))
        colors = ['#2ca02c' if p else '#d62728' for p in passed]
        
        bars = ax.bar(x, values, color=colors, alpha=0.7)
        ax.plot(x, targets, 'r--', linewidth=2, label='Target')
        
        ax.set_xlabel('Criteria')
        ax.set_ylabel('Value')
        ax.set_title('Acceptance Criteria Evaluation')
        ax.set_xticks(x)
        ax.set_xticklabels(criterion_names, rotation=45, ha='right')
        ax.legend()
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'acceptance_criteria.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run complete evaluation.
        
        Returns:
            Dictionary of evaluation results
        """
        logger.info("Starting enhanced model evaluation...")
        
        # Generate report
        report = self.generate_performance_report()
        print(report)
        
        # Save report
        report_path = "evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {report_path}")
        
        # Create visualizations
        self.create_visualizations()
        
        # Evaluate acceptance criteria
        criteria = self.evaluate_acceptance_criteria()
        
        # Summary
        total_criteria = len([c for c in criteria.values() if isinstance(c, bool)])
        passed_criteria = sum([c for c in criteria.values() if isinstance(c, bool)])
        
        logger.info(f"Acceptance criteria: {passed_criteria}/{total_criteria} passed")
        
        return {
            'report': report,
            'criteria': criteria,
            'passed_criteria': passed_criteria,
            'total_criteria': total_criteria,
            'all_passed': passed_criteria == total_criteria
        }


def main():
    """Main function to run evaluation."""
    try:
        evaluator = EnhancedModelEvaluator()
        results = evaluator.run_evaluation()
        
        if results['all_passed']:
            print("\n🎉 ALL ACCEPTANCE CRITERIA PASSED!")
            return 0
        else:
            print(f"\n⚠️  {results['passed_criteria']}/{results['total_criteria']} ACCEPTANCE CRITERIA PASSED")
            return 1
            
    except Exception as e:
        print(f"\n❌ EVALUATION FAILED: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
