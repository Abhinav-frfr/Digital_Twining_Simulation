# evaluation/visualizer.py
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
import os
from datetime import datetime

class SimulationVisualizer:
    
    def __init__(self, output_dir: str = "results/evaluation"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_score_progression(self, scores: List[float], title: str = "Score Progression"):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        steps = list(range(1, len(scores) + 1))
        
        ax.plot(steps, scores, 'b-', linewidth=2, label='Score')
        ax.axhline(y=0.7, color='g', linestyle='--', label='Target (0.7)')
        ax.axhline(y=0.5, color='y', linestyle='--', alpha=0.5, label='Good (0.5)')
        
        ax.fill_between(steps, scores, 0.7, where=[s >= 0.7 for s in scores], 
                        color='green', alpha=0.3, label='Goal Achieved')
        ax.fill_between(steps, scores, 0.5, where=[0.5 <= s < 0.7 for s in scores], 
                        color='yellow', alpha=0.2, label='Good Progress')
        
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Homogeneity Score', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        best_score = max(scores)
        best_step = scores.index(best_score) + 1
        ax.annotate(f'Best: {best_score:.3f}', 
                   xy=(best_step, best_score),
                   xytext=(best_step + 2, best_score + 0.05),
                   arrowprops=dict(arrowstyle='->', color='red'))
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"score_progression_{timestamp}.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def plot_action_performance(self, action_stats: Dict[str, Dict]):
        if not action_stats:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        actions = list(action_stats.keys())
        rewards = [action_stats[a]['avg_reward'] for a in actions]
        success_rates = [action_stats[a]['success_rate'] for a in actions]
        
        # Bar colors based on reward
        colors = ['green' if r > 0 else 'red' if r < 0 else 'gray' for r in rewards]
        
        # Reward bar chart
        bars1 = ax1.bar(actions, rewards, color=colors, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_xlabel('Action', fontsize=12)
        ax1.set_ylabel('Average Reward', fontsize=12)
        ax1.set_title('Action Rewards', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, reward in zip(bars1, rewards):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.05 if reward > 0 else -0.1),
                    f'{reward:.2f}', ha='center', va='bottom' if reward > 0 else 'top', fontsize=10)
        
        # Success rate bar chart
        colors2 = ['green' if r > 50 else 'orange' if r > 30 else 'red' for r in success_rates]
        bars2 = ax2.bar(actions, success_rates, color=colors2, alpha=0.7)
        ax2.set_xlabel('Action', fontsize=12)
        ax2.set_ylabel('Success Rate (%)', fontsize=12)
        ax2.set_title('Action Success Rates', fontsize=14)
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, rate in zip(bars2, success_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{rate:.0f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"action_performance_{timestamp}.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def plot_learning_curve(self, scores: List[float], deltas: List[float]):
        """Plot learning curve with improvements/decrements"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        steps = list(range(1, len(scores) + 1))
        
        # Score progression
        ax1.plot(steps, scores, 'b-', linewidth=2, label='Score')
        ax1.axhline(y=0.7, color='g', linestyle='--', label='Target')
        ax1.set_xlabel('Step', fontsize=12)
        ax1.set_ylabel('Homogeneity Score', fontsize=12)
        ax1.set_title('Learning Curve - Score Progression', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Score changes (improvements vs decrements)
        colors = ['green' if d > 0 else 'red' if d < 0 else 'gray' for d in deltas]
        ax2.bar(steps, deltas, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Step', fontsize=12)
        ax2.set_ylabel('Score Change', fontsize=12)
        ax2.set_title('Learning Curve - Step Improvements', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Add cumulative improvement annotation
        total_improvement = sum(deltas)
        ax2.text(0.02, 0.95, f'Total Improvement: {total_improvement:+.3f}', 
                transform=ax2.transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"learning_curve_{timestamp}.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def create_evaluation_dashboard(self, report: Dict[str, Any], scores: List[float], 
                                     deltas: List[float], action_stats: Dict):
        """Create a complete evaluation dashboard"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Score Progression
        ax1 = axes[0, 0]
        steps = list(range(1, len(scores) + 1))
        ax1.plot(steps, scores, 'b-', linewidth=2)
        ax1.axhline(y=0.7, color='g', linestyle='--', label='Target (0.7)')
        ax1.fill_between(steps, scores, 0.7, where=[s >= 0.7 for s in scores], 
                         color='green', alpha=0.3)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Score')
        ax1.set_title('Score Progression')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # 2. Improvements/Decrements
        ax2 = axes[0, 1]
        colors = ['green' if d > 0 else 'red' if d < 0 else 'gray' for d in deltas]
        ax2.bar(steps, deltas, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linewidth=0.5)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Change')
        ax2.set_title('Step Improvements')
        ax2.grid(True, alpha=0.3)
        
        # 3. Action Performance
        ax3 = axes[1, 0]
        if action_stats:
            actions = list(action_stats.keys())
            rewards = [action_stats[a]['avg_reward'] for a in actions]
            colors3 = ['green' if r > 0 else 'red' if r < 0 else 'gray' for r in rewards]
            ax3.bar(actions, rewards, color=colors3, alpha=0.7)
            ax3.axhline(y=0, color='black', linewidth=0.5)
            ax3.set_xlabel('Action')
            ax3.set_ylabel('Avg Reward')
            ax3.set_title('Action Performance')
            ax3.grid(True, alpha=0.3)
            
            # Add value labels
            for i, (bar, reward) in enumerate(zip(ax3.patches, rewards)):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.02 if reward > 0 else -0.05),
                        f'{reward:.2f}', ha='center', va='bottom' if reward > 0 else 'top', fontsize=9)
        
        # 4. Summary Stats
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""
        📊 EVALUATION SUMMARY
        ─────────────────────────────────
        
        Initial Score: {report['summary']['initial_score']:.3f}
        Final Score: {report['summary']['final_score']:.3f}
        Best Score: {report['summary']['best_score']:.3f}
        Total Improvement: {report['summary']['total_improvement']:+.3f}
        
        Success Rate: {report['summary']['success_rate']:.1f}%
        Overall Score: {report['summary']['overall_score']:.1f}/100
        
        Learning Rate: {report['learning_metrics'].get('learning_rate', 0):+.4f}
        Action Diversity: {report['efficiency_metrics'].get('action_diversity', 0):.1f}%
        
        Best Action: {report['action_performance'].get('best_action', 'N/A')}
        Worst Action: {report['action_performance'].get('worst_action', 'N/A')}
        """
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.suptitle('LLM Simulation Agent - Evaluation Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"dashboard_{timestamp}.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filename