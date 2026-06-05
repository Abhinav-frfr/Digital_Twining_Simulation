import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimulationMetrics:
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.scores = []
        self.actions = []
        self.deltas = []
        self.outcomes = []
        self.response_times = []
        self.action_rewards = defaultdict(list)
    
    def update(self, step_data: Dict[str, Any]):

        self.scores.append(step_data.get('score_after', 0))
        self.actions.append(step_data.get('action', 'unknown'))
        self.deltas.append(step_data.get('delta', 0))
        self.outcomes.append(step_data.get('outcome', 'unknown'))
        self.response_times.append(step_data.get('response_time', 0))
        
        action = step_data.get('action', 'unknown')
        reward = step_data.get('reward', 0)
        self.action_rewards[action].append(reward)
    
    def calculate_learning_efficiency(self) -> Dict[str, Any]:
    
        if len(self.scores) < 2:
            return {"error": "Not enough data"}
        
        initial_score = self.scores[0]
        final_score = self.scores[-1]
        best_score = max(self.scores)
        
        total_improvement = final_score - initial_score
        max_improvement = best_score - initial_score
        
        # Learning rate (average improvement per step)
        learning_rate = total_improvement / len(self.scores) if len(self.scores) > 0 else 0
        
        successful_actions = sum(1 for d in self.deltas if d > 0)
        failure_actions = sum(1 for d in self.deltas if d < 0)
        neutral_actions = sum(1 for d in self.deltas if d == 0)
        
        success_rate = successful_actions / len(self.deltas) if self.deltas else 0
        
        outcome_counts = defaultdict(int)
        for outcome in self.outcomes:
            outcome_counts[outcome] += 1
        
        return {
            "initial_score": round(initial_score, 4),
            "final_score": round(final_score, 4),
            "best_score": round(best_score, 4),
            "total_improvement": round(total_improvement, 4),
            "max_improvement": round(max_improvement, 4),
            "learning_rate": round(learning_rate, 4),
            "successful_actions": successful_actions,
            "failure_actions": failure_actions,
            "neutral_actions": neutral_actions,
            "success_rate": round(success_rate * 100, 2),
            "outcome_distribution": dict(outcome_counts),
            "total_steps": len(self.scores)
        }
    
    def calculate_action_performance(self) -> Dict[str, Any]:
        action_stats = {}
        
        for action, rewards in self.action_rewards.items():
            if rewards:
                avg_reward = np.mean(rewards)
                std_reward = np.std(rewards)
                min_reward = np.min(rewards)
                max_reward = np.max(rewards)
                count = len(rewards)
                
                action_deltas = [d for i, d in enumerate(self.deltas) 
                               if self.actions[i] == action]
                if action_deltas:
                    success_count = sum(1 for d in action_deltas if d > 0)
                    success_rate = success_count / len(action_deltas)
                    avg_delta = np.mean(action_deltas)
                else:
                    success_rate = 0
                    avg_delta = 0
                
                action_stats[action] = {
                    "avg_reward": round(avg_reward, 4),
                    "std_reward": round(std_reward, 4),
                    "min_reward": round(min_reward, 4),
                    "max_reward": round(max_reward, 4),
                    "count": count,
                    "success_rate": round(success_rate * 100, 2),
                    "avg_delta": round(avg_delta, 4)
                }

        if action_stats:
            best_action = max(action_stats.items(), key=lambda x: x[1]['avg_reward'])
            worst_action = min(action_stats.items(), key=lambda x: x[1]['avg_reward'])
        else:
            best_action = ("none", {})
            worst_action = ("none", {})
        
        return {
            "action_stats": action_stats,
            "best_action": best_action[0],
            "best_action_avg_reward": best_action[1].get('avg_reward', 0),
            "worst_action": worst_action[0],
            "worst_action_avg_reward": worst_action[1].get('avg_reward', 0)
        }
    
    def calculate_convergence_metrics(self) -> Dict[str, Any]:
        
        if len(self.scores) < 5:
            return {"error": "Not enough data for convergence analysis"}
        
        window_size = min(5, len(self.scores))
        rolling_avg = []
        for i in range(len(self.scores) - window_size + 1):
            window_avg = np.mean(self.scores[i:i+window_size])
            rolling_avg.append(window_avg)
        
        if len(rolling_avg) >= 3:
            last_3_avg = rolling_avg[-3:]
            variance = np.var(last_3_avg)
            stabilized = variance < 0.001
        else:
            stabilized = False
        
        convergence_step = len(self.scores)
        improvement_threshold = 0.01
        for i in range(len(self.scores) - 1, 0, -1):
            if abs(self.scores[i] - self.scores[i-1]) > improvement_threshold:
                convergence_step = i
                break
        
        return {
            "stabilized": stabilized,
            "convergence_step": convergence_step,
            "final_rolling_avg": round(rolling_avg[-1], 4) if rolling_avg else 0,
            "stability_variance": round(variance, 6) if 'variance' in locals() else 0
        }
    
    def calculate_efficiency_metrics(self) -> Dict[str, Any]:
        
        if not self.response_times:
            return {"error": "No response time data"}
        
        avg_response_time = np.mean(self.response_times)
        total_response_time = np.sum(self.response_times)
        max_response_time = np.max(self.response_times)
        
        unique_actions = len(set(self.actions))
        total_actions = len(self.actions)
        action_diversity = unique_actions / 4  # 4 is total number of possible actions
        
        if total_response_time > 0:
            score_improvement = self.scores[-1] - self.scores[0] if self.scores else 0
            efficiency_per_second = score_improvement / total_response_time
        else:
            efficiency_per_second = 0
        
        return {
            "avg_response_time": round(avg_response_time, 2),
            "total_response_time": round(total_response_time, 2),
            "max_response_time": round(max_response_time, 2),
            "unique_actions_used": unique_actions,
            "action_diversity": round(action_diversity * 100, 2),
            "efficiency_per_second": round(efficiency_per_second, 4)
        }
    
    def get_comprehensive_report(self) -> Dict[str, Any]:

        learning = self.calculate_learning_efficiency()
        action_perf = self.calculate_action_performance()
        convergence = self.calculate_convergence_metrics()
        efficiency = self.calculate_efficiency_metrics()
        
        overall_score = 0
        
        if 'success_rate' in learning:
            overall_score += learning['success_rate'] * 0.3
        
        # Total improvement contributes 25%
        if 'total_improvement' in learning:
            improvement_score = min(learning['total_improvement'] * 100, 100)
            overall_score += improvement_score * 0.25
        
        # Final score contributes 25%
        if 'final_score' in learning:
            final_score_percent = learning['final_score'] * 100
            overall_score += final_score_percent * 0.25
        
        # Action diversity contributes 10%
        if 'action_diversity' in efficiency:
            overall_score += efficiency['action_diversity'] * 0.1
        
        # Convergence contributes 10%
        if convergence.get('stabilized', False):
            overall_score += 10
        
        report = {
            "summary": {
                "total_steps": learning.get('total_steps', 0),
                "initial_score": learning.get('initial_score', 0),
                "final_score": learning.get('final_score', 0),
                "best_score": learning.get('best_score', 0),
                "total_improvement": learning.get('total_improvement', 0),
                "success_rate": learning.get('success_rate', 0),
                "overall_score": round(overall_score, 2)
            },
            "learning_metrics": learning,
            "action_performance": action_perf,
            "convergence_metrics": convergence,
            "efficiency_metrics": efficiency
        }
        
        return report
    
    def print_report(self):

        report = self.get_comprehensive_report()
        
        print("\n" + "="*70)
        print("📊 SIMULATION EVALUATION REPORT")
        print("="*70)
        

        print("\n📈 SUMMARY:")
        print(f"   Total Steps: {report['summary']['total_steps']}")
        print(f"   Initial Score: {report['summary']['initial_score']:.3f}")
        print(f"   Final Score: {report['summary']['final_score']:.3f}")
        print(f"   Best Score: {report['summary']['best_score']:.3f}")
        print(f"   Total Improvement: {report['summary']['total_improvement']:+.3f}")
        print(f"   Success Rate: {report['summary']['success_rate']:.1f}%")
        print(f"   Overall Score: {report['summary']['overall_score']:.1f}/100")
        

        print("\n🎓 LEARNING METRICS:")
        print(f"   Learning Rate: {report['learning_metrics'].get('learning_rate', 0):+.4f} per step")
        print(f"   Successful Actions: {report['learning_metrics'].get('successful_actions', 0)}")
        print(f"   Failed Actions: {report['learning_metrics'].get('failure_actions', 0)}")
        
       
        outcome_dist = report['learning_metrics'].get('outcome_distribution', {})
        if outcome_dist:
            print("\n📊 OUTCOME DISTRIBUTION:")
            for outcome, count in outcome_dist.items():
                emoji = {
                    'strong_success': '✅✅',
                    'weak_success': '✅',
                    'neutral_or_noise': '⚪',
                    'moderate_failure': '🟠',
                    'severe_failure': '🔴'
                }.get(outcome, '❓')
                print(f"   {emoji} {outcome}: {count} times")
        
        print("\n🎯 ACTION PERFORMANCE:")
        action_stats = report['action_performance'].get('action_stats', {})
        for action, stats in sorted(action_stats.items(), key=lambda x: x[1]['avg_reward'], reverse=True):
            reward = stats['avg_reward']
            if reward > 0.05:
                symbol = "✅✅"
            elif reward > 0:
                symbol = "✅"
            elif reward > -0.05:
                symbol = "⚠️"
            else:
                symbol = "❌"
            print(f"   {symbol} {action}: avg_reward={reward:+.3f}, success_rate={stats['success_rate']:.0f}%")
     
        print("\n⚡ EFFICIENCY METRICS:")
        print(f"   Avg Response Time: {report['efficiency_metrics'].get('avg_response_time', 0):.1f}s")
        print(f"   Action Diversity: {report['efficiency_metrics'].get('action_diversity', 0):.1f}%")
        print(f"   Efficiency per Second: {report['efficiency_metrics'].get('efficiency_per_second', 0):+.4f}")
        
        print("\n🎯 CONVERGENCE:")
        print(f"   Stabilized: {'✅ Yes' if report['convergence_metrics'].get('stabilized') else '❌ No'}")
        print(f"   Convergence Step: {report['convergence_metrics'].get('convergence_step', 'N/A')}")
        
        print("\n" + "="*70)
        
        return report