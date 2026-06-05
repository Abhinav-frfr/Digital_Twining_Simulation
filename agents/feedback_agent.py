import json
import os
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from datetime import datetime
from .base_agent import BaseAgent
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeedbackAgent(BaseAgent):
    
    def __init__(self, model: str = "llama3.1:8b", memory_file: str = None):
        super().__init__("Feedback Agent", model)
        self.memory_file = None
        self.action_history = []
        self.action_rewards = defaultdict(list)
        self.action_outcomes = defaultdict(list)
        self.bad_actions = set()
        self.consecutive_bad_actions = 0
        logger.info("Enhanced Feedback Agent initialized with outcome categorization")
    
    def categorize_outcome(self, delta: float) -> Tuple[str, str]:

        if delta > 0.05:
            return "strong_success", f"Strong improvement (+{delta:.3f})"
        elif delta > 0:
            return "weak_success", f"Small improvement (+{delta:.3f})"
        elif delta > -0.05:
            return "neutral_or_noise", f"Negligible change ({delta:+.3f})"
        elif delta > -0.15:
            return "moderate_failure", f"Moderate decrease ({delta:+.3f})"
        else:
            return "severe_failure", f"Severe decrease ({delta:+.3f})"
    
    def record_action(self, action: str, parameters: Dict, score_before: float, 
                      score_after: float, reason: str) -> float:
       
        delta = score_after - score_before
        outcome, outcome_desc = self.categorize_outcome(delta)
        
        reward = self._calculate_reward(delta, outcome)
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "parameters": parameters,
            "score_before": score_before,
            "score_after": score_after,
            "delta": delta,
            "outcome": outcome,
            "outcome_description": outcome_desc,
            "reward": reward,
            "reason": reason
        }
        
        self.action_history.append(record)
        
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]
        
        self.action_rewards[action].append(reward)
        self.action_outcomes[action].append(outcome)
        
        if outcome in ["moderate_failure", "severe_failure"]:
            self.bad_actions.add(action)
            self.consecutive_bad_actions += 1
        else:
            self.consecutive_bad_actions = 0
        
        # Clearing of bad action if it later succeeds
        if outcome in ["strong_success", "weak_success"] and action in self.bad_actions:
            # We will only remove it if it has recent successes
            recent_outcomes = self.action_outcomes[action][-3:]
            if all(o in ["strong_success", "weak_success"] for o in recent_outcomes):
                self.bad_actions.discard(action)
        
        logger.info(f"Action recorded: {action} | Delta: {delta:+.3f} | Outcome: {outcome} | Reward: {reward:.2f}")
        
        return reward
    
    def _calculate_reward(self, delta: float, outcome: str) -> float:
        reward_map = {
            "strong_success": min(delta * 15, 5.0),
            "weak_success": delta * 8,
            "neutral_or_noise": 0.0,
            "moderate_failure": delta * 12,
            "severe_failure": delta * 20     
        }
        return reward_map.get(outcome, 0.0)
    
    def get_action_confidence(self, action: str) -> Tuple[str, float, str]:

        if action not in self.action_rewards or not self.action_rewards[action]:
            return "unknown", 0.0, "No data yet - try this action"
        
        avg_reward = sum(self.action_rewards[action]) / len(self.action_rewards[action])
        
        if avg_reward > 0.05:
            return "high", avg_reward, "Highly recommended - This action consistently improves score"
        elif avg_reward > 0:
            return "medium", avg_reward, "Slightly positive - This may or may not help"
        elif avg_reward > -0.05:
            return "low", avg_reward, "Neutral - This action has uncertain effect"
        else:
            return "very_low", avg_reward, "Avoid - This action consistently decreases score"
    
    def get_action_summary(self) -> str:
        if not self.action_rewards:
            return "No action data available yet."
        
        summary = "\n=== ACTION PERFORMANCE SUMMARY ===\n"
        
        action_performance = []
        for action in self.action_rewards:
            if self.action_rewards[action]:
                avg_reward = sum(self.action_rewards[action]) / len(self.action_rewards[action])
                confidence, _, rec = self.get_action_confidence(action)
                
                outcomes = self.action_outcomes.get(action, [])
                strong_success = outcomes.count("strong_success")
                weak_success = outcomes.count("weak_success")
                moderate_fail = outcomes.count("moderate_failure")
                severe_fail = outcomes.count("severe_failure")
                
                action_performance.append({
                    "action": action,
                    "avg_reward": avg_reward,
                    "confidence": confidence,
                    "strong_success": strong_success,
                    "weak_success": weak_success,
                    "moderate_fail": moderate_fail,
                    "severe_fail": severe_fail,
                    "recommendation": rec
                })
        
        # Sorting all the actions by performance
        action_performance.sort(key=lambda x: x["avg_reward"], reverse=True)
        
        for perf in action_performance:
            # Color code based on confidence
            if perf["confidence"] == "high":
                symbol = "✅✅"
            elif perf["confidence"] == "medium":
                symbol = "✅"
            elif perf["confidence"] == "low":
                symbol = "⚠️"
            else:
                symbol = "❌"
            
            summary += f"\n{symbol} {perf['action']}: Avg Reward: {perf['avg_reward']:.3f} ({perf['confidence']} confidence)\n"
            summary += f"   Successes: {perf['strong_success']} strong, {perf['weak_success']} weak\n"
            summary += f"   Failures: {perf['moderate_fail']} moderate, {perf['severe_fail']} severe\n"
            summary += f"   → {perf['recommendation']}\n"
        
        return summary
    
    def get_severe_failures(self) -> List[Dict]:
        return [a for a in self.action_history[-20:] if a.get('outcome') == 'severe_failure']
    
    def get_recent_trend(self) -> str:
        if len(self.action_history) < 3:
            return "Not enough data"
        
        recent = self.action_history[-5:]
        improvements = sum(1 for a in recent if a['delta'] > 0)
        decreases = sum(1 for a in recent if a['delta'] < 0)
        
        if improvements > decreases:
            return "📈 Improving trend - current strategy working"
        elif decreases > improvements:
            return "📉 Declining trend - need strategy change"
        else:
            return "📊 Mixed results - continue exploring"
    
    def get_best_action(self) -> Tuple[str, float]:
        best_action = None
        best_reward = -float('inf')
        
        for action, rewards in self.action_rewards.items():
            if rewards:
                avg_reward = sum(rewards) / len(rewards)
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    best_action = action
        
        return best_action, best_reward
    
    def should_avoid_action(self, action: str, score: float) -> Tuple[bool, str]:

        if action not in self.action_rewards or not self.action_rewards[action]:
            return False, "No data yet"
        
        confidence, avg_reward, rec = self.get_action_confidence(action)
        
        if confidence == "very_low":
            return True, f"Consistently decreases score (avg: {avg_reward:.3f})"
        
        recent_failures = [a for a in self.action_history[-5:] 
                          if a['action'] == action and a.get('outcome') == 'severe_failure']
        if recent_failures:
            return True, f"Recent severe failure ({len(recent_failures)} times)"
        
        return False, rec
    
    def save_memory(self):
        try:
            memory = {
                "action_history": self.action_history[-50:],
                "action_rewards": {k: v for k, v in self.action_rewards.items()},
                "action_outcomes": {k: v for k, v in self.action_outcomes.items()},
                "bad_actions": list(self.bad_actions),
                "consecutive_bad_actions": self.consecutive_bad_actions
            }
            
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            with open(self.memory_file, 'w') as f:
                json.dump(memory, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
    
    def load_memory(self):
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    memory = json.load(f)
                    self.action_history = memory.get("action_history", [])
                    self.action_rewards = defaultdict(list, memory.get("action_rewards", {}))
                    self.action_outcomes = defaultdict(list, memory.get("action_outcomes", {}))
                    self.bad_actions = set(memory.get("bad_actions", []))
                    self.consecutive_bad_actions = memory.get("consecutive_bad_actions", 0)
                logger.info(f"Loaded {len(self.action_history)} actions from memory")
        except Exception as e:
            logger.warning(f"Failed to load memory: {e}")
    
    def get_statistics(self) -> Dict:
        stats = {
            "total_actions": len(self.action_history),
            "successful_actions": sum(1 for a in self.action_history if a['delta'] > 0),
            "failed_actions": sum(1 for a in self.action_history if a['delta'] < 0),
            "strong_successes": sum(1 for a in self.action_history if a.get('outcome') == 'strong_success'),
            "weak_successes": sum(1 for a in self.action_history if a.get('outcome') == 'weak_success'),
            "moderate_failures": sum(1 for a in self.action_history if a.get('outcome') == 'moderate_failure'),
            "severe_failures": sum(1 for a in self.action_history if a.get('outcome') == 'severe_failure'),
            "best_action": None,
            "worst_action": None,
            "recent_trend": self.get_recent_trend(),
            "action_performance": {}
        }
        
        for action, rewards in self.action_rewards.items():
            if rewards:
                stats["action_performance"][action] = sum(rewards) / len(rewards)
        
        if stats["action_performance"]:
            stats["best_action"] = max(stats["action_performance"], key=stats["action_performance"].get)
            stats["worst_action"] = min(stats["action_performance"], key=stats["action_performance"].get)
        
        return stats