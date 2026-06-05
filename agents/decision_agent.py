import re
from typing import Dict, Any, Tuple, List
from .base_agent import BaseAgent
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DecisionAgent(BaseAgent):
    """This Agent makes decisions using LLM with enhanced learning"""
    
    def __init__(self, model: str = "llama3.1:8b"):
        super().__init__("Decision Agent", model)
        self.feedback_agent = None
        self.consecutive_decreases = 0
    
    def set_feedback_agent(self, feedback_agent):
        self.feedback_agent = feedback_agent
    
    def process(self, reasoning: Dict[str, Any], container_state: list,
            homogeneity_score: float, target_score: float,  
            simulation=None) -> Tuple[str, Dict[str, Any], str]:
    
        strategy = reasoning.get('proposed_strategy', '')
    
        learning_insights = self._get_enhanced_learning_insights()
    
        action_confidence = self._get_action_confidence_summary()
    
        prompt = self._build_enhanced_prompt(
        strategy, homogeneity_score, target_score,  
        learning_insights, action_confidence
        )
    
        response = self.call_llm(prompt)
    
        if not response:
            return self._get_adaptive_fallback(homogeneity_score)
    
        action, parameters, reason = self._parse_decision(response)
    
        return action, parameters, reason
    
    def _get_enhanced_learning_insights(self) -> str:
        if not self.feedback_agent or not self.feedback_agent.action_history:
            return "No past actions yet. Explore different strategies."
        
        insights = "\n" + "="*60 + "\n"
        insights += "DETAILED LEARNING FROM PAST ACTIONS\n"
        insights += "="*60 + "\n"
        
        strong_successes = [a for a in self.feedback_agent.action_history 
                           if a.get('outcome') == 'strong_success']
        weak_successes = [a for a in self.feedback_agent.action_history 
                         if a.get('outcome') == 'weak_success']
        moderate_failures = [a for a in self.feedback_agent.action_history 
                            if a.get('outcome') == 'moderate_failure']
        severe_failures = [a for a in self.feedback_agent.action_history 
                          if a.get('outcome') == 'severe_failure']
        
        if strong_successes:
            insights += "\n STRONG SUCCESSES (score increased >0.05):\n"
            seen = set()
            for a in strong_successes[-3:]:
                if a['action'] not in seen:
                    insights += f"  • {a['action']}: +{a['delta']:.3f} → {a['outcome_description']}\n"
                    seen.add(a['action'])
            insights += "  → REPEAT these actions!\n"
        
        if weak_successes:
            insights += "\n WEAK SUCCESSES (score increased <0.05):\n"
            seen = set()
            for a in weak_successes[-3:]:
                if a['action'] not in seen:
                    insights += f"  • {a['action']}: +{a['delta']:.3f} → {a['outcome_description']}\n"
                    seen.add(a['action'])
            insights += "  → Can use, but try to improve\n"
        
        if moderate_failures:
            insights += "\n MODERATE FAILURES (score decreased 0.05-0.15):\n"
            seen = set()
            for a in moderate_failures[-3:]:
                if a['action'] not in seen:
                    insights += f"  • {a['action']}: {a['delta']:.3f} → {a['outcome_description']}\n"
                    insights += f"    Try VARIATION: different duration/count\n"
                    seen.add(a['action'])
        
        if severe_failures:
            insights += "\n SEVERE FAILURES (score decreased >0.15):\n"
            seen = set()
            for a in severe_failures[-3:]:
                if a['action'] not in seen:
                    insights += f"  • {a['action']}: {a['delta']:.3f} → {a['outcome_description']}\n"
                    insights += f"  → Strongly avoid UNLESS exploring new strategy or conditions changed\n"
                    seen.add(a['action'])
        
       
        insights += f"\n RECENT TREND: {self.feedback_agent.get_recent_trend()}\n"
        
        if self.consecutive_decreases >= 2:
            insights += f"\n{self.consecutive_decreases} consecutive decreases! TRY SOMETHING COMPLETELY DIFFERENT!\n"
        
        return insights
    
    def _get_action_confidence_summary(self) -> str:
        if not self.feedback_agent:
            return ""
        
        summary = "\n🎯 ACTION CONFIDENCE LEVELS:\n"
        summary += "-"*40 + "\n"
        
        actions = ["SHAKE", "ADD_LIGHT", "ADD_NORMAL", "ADD_HEAVY"]
        
        for action in actions:
            confidence, avg_reward, recommendation = self.feedback_agent.get_action_confidence(action)
            
            if confidence == "high":
                icon = "✅✅"
            elif confidence == "medium":
                icon = "✅"
            elif confidence == "low":
                icon = "⚠️"
            elif confidence == "very_low":
                icon = "❌"
            else:
                icon = "❓"
            
            summary += f"{icon} {action}: {confidence.upper()} confidence (avg: {avg_reward:+.3f})\n"
            summary += f"   → {recommendation}\n"
        
        return summary
    
    def _build_enhanced_prompt(self, strategy: str, score: float, 
                            target_score: float, insights: str, confidence: str) -> str:
    
        if score < 0.3:
            strategy_hint = "Score is VERY LOW. Focus on SHAKE actions to mix."
        elif score < 0.5:
            strategy_hint = "Score is LOW. Mixing is priority, but can experiment."
        elif score < 0.7:
            strategy_hint = "Score is MODERATE. Fine-tuning, consider adding balls."
        else:
            strategy_hint = "Score is HIGH. Small adjustments only."
    
        return f"""You are a Decision Agent that LEARNS from detailed outcome analysis.

Current Strategy: {strategy}
{strategy_hint}

Current Score: {score:.3f}
Target Score: {target_score:.3f}

{insights}

{confidence}

AVAILABLE ACTIONS:
- SHAKE (duration 10-30) - Mixes balls randomly
- ADD_LIGHT (count 1-5) - Adds lightweight balls (float up)
- ADD_NORMAL (count 1-5) - Adds normal weight balls
- ADD_HEAVY (count 1-5) - Adds heavyweight balls (sink down)

YOUR DECISION PROCESS:
1. Check confidence levels - prefer HIGH confidence actions
2. If recent trend is declining, CHANGE strategy
3. For severe failures - COMPLETELY AVOID those actions
4. For moderate failures - try VARIATIONS
5. For successes - REPEAT or similar

Output format:
ACTION: [action_name]
PARAMETERS: [duration or count]
REASON: [reference specific learning from past actions]

Decision:"""
    
    def _parse_decision(self, response: str) -> Tuple[str, Dict, str]:
        """Parse LLM response into action decision"""
        action = "SHAKE"
        parameters = {"duration": 15}
        reason = ""
        
        action_match = re.search(r"ACTION:\s*(\w+)", response, re.IGNORECASE)
        if action_match:
            action = action_match.group(1).strip().upper()
        
        params_match = re.search(r"PARAMETERS:\s*(.+)", response, re.IGNORECASE)
        if params_match:
            params_str = params_match.group(1).strip()
            if "duration" in params_str.lower():
                try:
                    duration = int(re.search(r"(\d+)", params_str).group(1))
                    parameters = {"duration": min(max(duration, 10), 30)}
                except:
                    parameters = {"duration": 15}
            elif "count" in params_str.lower():
                try:
                    count = int(re.search(r"(\d+)", params_str).group(1))
                    parameters = {"count": min(max(count, 1), 10)}
                except:
                    parameters = {"count": 3}
        
        reason_match = re.search(r"REASON:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
        if reason_match:
            reason = reason_match.group(1).strip()[:250]
        
        return action, parameters, reason
    
    def _get_adaptive_fallback(self, score: float) -> Tuple[str, Dict, str]:
        """Adaptive fallback using learning from feedback agent"""
        if not self.feedback_agent or not self.feedback_agent.action_rewards:
            if score < 0.3:
                return "SHAKE", {"duration": 20}, "Fallback: low score - need mixing"
            elif score < 0.5:
                return "SHAKE", {"duration": 15}, "Fallback: medium score - continue mixing"
            elif score < 0.7:
                return "ADD_NORMAL", {"count": 3}, "Fallback: good score - add balls"
            else:
                return "ADD_LIGHT", {"count": 2}, "Fallback: high score - fine tune"
    
        best_action, best_reward = self.feedback_agent.get_best_action()
    
        if best_action and best_reward > 0:
            if best_action == "SHAKE":
                return best_action, {"duration": 15}, f"Fallback: using best action (reward: {best_reward:.2f})"
            else:
                return best_action, {"count": 3}, f"Fallback: using best action (reward: {best_reward:.2f})"
    
        return "SHAKE", {"duration": 15}, "Fallback: default to shaking"