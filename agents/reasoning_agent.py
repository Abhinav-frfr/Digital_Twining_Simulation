import re
import logging
from typing import Dict, Any
from .base_agent import BaseAgent
from config.prompts import BALL_SYMBOLS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReasoningAgent(BaseAgent):
    
    def __init__(self, model: str = "llama3.1:8b"):
        super().__init__("Reasoning Agent", model)
        self.feedback_agent = None
    
    def set_feedback_agent(self, feedback_agent):
        self.feedback_agent = feedback_agent
    
    def process(self, observation: Dict[str, Any], container_state: list, 
                homogeneity_score: float, target_goal: str) -> Dict[str, Any]:
        
        state_text = self._format_state_simple(container_state)
        
        obs_text = f"""
Distribution: {observation.get('distribution_summary', 'N/A')}
Homogeneity: {observation.get('homogeneity_assessment', 'N/A')}
Issues: {observation.get('key_issues', 'N/A')}
"""
        
        feedback_text = ""
        if self.feedback_agent and self.feedback_agent.action_history:
            last_action = self.feedback_agent.action_history[-1]
            delta = last_action.get('delta', 0)
            outcome = last_action.get('outcome', 'unknown')
            
            if delta > 0:
                feedback_text = f"Last action ({last_action['action']}) was GOOD! Score increased by {delta:.3f} ({outcome})"
            elif delta < 0:
                feedback_text = f"Last action ({last_action['action']}) was BAD! Score decreased by {abs(delta):.3f} ({outcome})"
            else:
                feedback_text = f"Last action ({last_action['action']}) had no significant effect"
        
        avoid_text = ""
        if self.feedback_agent:
            for action in ["SHAKE", "ADD_LIGHT", "ADD_NORMAL", "ADD_HEAVY"]:
                should_avoid, reason = self.feedback_agent.should_avoid_action(action, homogeneity_score)
                if should_avoid:
                    avoid_text += f"\n- AVOID {action}: {reason}"
        
        confidence_summary = ""
        if self.feedback_agent:
            confidence_summary = self.feedback_agent.get_action_summary()
        
        prompt = f"""Analyze this container mixing simulation:

State:
{state_text}

Current score: {homogeneity_score:.3f}
Goal: {target_goal}

Observations:
{obs_text}

{feedback_text}

{avoid_text}

{confidence_summary}

Provide your reasoning:
1. ANALYSIS: What's the problem with current distribution?
2. STRATEGY: What should we do? (Consider the feedback above)
3. ACTIONS: What specific action should we take?
4. OUTCOME: What will happen after this action?

Actions that can be taken:-
1. "SHAKE" 
2. "ADD_LIGHT" 
3. "ADD_NORMAL" 
4. "ADD_HEAVY"

Answer in this format:
ANALYSIS: [your analysis]
STRATEGY: [your strategy]
ACTIONS: [recommended actions]
OUTCOME: [expected outcome]"""
        
        response = self.call_llm(prompt)
        
        if not response:
            return self._get_fallback_reasoning(homogeneity_score)
        
        reasoning = self._parse_reasoning(response)
        
        return reasoning
    
    def _format_state_simple(self, state: list) -> str:
        light = sum(row.count(1) for row in state)
        normal = sum(row.count(2) for row in state)
        heavy = sum(row.count(3) for row in state)
        
        top_half = state[:5]
        bottom_half = state[5:]
        
        light_top = sum(row.count(1) for row in top_half)
        heavy_bottom = sum(row.count(3) for row in bottom_half)
        
        return f"""Balls: Light={light}, Normal={normal}, Heavy={heavy}
Light balls in top half: {light_top}
Heavy balls in bottom half: {heavy_bottom}"""
    
    def _parse_reasoning(self, response: str) -> Dict[str, Any]:
        reasoning = {
            "analysis": "",
            "proposed_strategy": "",
            "recommended_actions": "",
            "expected_outcome": "",
            "raw_response": response
        }
        
        sections = {
            "ANALYSIS": "analysis",
            "STRATEGY": "proposed_strategy",
            "ACTIONS": "recommended_actions",
            "OUTCOME": "expected_outcome"
        }
        
        for section, key in sections.items():
            pattern = rf"{section}:\s*(.*?)(?=\n[A-Z]|\Z)"
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                reasoning[key] = match.group(1).strip()[:500]
        
        if not reasoning["analysis"]:
            reasoning["analysis"] = "Need to mix balls for better distribution"
        if not reasoning["proposed_strategy"]:
            reasoning["proposed_strategy"] = "Shake the container to mix"
        if not reasoning["recommended_actions"]:
            reasoning["recommended_actions"] = "SHAKE with duration 15"
        if not reasoning["expected_outcome"]:
            reasoning["expected_outcome"] = "Better distribution after shaking"
        
        return reasoning
    
    def _get_fallback_reasoning(self, score: float) -> Dict[str, Any]:
        """Return fallback reasoning if LLM fails"""
        if score < 0.3:
            strategy = "Shake more vigorously"
            actions = "SHAKE duration 20"
        elif score < 0.5:
            strategy = "Continue shaking"
            actions = "SHAKE duration 15"
        else:
            strategy = "Add more balls"
            actions = "ADD_NORMAL count 3"
        
        return {
            "analysis": f"Current homogeneity is {score:.3f}",
            "proposed_strategy": strategy,
            "recommended_actions": actions,
            "expected_outcome": "Improved score",
            "raw_response": ""
        }