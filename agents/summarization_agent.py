import re
from typing import Dict, Any, List
from .base_agent import BaseAgent

class SummarizationAgent(BaseAgent):
    """This Agent summarizes the simulation results"""
    
    def __init__(self, model: str = "llama3.1:8b"):
        super().__init__("Summarization Agent", model)
    
    def process(self, history: List[Dict], final_state: list, 
                final_score: float, target_goal: str) -> Dict[str, Any]:
        
        history_text = self._format_history(history)
        
        prompt = f"""Summarize this simulation:

Actions taken:
{history_text}

Final score: {final_score:.3f}
Target: {target_goal}

Provide:
ACTIONS: (what was done)
RESULT: (was goal achieved)
RECOMMEND: (what to improve)"""
        
        response = self.call_llm(prompt)
        
        if not response:
            return self._get_fallback_summary(final_score, len(history))
        
        summary = self._parse_summary(response)
        summary["total_steps"] = len(history)
        summary["final_score"] = final_score
        
        return summary
    
    def _format_history(self, history: List[Dict]) -> str:
        if not history:
            return "No actions taken."
        
        formatted = []
        for step in history[-10:]:
            formatted.append(f"Step {step['step']}: {step['action']} - Score: {step['score_after']:.3f}")
        return "\n".join(formatted)
    
    def _parse_summary(self, response: str) -> Dict[str, Any]:
        summary = {
            "actions_sequence": "",
            "result_analysis": "",
            "recommendations": "",
            "raw_response": response
        }
        
        sections = {
            "ACTIONS": "actions_sequence",
            "RESULT": "result_analysis",
            "RECOMMEND": "recommendations"
        }
        
        for section, key in sections.items():
            pattern = rf"{section}:\s*(.*?)(?=\n[A-Z]|\Z)"
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                summary[key] = match.group(1).strip()[:300]
        
        return summary
    
    def _get_fallback_summary(self, score: float, steps: int) -> Dict[str, Any]:
        return {
            "actions_sequence": f"Took {steps} steps to reach score {score:.3f}",
            "key_decisions": "Mainly used SHAKE actions to mix balls",
            "result_analysis": f"Achieved {score:.3f} score",
            "recommendations": "Try different ball addition sequences",
            "total_steps": steps,
            "final_score": score,
            "raw_response": ""
        }