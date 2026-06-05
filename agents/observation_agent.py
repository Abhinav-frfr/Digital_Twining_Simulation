import re
from typing import Dict, Any
from .base_agent import BaseAgent
from config.prompts import OBSERVATION_PROMPT, BALL_SYMBOLS

class ObservationAgent(BaseAgent):
    """This Agent observes and analyzes simulation state"""
    
    def __init__(self, model: str = "llama3.1:8b"):
        super().__init__("Observation Agent", model)
    
    def process(self, container_state: list, homogeneity_score: float) -> Dict[str, Any]:
        
        state_text = self._format_state_visual(container_state)
        
        prompt = OBSERVATION_PROMPT.format(
            container_state=state_text,
            homogeneity_score=homogeneity_score
        )
        
        response = self.call_llm(prompt)
        
        if not response:
            return self._get_fallback_observation(container_state, homogeneity_score)
        
        observation = self._parse_observation(response,homogeneity_score)
        observation["homogeneity_score"] = homogeneity_score
        
        return observation
    
    def _format_state_visual(self, state: list) -> str:

        formatted = []
        formatted.append("Container Grid (10x10):")
        formatted.append("    " + " ".join([f"{i:2}" for i in range(10)]))
        formatted.append("   " + "-" * 30)
        
        for i, row in enumerate(state):
            row_text = f"{i:2} | "
            for cell in row:
                row_text += f" {BALL_SYMBOLS.get(cell, '?')}"
            formatted.append(row_text)
        
        return "\n".join(formatted)
    
    def _parse_observation(self, response: str, homogeneity_score: float) -> Dict[str, Any]:

        observation = {
            "distribution_summary": "",
            "homogeneity_assessment": "",
            "key_issues": "",
            "notable_patterns": "",
            "raw_response": response
        }
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line_lower = line.lower()
            if 'distribution' in line_lower:
                current_section = 'distribution_summary'
                observation[current_section] = line.split(':', 1)[-1].strip() if ':' in line else ''
            elif 'homogeneity' in line_lower or 'assessment' in line_lower:
                current_section = 'homogeneity_assessment'
                observation[current_section] = line.split(':', 1)[-1].strip() if ':' in line else ''
            elif 'issue' in line_lower:
                current_section = 'key_issues'
                observation[current_section] = line.split(':', 1)[-1].strip() if ':' in line else ''
            elif 'pattern' in line_lower:
                current_section = 'notable_patterns'
                observation[current_section] = line.split(':', 1)[-1].strip() if ':' in line else ''
            elif current_section and line.strip():
                observation[current_section] += ' ' + line.strip()
        
        if not observation["distribution_summary"]:
            observation["distribution_summary"] = "Balls are clustered in groups"
        if not observation["homogeneity_assessment"]:
            observation["homogeneity_assessment"] = f"Score is {homogeneity_score:.3f}"
        if not observation["key_issues"]:
            observation["key_issues"] = "Balls are not evenly distributed"
        if not observation["notable_patterns"]:
            observation["notable_patterns"] = "Similar balls are grouped together"
        
        return observation
    
    def _get_fallback_observation(self, state: list, score: float) -> Dict[str, Any]:
        
        light_count = sum(row.count(1) for row in state)
        normal_count = sum(row.count(2) for row in state)
        heavy_count = sum(row.count(3) for row in state)
        
        return {
            "distribution_summary": f"Light: {light_count}, Normal: {normal_count}, Heavy: {heavy_count}",
            "homogeneity_assessment": f"Score is {score:.3f}",
            "key_issues": "Balls are clustered together",
            "notable_patterns": "Similar balls are grouped",
            "raw_response": ""
        }