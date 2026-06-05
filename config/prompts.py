BALL_TYPES = {
    0: "EMPTY",
    1: "LIGHT",
    2: "NORMAL", 
    3: "HEAVY"
}


BALL_SYMBOLS = {
    0: "□",   
    1: "○",
    2: "●",  
    3: "⬤"    
}


OBSERVATION_PROMPT = """You are an Observation Agent in a container mixing simulation.

Current Container State (10x10 grid, showing ball positions):
{container_state}

Ball Types:
- LIGHT (1): Light balls that tend to float upward
- NORMAL (2): Regular balls that stay in place
- HEAVY (3): Heavy balls that tend to sink downward
- EMPTY (0): Empty cells

Current Homogeneity Score: {homogeneity_score:.2f} (0-1 scale, 1 is perfectly mixed)

Analyze this state and provide your observations in this exact format:

DISTRIBUTION SUMMARY: (where are different ball types located? are they clustered?)
HOMOGENEITY ASSESSMENT: (how well mixed are they? what's the current score telling us?)
KEY ISSUES: (what's preventing even distribution?)
NOTABLE PATTERNS: (any interesting patterns you notice?)

Observation:"""



SUMMARIZATION_PROMPT = """You are a Summarization Agent.

Simulation Actions History:
{simulation_history}

Final Container State:
{final_state}

Final Homogeneity Score: {final_score:.2f}
Target Goal: {target_goal}
Total Steps Taken: {total_steps}

Create a concise summary:

ACTIONS SEQUENCE: (list the main actions taken in order)
KEY DECISIONS: (what were the critical decisions and why?)
RESULT ANALYSIS: (was the goal achieved? why or why not?)
RECOMMENDATIONS: (what would you do differently next time?)

Summary:"""