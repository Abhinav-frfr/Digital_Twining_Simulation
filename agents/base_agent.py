import logging
from typing import Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseAgent:
    
    def __init__(self, name: str, model: str = "llama3.1:8b", temperature: float = 0.7):
        self.name = name
        self.model = model
        self.temperature = temperature
        self.llm = ChatOllama(
            model=model,
            temperature=temperature,
            base_url="http://localhost:11434"
        )
        logger.info(f"Initialized {name} Agent with model: {model}")
    
    def call_llm(self, prompt: str) -> str:

        try:
            messages = [
                SystemMessage(content=f"You are a {self.name}. Be analytical and precise."),
                HumanMessage(content=prompt)
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error in {self.name}: {str(e)}")
            return ""
    
    def process(self, input_data: Any) -> Any:
        raise NotImplementedError