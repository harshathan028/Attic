"""
FactCheck Agent - Verifies claims and flags weak statements.
"""

import logging
from .base_agent import BaseAgent, AgentContext
from tools.llm_client import LLMClient

logger = logging.getLogger(__name__)


class FactCheckAgent(BaseAgent):
    """Agent that verifies claims and suggests corrections."""

    def __init__(self, llm_client: LLMClient, **kwargs):
        super().__init__(
            name="factcheck_agent",
            role="Fact Checker",
            llm_client=llm_client,
            prompt_file="factcheck.txt",
            **kwargs,
        )

    def run(self, context: AgentContext) -> str:
        logger.info(f"[{self.name}] Fact-checking content")
        
        draft = context.draft_content
        if not draft:
            return "No content to fact-check."

        research = context.research_notes
        result = self._verify_content(draft, research, context.topic)
        context.fact_check_results = result
        return result

    def _verify_content(self, draft: str, research: str, topic: str) -> str:
        prompt = self.get_prompt(topic=topic, draft=draft, research=research)
        if not prompt:
            prompt = f"""Fact-check this article about "{topic}":

ARTICLE:
{draft}

RESEARCH:
{research}

For each claim:
1. Verify against research
2. Flag unsupported statements
3. Suggest corrections

Provide the corrected article with [VERIFIED], [NEEDS CITATION], or [CORRECTED] tags."""

        system_prompt = "You are a meticulous fact-checker ensuring accuracy and credibility."
        try:
            return self.llm_client.generate(prompt=prompt, system_prompt=system_prompt, max_tokens=3000)
        except Exception as e:
            logger.error(f"Fact-check failed: {e}")
            return draft
