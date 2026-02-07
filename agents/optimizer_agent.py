"""
Optimizer Agent - Improves readability, SEO, and formatting.
"""

import logging
from .base_agent import BaseAgent, AgentContext
from tools.llm_client import LLMClient

logger = logging.getLogger(__name__)


class OptimizerAgent(BaseAgent):
    """Agent that optimizes content for readability and SEO."""

    def __init__(self, llm_client: LLMClient, **kwargs):
        super().__init__(
            name="optimizer_agent",
            role="Content Optimizer",
            llm_client=llm_client,
            prompt_file="optimize.txt",
            **kwargs,
        )

    def run(self, context: AgentContext) -> str:
        logger.info(f"[{self.name}] Optimizing content")

        content = context.fact_check_results or context.draft_content
        if not content:
            return "No content to optimize."

        # Evaluate current quality
        pre_score = self._evaluate_content(content)
        
        # Optimize content
        optimized = self._optimize_content(content, context.topic)
        
        # Evaluate optimized content
        post_score = self._evaluate_content(optimized)
        
        logger.info(f"Quality improved: {pre_score:.1f} -> {post_score:.1f}")
        context.optimized_content = optimized
        return optimized

    def _evaluate_content(self, content: str) -> float:
        if self.evaluator:
            score = self.evaluator.evaluate(content)
            return score.overall_score
        return 0.0

    def _optimize_content(self, content: str, topic: str) -> str:
        prompt = self.get_prompt(topic=topic, content=content)
        if not prompt:
            prompt = f"""Optimize this article about "{topic}" for readability and SEO:

{content}

Improvements needed:
1. Enhance readability (shorter sentences, simpler words)
2. Add SEO keywords naturally
3. Improve structure with clear headings
4. Add bullet points where appropriate
5. Ensure engaging introduction and conclusion
6. Format for web readability

Return the fully optimized article."""

        system_prompt = "You are an SEO and content optimization expert."
        try:
            return self.llm_client.generate(prompt=prompt, system_prompt=system_prompt, max_tokens=3000)
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return content
