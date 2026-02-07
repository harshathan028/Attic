"""
Writer Agent - Drafts structured articles from research.
"""

import logging
from .base_agent import BaseAgent, AgentContext
from tools.llm_client import LLMClient

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """Writer agent that creates structured content from research."""

    def __init__(self, llm_client: LLMClient, **kwargs):
        super().__init__(
            name="writer_agent",
            role="Content Writer",
            llm_client=llm_client,
            prompt_file="writer.txt",
            **kwargs,
        )

    def run(self, context: AgentContext) -> str:
        topic = context.topic
        logger.info(f"[{self.name}] Writing article on: {topic}")

        retrieved_context = self._retrieve_context(topic)
        full_context = self._build_context(context, retrieved_context)
        draft = self._write_article(topic, full_context)
        context.draft_content = draft
        return draft

    def _retrieve_context(self, topic: str) -> str:
        if not self.vector_store:
            return ""
        try:
            docs = self.vector_store.retrieve_context(query=topic, n_results=5)
            return self.vector_store.format_context(docs)
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return ""

    def _build_context(self, context: AgentContext, retrieved: str) -> str:
        parts = []
        if context.research_notes:
            parts.append("=== Research Notes ===\n" + context.research_notes)
        if retrieved:
            parts.append("=== Additional Context ===\n" + retrieved)
        return "\n\n".join(parts) if parts else f"Topic: {context.topic}"

    def _write_article(self, topic: str, full_context: str) -> str:
        prompt = self.get_prompt(topic=topic, context=full_context)
        if not prompt:
            prompt = f"""Write a comprehensive article about "{topic}" using this research:

{full_context}

Include: introduction, clear headings, facts, insights, conclusion. 800-1200 words."""

        system_prompt = "You are an expert content writer creating engaging, accurate articles."
        try:
            return self.llm_client.generate(prompt=prompt, system_prompt=system_prompt, max_tokens=3000)
        except Exception as e:
            logger.error(f"Article generation failed: {e}")
            return f"# {topic}\n\nGeneration failed.\n\n{full_context}"

    def post_run(self, context: AgentContext, output: str) -> str:
        output = output.strip()
        if not output.startswith("#"):
            output = f"# {context.topic}\n\n{output}"
        return output
