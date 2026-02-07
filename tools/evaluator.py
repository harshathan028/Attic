"""
Content Evaluator - Scoring and analysis for generated content.

This module provides content quality metrics including readability,
keyword density, length scoring, and overall quality assessment.
Uses simple implementations that don't require NLTK.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class ContentScore:
    """Comprehensive content quality score."""
    overall_score: float  # 0-100
    readability_score: float  # 0-100
    length_score: float  # 0-100
    keyword_density_score: float  # 0-100
    structure_score: float  # 0-100
    details: Dict[str, Any]


class ContentEvaluator:
    """
    Content quality evaluator with multiple metrics.
    
    Analyzes text for readability, structure, keyword usage,
    and other quality factors relevant to content optimization.
    Uses simple implementations without external NLP dependencies.
    """

    def __init__(
        self,
        target_word_count: int = 1000,
        target_keyword_density: float = 0.02,  # 2%
    ):
        """
        Initialize the content evaluator.

        Args:
            target_word_count: Ideal word count for content.
            target_keyword_density: Target keyword density (0-1).
        """
        self.target_word_count = target_word_count
        self.target_keyword_density = target_keyword_density
        logger.info("Initialized ContentEvaluator")

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word using simple heuristics."""
        word = word.lower().strip()
        if not word:
            return 0
        
        vowels = "aeiouy"
        count = 0
        prev_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        
        # Handle silent 'e' at end
        if word.endswith('e') and count > 1:
            count -= 1
        
        # Handle special endings
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            count += 1
            
        return max(1, count)

    def _sentence_count(self, text: str) -> int:
        """Count sentences in text."""
        sentences = re.split(r'[.!?]+', text)
        return len([s for s in sentences if s.strip()])

    def _flesch_reading_ease(self, content: str) -> float:
        """
        Calculate Flesch Reading Ease score.
        
        Formula: 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
        """
        sentences = re.split(r'[.!?]+', content)
        sentences = [s for s in sentences if s.strip()]
        sentence_count = max(1, len(sentences))
        
        words = re.findall(r'\b[a-zA-Z]+\b', content)
        word_count = max(1, len(words))
        
        syllable_count = sum(self._count_syllables(w) for w in words)
        
        flesch = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count)
        return flesch

    def _flesch_kincaid_grade(self, content: str) -> float:
        """
        Calculate Flesch-Kincaid Grade Level.
        
        Formula: 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
        """
        sentences = re.split(r'[.!?]+', content)
        sentences = [s for s in sentences if s.strip()]
        sentence_count = max(1, len(sentences))
        
        words = re.findall(r'\b[a-zA-Z]+\b', content)
        word_count = max(1, len(words))
        
        syllable_count = sum(self._count_syllables(w) for w in words)
        
        grade = 0.39 * (word_count / sentence_count) + 11.8 * (syllable_count / word_count) - 15.59
        return grade

    def _gunning_fog_index(self, content: str) -> float:
        """
        Calculate Gunning Fog Index.
        
        Formula: 0.4 * ((words/sentences) + 100 * (complex_words/words))
        Complex words = words with 3+ syllables
        """
        sentences = re.split(r'[.!?]+', content)
        sentences = [s for s in sentences if s.strip()]
        sentence_count = max(1, len(sentences))
        
        words = re.findall(r'\b[a-zA-Z]+\b', content)
        word_count = max(1, len(words))
        
        complex_words = sum(1 for w in words if self._count_syllables(w) >= 3)
        
        fog = 0.4 * ((word_count / sentence_count) + 100 * (complex_words / word_count))
        return fog

    def evaluate(
        self,
        content: str,
        keywords: Optional[List[str]] = None,
    ) -> ContentScore:
        """
        Evaluate content quality across multiple dimensions.

        Args:
            content: The text content to evaluate.
            keywords: Optional list of target keywords.

        Returns:
            ContentScore with detailed metrics.
        """
        if not content or not content.strip():
            return ContentScore(
                overall_score=0,
                readability_score=0,
                length_score=0,
                keyword_density_score=0,
                structure_score=0,
                details={"error": "Empty content"},
            )

        readability = self._evaluate_readability(content)
        length = self._evaluate_length(content)
        keyword_density = self._evaluate_keywords(content, keywords or [])
        structure = self._evaluate_structure(content)

        # Weighted average for overall score
        overall = (
            readability * 0.30 +
            length * 0.20 +
            keyword_density * 0.20 +
            structure * 0.30
        )

        details = {
            "word_count": len(content.split()),
            "sentence_count": self._sentence_count(content),
            "paragraph_count": len([p for p in content.split("\n\n") if p.strip()]),
            "flesch_reading_ease": round(self._flesch_reading_ease(content), 2),
            "flesch_kincaid_grade": round(self._flesch_kincaid_grade(content), 2),
            "gunning_fog": round(self._gunning_fog_index(content), 2),
            "keywords_found": self._count_keywords(content, keywords or []),
        }

        score = ContentScore(
            overall_score=round(overall, 2),
            readability_score=round(readability, 2),
            length_score=round(length, 2),
            keyword_density_score=round(keyword_density, 2),
            structure_score=round(structure, 2),
            details=details,
        )

        logger.info(f"Content evaluation: overall={score.overall_score}")
        return score

    def _evaluate_readability(self, content: str) -> float:
        """
        Evaluate content readability.

        Uses Flesch Reading Ease approximation, targeting 60-70 range
        for general audience accessibility.
        """
        try:
            flesch = self._flesch_reading_ease(content)
            
            # Optimal range is 60-70 (plain English)
            if 60 <= flesch <= 70:
                return 100
            elif 50 <= flesch < 60 or 70 < flesch <= 80:
                return 85
            elif 40 <= flesch < 50 or 80 < flesch <= 90:
                return 70
            elif 30 <= flesch < 40 or 90 < flesch <= 100:
                return 55
            else:
                return max(30, 100 - abs(flesch - 65))

        except Exception as e:
            logger.warning(f"Readability calculation failed: {e}")
            return 50

    def _evaluate_length(self, content: str) -> float:
        """
        Evaluate content length relative to target.

        Penalizes both too short and too long content,
        with optimal score at target word count.
        """
        word_count = len(content.split())
        
        if word_count == 0:
            return 0

        # Calculate ratio to target
        ratio = word_count / self.target_word_count

        if 0.8 <= ratio <= 1.2:  # Within 20% of target
            return 100
        elif 0.5 <= ratio < 0.8 or 1.2 < ratio <= 1.5:
            return 75
        elif 0.3 <= ratio < 0.5 or 1.5 < ratio <= 2.0:
            return 50
        else:
            return 25

    def _evaluate_keywords(self, content: str, keywords: List[str]) -> float:
        """
        Evaluate keyword usage and density.

        Checks for presence and appropriate density of target keywords.
        """
        if not keywords:
            return 75  # Neutral score if no keywords specified

        content_lower = content.lower()
        word_count = len(content.split())
        
        if word_count == 0:
            return 0

        keyword_counts = self._count_keywords(content, keywords)
        total_keyword_occurrences = sum(keyword_counts.values())
        
        # Calculate keyword density
        density = total_keyword_occurrences / word_count

        # Check how many keywords were found
        keywords_found_ratio = len([k for k, v in keyword_counts.items() if v > 0]) / len(keywords)

        # Score based on density proximity to target and keyword coverage
        density_score = 100 - min(100, abs(density - self.target_keyword_density) * 2500)
        coverage_score = keywords_found_ratio * 100

        return (density_score * 0.5 + coverage_score * 0.5)

    def _count_keywords(self, content: str, keywords: List[str]) -> Dict[str, int]:
        """Count occurrences of each keyword in content."""
        content_lower = content.lower()
        counts = {}
        for keyword in keywords:
            pattern = rf"\b{re.escape(keyword.lower())}\b"
            counts[keyword] = len(re.findall(pattern, content_lower))
        return counts

    def _evaluate_structure(self, content: str) -> float:
        """
        Evaluate content structure.

        Checks for proper use of headings, paragraphs,
        and other structural elements.
        """
        score = 50  # Base score

        # Check for headings (markdown or plain)
        heading_patterns = [
            r"^#+\s+.+$",  # Markdown headings
            r"^[A-Z][^.!?]*:$",  # Title case lines ending with colon
        ]
        has_headings = any(
            re.search(pattern, content, re.MULTILINE)
            for pattern in heading_patterns
        )
        if has_headings:
            score += 15

        # Check for paragraphs (content should have multiple paragraphs)
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        if len(paragraphs) >= 3:
            score += 15
        elif len(paragraphs) >= 2:
            score += 10

        # Check for lists
        list_patterns = [
            r"^[-*•]\s+.+$",  # Unordered lists
            r"^\d+[.)]\s+.+$",  # Ordered lists
        ]
        has_lists = any(
            re.search(pattern, content, re.MULTILINE)
            for pattern in list_patterns
        )
        if has_lists:
            score += 10

        # Check for reasonable sentence variety
        sentences = re.split(r"[.!?]+", content)
        if sentences:
            lengths = [len(s.split()) for s in sentences if s.strip()]
            if lengths:
                avg_length = sum(lengths) / len(lengths)
                # Optimal sentence length is 15-20 words
                if 10 <= avg_length <= 25:
                    score += 10

        return min(100, score)

    def get_improvement_suggestions(self, score: ContentScore) -> List[str]:
        """
        Generate improvement suggestions based on score.

        Args:
            score: The content score to analyze.

        Returns:
            List of actionable improvement suggestions.
        """
        suggestions = []

        if score.readability_score < 70:
            suggestions.append(
                "Improve readability: Use shorter sentences and simpler words. "
                f"Current Flesch score: {score.details.get('flesch_reading_ease', 'N/A')}"
            )

        if score.length_score < 70:
            word_count = score.details.get("word_count", 0)
            if word_count < self.target_word_count * 0.8:
                suggestions.append(
                    f"Content is too short ({word_count} words). "
                    f"Target: {self.target_word_count} words."
                )
            else:
                suggestions.append(
                    f"Content may be too long ({word_count} words). "
                    "Consider condensing for better engagement."
                )

        if score.keyword_density_score < 70:
            suggestions.append(
                "Optimize keyword usage: Ensure target keywords appear "
                "naturally throughout the content (aim for 1-2% density)."
            )

        if score.structure_score < 70:
            suggestions.append(
                "Improve structure: Add clear headings, use bullet points "
                "for lists, and ensure proper paragraph breaks."
            )

        if not suggestions:
            suggestions.append("Content quality is good. Minor refinements may still help.")

        return suggestions
