"""Unit tests for scorer.py — pure logic tests, no model calls."""
import pytest

from scorer import _detect_resume_sections, _extract_jd_keywords, _get_skills_section, _tfidf_similarity


class TestExtractJdKeywords:
    def test_returns_list(self):
        kws = _extract_jd_keywords("We need Python, SQL, and AWS experience.")
        assert isinstance(kws, list)

    def test_captures_single_keywords(self):
        kws = _extract_jd_keywords("Proficiency in Python and SQL required. AWS knowledge a plus.")
        kws_lower = [k.lower() for k in kws]
        assert any("python" in k for k in kws_lower)

    def test_captures_bigrams(self):
        jd = "Strong stakeholder management and problem solving skills required."
        kws = _extract_jd_keywords(jd)
        kws_lower = [k.lower() for k in kws]
        assert any("stakeholder management" in k or "problem solving" in k for k in kws_lower)

    def test_filters_boilerplate(self):
        jd = "Required experience with SQL. We are seeking a senior engineer with degree."
        kws = _extract_jd_keywords(jd)
        kws_lower = [k.lower() for k in kws]
        # "required", "seeking", "senior", "engineer", "degree" are boilerplate
        assert "required" not in kws_lower
        assert "seeking" not in kws_lower

    def test_empty_text_returns_empty(self):
        assert _extract_jd_keywords("") == []


class TestDetectResumeSections:
    RESUME_WITH_SECTIONS = (
        "TECHNICAL SKILLS\nPython, JavaScript, React, PostgreSQL\n\n"
        "EXPERIENCE\nSoftware Engineer at Acme Corp 2020-2024\n\n"
        "EDUCATION\nBSc Computer Science, University of Melbourne"
    )
    RESUME_BLOB = "Python developer with 5 years experience at Google building ML systems."

    def test_detects_skills_section(self):
        sections = _detect_resume_sections(self.RESUME_WITH_SECTIONS)
        skill_key = next((k for k in sections if "skill" in k), None)
        assert skill_key is not None

    def test_skills_section_contains_tools(self):
        sections = _detect_resume_sections(self.RESUME_WITH_SECTIONS)
        skills_text = _get_skills_section(sections)
        assert skills_text is not None
        assert "python" in skills_text.lower()

    def test_detects_experience_section(self):
        sections = _detect_resume_sections(self.RESUME_WITH_SECTIONS)
        assert any("experience" in k for k in sections)

    def test_returns_empty_for_blob(self):
        sections = _detect_resume_sections(self.RESUME_BLOB)
        assert sections == {}

    def test_get_skills_section_returns_none_when_absent(self):
        sections = {"experience": "5 years at Google", "education": "BSc CS"}
        assert _get_skills_section(sections) is None

    def test_section_content_stripped(self):
        sections = _detect_resume_sections(self.RESUME_WITH_SECTIONS)
        for content in sections.values():
            assert content == content.strip()


class TestTfidfSimilarity:
    def test_identical_texts_score_one(self):
        text = "Python developer with AWS and Docker experience"
        score = _tfidf_similarity(text, text)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_unrelated_texts_score_low(self):
        jd = "Senior Python engineer with machine learning and AWS"
        candidate = "Marketing manager with Excel and PowerPoint skills"
        score = _tfidf_similarity(jd, candidate)
        assert score < 0.3

    def test_similar_texts_score_higher_than_unrelated(self):
        # Use texts with heavy keyword overlap — scores are corpus-relative so
        # we assert similarity > unrelated rather than a fixed threshold.
        jd = "Python Flask REST API PostgreSQL backend engineer"
        candidate = "Python Flask REST API PostgreSQL backend developer"
        related_score = _tfidf_similarity(jd, candidate)

        unrelated_jd = "Senior Python engineer machine learning AWS"
        unrelated_candidate = "Marketing manager Excel PowerPoint skills"
        unrelated_score = _tfidf_similarity(unrelated_jd, unrelated_candidate)

        assert related_score > unrelated_score

    def test_returns_float_in_range(self):
        score = _tfidf_similarity("some text", "other text")
        assert 0.0 <= score <= 1.0
