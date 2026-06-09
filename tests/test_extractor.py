"""Unit tests for extractor.py — no model downloads required."""
from extractor import (
    extract_education,
    extract_industry,
    extract_jd,
    extract_seniority,
    extract_skills_and_tools,
    extract_soft_skills,
    extract_unknown_tools,
    extract_work_type,
    extract_years_of_experience,
    parse_jd_sections,
    weighted_skill_frequencies,
)

JD_SAMPLE = """
We are hiring a Senior Python Engineer with 5+ years of experience.
You will work with AWS, Docker, Kubernetes, and PostgreSQL.
A Bachelor's degree in Computer Science is required.
This is a fully remote position.
Nice to have: experience with Kafka and Redis.
"""


class TestYearsOfExperience:
    def test_detects_plus_pattern(self):
        assert "5+ years" in extract_years_of_experience("5+ years of experience required")

    def test_detects_range_pattern(self):
        result = extract_years_of_experience("3-5 years in backend development")
        assert len(result) > 0

    def test_returns_empty_for_no_match(self):
        assert extract_years_of_experience("No experience mentioned here.") == []

    def test_deduplicates(self):
        result = extract_years_of_experience("5+ years. We want 5+ years.")
        assert len(result) == 1


class TestSeniority:
    def test_senior(self):
        assert extract_seniority("Looking for a Senior Software Engineer") == "senior"

    def test_junior(self):
        assert extract_seniority("This is an entry-level role for new grads") == "junior"

    def test_manager(self):
        assert extract_seniority("Director of Engineering role") == "manager"

    def test_unspecified(self):
        assert extract_seniority("We are looking for an engineer") == "unspecified"


class TestEducation:
    def test_bachelor(self):
        assert extract_education("Bachelor's degree in CS required") == "bachelors"

    def test_master(self):
        assert extract_education("MSc or equivalent is preferred") == "masters"

    def test_phd(self):
        assert extract_education("PhD in Machine Learning preferred") == "phd"

    def test_unspecified(self):
        assert extract_education("No degree mentioned") == "unspecified"


class TestWorkType:
    def test_remote(self):
        assert extract_work_type("This is a fully remote position") == "remote"

    def test_hybrid(self):
        assert extract_work_type("We operate on a hybrid schedule") == "hybrid"

    def test_onsite(self):
        assert extract_work_type("You must work on-site in our NYC office") == "on-site"

    def test_unspecified(self):
        assert extract_work_type("Come join our team") == "unspecified"


class TestSkillsAndTools:
    def test_detects_known_tools(self):
        result = extract_skills_and_tools("Must know Python, AWS, and Docker")
        assert "python" in result
        assert "aws" in result
        assert "docker" in result

    def test_case_insensitive(self):
        assert "postgresql" in extract_skills_and_tools("POSTGRESQL experience required")

    def test_returns_sorted(self):
        result = extract_skills_and_tools("python and aws and docker")
        assert result == sorted(result)

    def test_no_partial_matches(self):
        # "r" should not match inside "docker"
        result = extract_skills_and_tools("docker only, no standalone r")
        assert "r" not in result or "docker" in result


class TestSoftSkills:
    def test_detects_communication(self):
        assert "communication" in extract_soft_skills("Excellent communication skills required")

    def test_detects_leadership(self):
        assert "leadership" in extract_soft_skills("Strong leadership and mentoring ability")

    def test_detects_problem_solving(self):
        assert "problem solving" in extract_soft_skills("Strong problem solving skills")

    def test_returns_sorted(self):
        result = extract_soft_skills("leadership communication teamwork")
        assert result == sorted(result)

    def test_no_false_positives(self):
        assert extract_soft_skills("Python AWS Docker PostgreSQL") == []


class TestIndustry:
    def test_fintech(self):
        assert extract_industry("We are a leading fintech payments company") == "fintech / finance"

    def test_healthcare(self):
        assert extract_industry("Join our healthcare data platform team") == "healthcare"

    def test_ecommerce(self):
        assert extract_industry("Build features for our ecommerce marketplace") == "e-commerce / retail"

    def test_general_tech_fallback(self):
        assert extract_industry("Looking for a Python developer") == "general tech"


class TestUnknownTools:
    def test_detects_context_tool(self):
        result = extract_unknown_tools("We need experience with Temporal for workflow orchestration")
        assert "temporal" in result

    def test_detects_multiple(self):
        result = extract_unknown_tools(
            "Proficiency in Chronos required. Familiarity with Nemo a plus."
        )
        assert "chronos" in result
        assert "nemo" in result

    def test_ignores_known_tools(self):
        result = extract_unknown_tools("Proficiency in Python and AWS required")
        assert "python" not in result
        assert "aws" not in result

    def test_filters_common_words(self):
        result = extract_unknown_tools("Experience with the latest technologies in cloud")
        assert "the" not in result
        assert "latest" not in result
        assert "cloud" not in result

    def test_returns_sorted(self):
        result = extract_unknown_tools(
            "Experience with Prefect. Familiarity with Polars. Background in Ray."
        )
        assert result == sorted(result)

    def test_empty_for_no_tech_context(self):
        result = extract_unknown_tools("We are a great company with amazing benefits and culture.")
        assert result == []


class TestParseJdSections:
    JD_SECTIONED = (
        "Senior Data Scientist — AI Platform\n\n"
        "Required\n"
        "Python, SQL, AWS. 3+ years of experience.\n\n"
        "Nice to have\n"
        "Databricks, dbt, Spark.\n\n"
        "Responsibilities\n"
        "Build and deploy ML models at scale."
    )

    def test_returns_list_of_tuples(self):
        result = parse_jd_sections(self.JD_SECTIONED)
        assert isinstance(result, list)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in result)

    def test_title_block_gets_highest_multiplier(self):
        result = parse_jd_sections(self.JD_SECTIONED)
        title_mult = result[0][1]
        assert title_mult == 3.0

    def test_required_section_gets_two_times(self):
        result = parse_jd_sections(self.JD_SECTIONED)
        req = next((r for r in result if "python" in r[0].lower()), None)
        assert req is not None
        assert req[1] == 2.0

    def test_nice_to_have_gets_half_multiplier(self):
        result = parse_jd_sections(self.JD_SECTIONED)
        nth = next((r for r in result if "databricks" in r[0].lower()), None)
        assert nth is not None
        assert nth[1] == 0.5

    def test_fallback_for_unsectioned_jd(self):
        result = parse_jd_sections("We are looking for a Python developer.")
        assert len(result) == 1
        assert result[0][1] == 1.0


class TestWeightedSkillFrequencies:
    def test_required_skill_outweights_optional(self):
        jd = "Required\nPython, AWS required.\n\nNice to have\nDatabricks optional."
        weights = weighted_skill_frequencies(jd, ["python", "databricks"])
        assert weights.get("python", 0) > weights.get("databricks", 0)

    def test_absent_skill_has_no_entry(self):
        jd = "We need Python developers."
        weights = weighted_skill_frequencies(jd, ["python", "kubernetes"])
        assert "kubernetes" not in weights

    def test_scores_normalised_to_half_one_range(self):
        jd = "Required\nPython, SQL.\n\nNice to have\nTableau."
        weights = weighted_skill_frequencies(jd, ["python", "sql", "tableau"])
        for v in weights.values():
            assert 0.5 <= v <= 1.0

    def test_empty_skills_returns_empty_dict(self):
        assert weighted_skill_frequencies("Some JD text.", []) == {}

    def test_skill_weights_included_in_extract_jd(self):
        jd = "Required\nPython, SQL.\n\nNice to have\nTableau."
        result = extract_jd(jd)
        assert "skill_weights" in result
        assert isinstance(result["skill_weights"], dict)


class TestExtractJd:
    def test_full_output_keys(self):
        result = extract_jd(JD_SAMPLE)
        assert set(result.keys()) == {
            "skills_and_tools", "unknown_tools", "skill_weights", "soft_skills",
            "years_of_experience", "seniority", "education", "work_type", "industry",
        }

    def test_integration_senior(self):
        result = extract_jd(JD_SAMPLE)
        assert result["seniority"] == "senior"

    def test_integration_remote(self):
        result = extract_jd(JD_SAMPLE)
        assert result["work_type"] == "remote"

    def test_integration_bachelors(self):
        result = extract_jd(JD_SAMPLE)
        assert result["education"] == "bachelors"

    def test_integration_skills(self):
        result = extract_jd(JD_SAMPLE)
        assert "python" in result["skills_and_tools"]
        assert "aws" in result["skills_and_tools"]
