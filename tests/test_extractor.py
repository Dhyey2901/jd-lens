"""Unit tests for extractor.py — no model downloads required."""
from extractor import (
    extract_education,
    extract_industry,
    extract_jd,
    extract_seniority,
    extract_skills_and_tools,
    extract_soft_skills,
    extract_work_type,
    extract_years_of_experience,
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


class TestExtractJd:
    def test_full_output_keys(self):
        result = extract_jd(JD_SAMPLE)
        assert set(result.keys()) == {
            "skills_and_tools", "soft_skills", "years_of_experience",
            "seniority", "education", "work_type", "industry",
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
