"""Unit tests for signals.py — hiring signal scorer."""
from signals import (
    _score_deployment,
    _score_depth,
    _score_impact,
    _score_trajectory,
    _score_verbs,
    _score_soft_coverage,
    _score_skill_gap,
    compute_hiring_signals,
    detect_role_type,
    generate_prediction,
)

STRONG_RESUME = """
Senior Data Scientist with 6 years of experience.

TECHNICAL SKILLS
Python, SQL, PyTorch, scikit-learn, AWS, Docker, Airflow, Kafka, MLflow

EXPERIENCE
- Built and deployed NLP models for document classification at scale (100M+ docs/day).
- Designed end-to-end ML pipelines using Airflow and Docker on AWS.
- Led stakeholder presentations and translated business requirements into ML solutions.
- Mentored a team of 3 junior data scientists, improving team velocity by 40%.
- Reduced model inference latency by 60% through optimised batch processing.

PROJECTS
Clinical RAG System | Python, NLP, Vector Search
- Developed an end-to-end Retrieval Augmented Generation system using hybrid retrieval.
- Deployed the system as a web application serving 500+ daily users.

EDUCATION
Master of Data Science, RMIT University (2024-Present)
Bachelor of Information Technology, DDU (2020-2024)
"""

WEAK_RESUME = """
I helped with some data projects and assisted the team with Excel work.
I participated in meetings and supported the manager with reports.
I was involved in data entry and aided with basic analysis tasks.
I have a high school diploma.
"""


class TestScoreImpact:
    def test_strong_resume_has_high_impact(self):
        score, count = _score_impact(STRONG_RESUME)
        assert score > 0.3
        assert count >= 3

    def test_weak_resume_has_low_impact(self):
        score, count = _score_impact(WEAK_RESUME)
        assert score < 0.3

    def test_percentage_boosts_score(self):
        text = "Improved model accuracy by 35% and reduced latency by 50%."
        score, _ = _score_impact(text)
        assert score > 0.5

    def test_scale_numbers_boost_score(self):
        text = "Processed 10M+ records daily across 3 data centres."
        score, _ = _score_impact(text)
        assert score > 0.4

    def test_returns_tuple(self):
        result = _score_impact("Some text with 5 numbers.")
        assert isinstance(result, tuple) and len(result) == 2


class TestScoreDepth:
    def test_end_to_end_pipeline_scores_high(self):
        text = "Developed an end-to-end pipeline using hybrid retrieval and semantic embeddings."
        assert _score_depth(text) > 0.3

    def test_pipe_separated_stacks_boost_score(self):
        text = "Project Title | Python, NLP, Vector Search\nAnother | React, Node.js, PostgreSQL"
        assert _score_depth(text) > 0.2

    def test_simple_text_scores_low(self):
        text = "Worked on a project using Python."
        assert _score_depth(text) < 0.3

    def test_multiple_complexity_signals(self):
        text = "Built a scalable distributed real-time pipeline with validation and evaluation."
        assert _score_depth(text) > 0.5


class TestScoreDeployment:
    def test_deployed_app_scores_high(self):
        text = "Deployed the model within a Flask web application serving end users."
        assert _score_deployment(text) > 0.6

    def test_no_deployment_scores_zero(self):
        text = "Built a model that predicts customer churn using logistic regression."
        assert _score_deployment(text) == 0.0

    def test_production_mention_scores(self):
        text = "Shipped to production with 99.9% uptime."
        assert _score_deployment(text) > 0.3


class TestScoreVerbs:
    def test_strong_verbs_score_high(self):
        text = "Led the team. Designed the architecture. Deployed the system. Mentored engineers."
        assert _score_verbs(text) > 0.7

    def test_weak_verbs_score_low(self):
        text = "Helped the team. Assisted the manager. Participated in meetings. Was involved."
        assert _score_verbs(text) < 0.3

    def test_no_verbs_returns_neutral(self):
        assert _score_verbs("Python SQL AWS Docker Kubernetes") == 0.5

    def test_mixed_verbs_returns_ratio(self):
        text = "Developed features and helped with testing. Built APIs and assisted deployment."
        score = _score_verbs(text)
        assert 0.3 < score < 0.8


class TestScoreTrajectory:
    def test_masters_plus_current_scores_high(self):
        text = "Master of Data Science, RMIT (2024-Present). Bachelor of IT (2020-2024)."
        assert _score_trajectory(text) > 0.7

    def test_bachelor_only_scores_mid(self):
        text = "Bachelor of Computer Science, completed 2022."
        score = _score_trajectory(text)
        assert 0.3 < score < 0.7

    def test_no_degree_scores_low(self):
        assert _score_trajectory("Worked at Google for 5 years.") < 0.3

    def test_phd_scores_highest(self):
        text = "PhD in Machine Learning, Stanford University."
        assert _score_trajectory(text) >= 0.75


class TestScoreSoftCoverage:
    def test_full_coverage_scores_one(self):
        resume = "Strong communication, leadership, and problem solving skills."
        jd_skills = ["communication", "leadership", "problem solving"]
        assert _score_soft_coverage(resume, jd_skills) == 1.0

    def test_no_match_scores_zero(self):
        resume = "Python AWS Docker Kubernetes SQL"
        jd_skills = ["communication", "leadership"]
        assert _score_soft_coverage(resume, jd_skills) == 0.0

    def test_empty_jd_skills_returns_neutral(self):
        assert _score_soft_coverage("any text", []) == 0.5


class TestComputeHiringSignals:
    def test_strong_resume_returns_high_score(self):
        result = compute_hiring_signals(STRONG_RESUME, ["communication", "leadership"])
        assert result["hiring_signal_score"] > 60
        assert result["grade"] in {"Strong Signals", "Good Signals"}

    def test_weak_resume_returns_low_score(self):
        result = compute_hiring_signals(WEAK_RESUME)
        assert result["hiring_signal_score"] < 50

    def test_output_keys_present(self):
        result = compute_hiring_signals(STRONG_RESUME)
        assert "hiring_signal_score" in result
        assert "grade" in result
        assert "breakdown" in result
        assert "quantified_lines" in result

    def test_breakdown_has_seven_dimensions(self):
        result = compute_hiring_signals(STRONG_RESUME)
        assert len(result["breakdown"]) == 7

    def test_skill_gap_dimension_present(self):
        result = compute_hiring_signals(STRONG_RESUME)
        assert "Skill Gap Coverage" in result["breakdown"]

    def test_score_in_valid_range(self):
        result = compute_hiring_signals(STRONG_RESUME)
        assert 0 <= result["hiring_signal_score"] <= 100


class TestGeneratePrediction:
    def test_high_match_high_signal_is_strong_candidate(self):
        result = generate_prediction(75, 80, [])
        assert result["verdict"] == "Strong Candidate"

    def test_low_match_high_signal_is_overlooked_gem(self):
        result = generate_prediction(35, 74, ["aws", "agile"])
        assert result["verdict"] == "Overlooked Gem"
        assert "aws" in result["explanation"]

    def test_high_match_low_signal_is_verify_depth(self):
        result = generate_prediction(70, 40, [])
        assert "Verify" in result["verdict"]

    def test_low_match_low_signal_is_gaps(self):
        result = generate_prediction(25, 35, ["python", "sql", "aws"])
        assert result["verdict"] == "Significant Gaps"

    def test_all_predictions_have_required_keys(self):
        for match, signal in [(80, 80), (30, 80), (70, 30), (20, 20), (50, 55)]:
            result = generate_prediction(match, signal, [])
            assert all(k in result for k in ("verdict", "icon", "color", "explanation"))


class TestDetectRoleType:
    def test_consulting_jd_detected(self):
        jd = "Looking for a consultant to manage client stakeholder engagement and deliverables."
        assert detect_role_type(jd) == "consulting"

    def test_data_engineering_jd_detected(self):
        jd = "Build and maintain data pipelines using Airflow, Kafka, and Spark on Snowflake."
        assert detect_role_type(jd) == "data_engineering"

    def test_data_analyst_jd_detected(self):
        jd = "Create dashboards and reporting insights using Power BI and Tableau."
        assert detect_role_type(jd) == "data_analyst"

    def test_research_ml_jd_detected(self):
        jd = "Conduct research and publish papers. PhD preferred. Run benchmarks and experiments."
        assert detect_role_type(jd) == "research_ml"

    def test_software_engineering_jd_detected(self):
        jd = "Design backend API microservices and manage Kubernetes infrastructure and CI/CD."
        assert detect_role_type(jd) == "software_engineering"

    def test_generic_jd_falls_back_to_general(self):
        jd = "We are hiring a motivated professional to join our growing team."
        assert detect_role_type(jd) == "general"

    def test_role_type_returned_in_compute_hiring_signals(self):
        result = compute_hiring_signals(STRONG_RESUME, jd_text="Client stakeholder consultant advisory.")
        assert result["role_type"] == "consulting"

    def test_no_jd_defaults_to_general(self):
        result = compute_hiring_signals(STRONG_RESUME)
        assert result["role_type"] == "general"


class TestScoreSkillGap:
    def test_no_missing_skills_scores_one(self):
        assert _score_skill_gap(["python", "sql", "aws"], []) == 1.0

    def test_all_missing_scores_zero(self):
        assert _score_skill_gap(["python", "sql", "aws"], ["python", "sql", "aws"]) == 0.0

    def test_half_missing_scores_half(self):
        assert _score_skill_gap(["a", "b", "c", "d"], ["a", "b"]) == 0.5

    def test_no_jd_skills_returns_neutral(self):
        assert _score_skill_gap([], ["python"]) == 1.0

    def test_none_missing_returns_neutral(self):
        assert _score_skill_gap(["python", "sql"], None) == 1.0

    def test_skill_gap_lowers_signal_score(self):
        all_present = compute_hiring_signals(
            STRONG_RESUME, jd_skills=["python", "sql"], missing_skills=[]
        )
        half_missing = compute_hiring_signals(
            STRONG_RESUME, jd_skills=["python", "sql", "aws", "kafka"], missing_skills=["aws", "kafka"]
        )
        assert all_present["hiring_signal_score"] >= half_missing["hiring_signal_score"]

    def test_required_missing_skill_penalises_more_than_optional(self):
        # Missing a "required" skill (weight=1.0) should penalise more
        # than missing a "nice to have" skill (weight=0.5).
        high_weight_missing = compute_hiring_signals(
            STRONG_RESUME,
            jd_skills=["python", "sql"],
            missing_skills=["python"],
            skill_weights={"python": 1.0, "sql": 0.5},
        )
        low_weight_missing = compute_hiring_signals(
            STRONG_RESUME,
            jd_skills=["python", "sql"],
            missing_skills=["sql"],
            skill_weights={"python": 1.0, "sql": 0.5},
        )
        assert high_weight_missing["hiring_signal_score"] < low_weight_missing["hiring_signal_score"]
