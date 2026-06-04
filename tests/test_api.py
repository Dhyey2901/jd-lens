"""Integration tests for the FastAPI /analyse endpoint."""
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)

_JD = (
    "We are looking for a Senior Python Engineer with 5+ years of experience. "
    "Required: Python, AWS, Docker, PostgreSQL, Kubernetes. "
    "Bachelor degree required. Fully remote position."
)
_CANDIDATE = (
    "5 years of Python development. Strong AWS and Docker experience. "
    "PostgreSQL databases daily. Bachelor of Computer Science. "
    "Built distributed systems on Kubernetes."
)


class TestHealthEndpoint:
    def test_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_response_shape(self):
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert "version" in data


class TestAnalyseEndpoint:
    def test_returns_200_for_valid_input(self):
        r = client.post("/analyse", json={"jd_text": _JD, "candidate_text": _CANDIDATE})
        assert r.status_code == 200

    def test_response_has_required_keys(self):
        data = client.post("/analyse", json={"jd_text": _JD, "candidate_text": _CANDIDATE}).json()
        assert "fit_score" in data
        assert "grade" in data
        assert "score_breakdown" in data
        assert "jd_signals" in data
        assert "skill_analysis" in data
        assert "keyword_analysis" in data

    def test_fit_score_in_range(self):
        data = client.post("/analyse", json={"jd_text": _JD, "candidate_text": _CANDIDATE}).json()
        assert 0.0 <= data["fit_score"] <= 100.0

    def test_grade_is_valid_string(self):
        data = client.post("/analyse", json={"jd_text": _JD, "candidate_text": _CANDIDATE}).json()
        assert data["grade"] in ("Strong Fit", "Partial Fit", "Weak Fit")

    def test_skill_analysis_has_semantic_field(self):
        data = client.post("/analyse", json={"jd_text": _JD, "candidate_text": _CANDIDATE}).json()
        assert "semantic" in data["skill_analysis"]

    def test_jd_signals_has_industry(self):
        data = client.post("/analyse", json={"jd_text": _JD, "candidate_text": _CANDIDATE}).json()
        assert "industry" in data["jd_signals"]

    def test_rejects_jd_too_short(self):
        r = client.post("/analyse", json={"jd_text": "short", "candidate_text": _CANDIDATE})
        assert r.status_code == 422

    def test_rejects_candidate_too_short(self):
        r = client.post("/analyse", json={"jd_text": _JD, "candidate_text": "hi"})
        assert r.status_code == 422

    def test_rejects_jd_too_long(self):
        r = client.post("/analyse", json={"jd_text": "word " * 3000, "candidate_text": _CANDIDATE})
        assert r.status_code == 422

    def test_high_overlap_scores_higher_than_low_overlap(self):
        good = client.post("/analyse", json={"jd_text": _JD, "candidate_text": _CANDIDATE}).json()
        bad = client.post("/analyse", json={
            "jd_text": _JD,
            "candidate_text": "Marketing professional with Excel and PowerPoint experience.",
        }).json()
        assert good["fit_score"] > bad["fit_score"]
