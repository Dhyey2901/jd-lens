import re
import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm")

SENIORITY_SIGNALS = {
    "junior": ["junior", "entry level", "entry-level", "graduate", "0-2 years", "1+ year"],
    "mid": ["mid", "mid-level", "mid level", "2-4 years", "3+ years", "2+ years"],
    "senior": ["senior", "sr.", "lead", "principal", "staff", "5+ years", "7+ years", "10+ years"],
    "manager": ["manager", "director", "head of", "vp", "vice president"],
}

KNOWN_TOOLS = [
    "python", "java", "javascript", "typescript", "go", "rust", "c++", "c#", "ruby", "swift",
    "react", "angular", "vue", "next.js", "node.js", "django", "flask", "fastapi", "spring",
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "sqlite",
    "aws", "gcp", "azure", "docker", "kubernetes", "terraform", "ansible",
    "git", "github", "gitlab", "jenkins", "ci/cd", "github actions",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
    "kafka", "rabbitmq", "celery", "graphql", "rest", "grpc",
    "linux", "bash", "sql", "html", "css", "tailwind",
]


def extract_years_of_experience(text: str) -> list[str]:
    pattern = re.compile(
        r"(\d+\+?\s*(?:to|-)\s*\d+\s*years?|\d+\+\s*years?|\d+\s*years?\s*of\s*experience)",
        re.IGNORECASE,
    )
    return list({m.group().strip() for m in pattern.finditer(text)})


def extract_seniority(text: str) -> str:
    lower = text.lower()
    for level, signals in SENIORITY_SIGNALS.items():
        if any(s in lower for s in signals):
            return level
    return "unspecified"


def extract_skills_and_tools(text: str) -> list[str]:
    lower = text.lower()
    found = [tool for tool in KNOWN_TOOLS if re.search(r"\b" + re.escape(tool) + r"\b", lower)]

    doc = nlp(text)
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(t) for t in KNOWN_TOOLS]
    matcher.add("TOOLS", patterns)

    entity_skills = set()
    for ent in doc.ents:
        if ent.label_ in ("ORG", "PRODUCT"):
            entity_skills.add(ent.text.lower())

    combined = sorted(set(found) | entity_skills)
    return combined


def extract_jd(jd_text: str) -> dict:
    return {
        "skills_and_tools": extract_skills_and_tools(jd_text),
        "years_of_experience": extract_years_of_experience(jd_text),
        "seniority": extract_seniority(jd_text),
    }
