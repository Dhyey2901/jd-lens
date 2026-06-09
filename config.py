"""Central configuration — models, labels, weights, and vocabulary lists."""
from __future__ import annotations

# ── Model identifiers ──────────────────────────────────────────────────────────
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Cross-encoder for pairwise JD/resume relevance scoring (~22 MB).
# Used when required JD sentences are available; bi-encoder is the fallback.
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Zero-shot classification ───────────────────────────────────────────────────
CANDIDATE_LABELS: list[str] = [
    "required skill",
    "nice to have",
    "responsibility",
    "company info",
]

LABEL_DISPLAY: dict[str, str] = {
    "required skill": "Required",
    "nice to have": "Nice-to-Have",
    "responsibility": "Responsibility",
    "company info": "Company Info",
}

# ── Fit score weights (must sum to 1.0) ───────────────────────────────────────
# Skill match gets highest weight — explicit tool/keyword coverage is the most
# reliable signal and is least affected by resume vs JD format/tense differences.
SCORE_WEIGHTS: dict[str, float] = {
    "semantic": 0.30,   # sentence-transformer cosine similarity
    "tfidf": 0.30,      # TF-IDF bigram + CountVectorizer keyword overlap
    "skill": 0.40,      # explicit tool/skill hit-rate (exact + semantic)
}

# ── Skill alias normalisation ──────────────────────────────────────────────────
# Maps common abbreviations / alternate spellings → canonical KNOWN_TOOLS name.
# Applied to candidate text before skill matching so "sklearn" hits "scikit-learn".
SKILL_ALIASES: dict[str, str] = {
    # ML / Data Science
    "sklearn": "scikit-learn",
    "sk-learn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "xgb": "xgboost",
    "lgbm": "lightgbm",
    # Databases
    "postgres": "postgresql",
    "psql": "postgresql",
    "mongo": "mongodb",
    "dynamo": "dynamodb",
    "dynamo db": "dynamodb",
    "big query": "bigquery",
    # Cloud
    "google cloud": "gcp",
    "google cloud platform": "gcp",
    "amazon web services": "aws",
    "microsoft azure": "azure",
    "azure cloud": "azure",
    # Languages / runtimes
    "js": "javascript",
    "ts": "typescript",
    "nodejs": "node.js",
    "node js": "node.js",
    # Frontend frameworks
    "reactjs": "react",
    "react.js": "react",
    "vuejs": "vue",
    "vue.js": "vue",
    "nextjs": "next.js",
    "next js": "next.js",
    # DevOps / CI
    "gh actions": "github actions",
    "cicd": "ci/cd",
    "ci cd": "ci/cd",
    # AI / LLM tooling
    "hugging face": "huggingface",
    "open ai": "openai",
    "lang chain": "langchain",
}

# ── Extraction vocabulary ──────────────────────────────────────────────────────
SENIORITY_SIGNALS: dict[str, list[str]] = {
    "junior": [
        "junior", "entry level", "entry-level", "associate", "graduate",
        "new grad", "0-2 years", "1+ year",
    ],
    "mid": [
        "mid", "mid-level", "mid level", "intermediate",
        "2-4 years", "2+ years", "3+ years",
    ],
    "senior": [
        "senior", "sr.", "lead", "principal", "staff",
        "5+ years", "7+ years", "10+ years",
    ],
    "manager": [
        "manager", "director", "head of", "vp", "vice president", "cto", "ceo",
    ],
}

EDUCATION_SIGNALS: dict[str, list[str]] = {
    "phd": ["phd", "ph.d", "doctorate", "doctoral"],
    "masters": ["master", "msc", "m.s.", "m.eng", "postgraduate"],
    "bachelors": ["bachelor", "bsc", "b.s.", "b.eng", "undergraduate", "degree in"],
}

WORK_TYPE_SIGNALS: dict[str, list[str]] = {
    "remote": ["remote", "work from home", "wfh", "fully remote", "distributed"],
    "hybrid": ["hybrid", "flexible", "part remote"],
    "on-site": ["on-site", "onsite", "in-office", "in office"],
}

SOFT_SKILLS: list[str] = [
    # Communication
    "communication", "stakeholder management", "presentation skills",
    "written communication", "verbal communication", "public speaking",
    "report writing", "documentation", "storytelling", "active listening",
    # Leadership & Management
    "leadership", "mentoring", "coaching", "people management", "team management",
    "decision making", "strategic thinking", "change management", "conflict resolution",
    "performance management", "talent development", "delegation",
    # Analytical & Problem Solving
    "problem solving", "analytical thinking", "critical thinking",
    "attention to detail", "data-driven", "research", "root cause analysis",
    "creative thinking", "innovation", "continuous improvement",
    # Collaboration & Interpersonal
    "collaboration", "teamwork", "cross-functional", "interpersonal skills",
    "relationship management", "relationship building", "client management",
    "client relationship", "customer service", "stakeholder engagement",
    "empathy", "emotional intelligence", "cultural awareness",
    # Project & Delivery
    "project management", "planning", "prioritization", "time management",
    "organisation", "organized", "multitasking", "deadline management",
    "budget management", "risk management", "quality assurance",
    # Adaptability & Drive
    "adaptability", "self-starter", "initiative", "ownership", "fast-paced",
    "resilience", "accountability", "results-driven", "proactive",
    "work ethic", "self-motivated", "detail-oriented",
    # Influence & Commercial
    "influencing", "negotiation", "persuasion", "stakeholder",
    "business acumen", "commercial awareness", "sales", "customer focus",
    # Domain-neutral professional skills
    "compliance", "facilitation", "training", "onboarding", "process improvement",
    "policy development", "reporting", "coordination", "scheduling",
]

INDUSTRY_SIGNALS: dict[str, list[str]] = {
    "fintech / finance": [
        "banking", "fintech", "financial services", "trading", "investment",
        "insurance", "payments", "lending", "wealth management", "capital markets",
    ],
    "healthcare": [
        "healthcare", "medical", "clinical", "pharma", "biotech",
        "health tech", "ehr", "fhir", "patient data",
    ],
    "e-commerce / retail": [
        "e-commerce", "ecommerce", "retail", "marketplace", "consumer", "d2c", "omnichannel",
    ],
    "consulting": [
        "consulting", "advisory", "management consulting", "professional services", "client-facing",
    ],
    "data / AI": [
        "machine learning platform", "data science", "analytics platform",
        "artificial intelligence", "ai/ml", "mlops", "data platform",
    ],
    "saas / tech": [
        "saas", "enterprise software", "cloud platform", "startup", "scale-up", "b2b saas",
    ],
    "government / public sector": [
        "government", "public sector", "federal", "defence", "defense", "civic tech",
    ],
    "education": ["edtech", "higher education", "academic research", "university"],
}

KNOWN_TOOLS: list[str] = [
    # Languages
    "python", "java", "javascript", "typescript", "go", "golang", "rust",
    "c++", "c#", "ruby", "swift", "kotlin", "scala", "r", "matlab",
    "perl", "php", "elixir", "haskell", "clojure",
    # Frontend
    "react", "angular", "vue", "next.js", "nuxt", "svelte", "tailwind",
    "html", "css", "sass", "webpack", "vite", "redux", "mobx", "graphql",
    # Backend
    "node.js", "django", "flask", "fastapi", "spring", "express", "rails",
    "laravel", "gin", "nestjs", "grpc", "rest", "soap",
    # Databases
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "sqlite",
    "cassandra", "dynamodb", "bigquery", "snowflake", "clickhouse",
    "neo4j", "oracle", "mariadb",
    # Cloud & DevOps
    "aws", "gcp", "azure", "docker", "kubernetes", "k8s", "terraform",
    "ansible", "helm", "jenkins", "github actions", "gitlab ci", "circleci",
    "argocd", "pulumi", "prometheus", "grafana", "datadog", "sentry",
    "nginx", "apache",
    # Data / ML / AI
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
    "xgboost", "lightgbm", "spark", "hadoop", "airflow", "dbt", "mlflow",
    "huggingface", "transformers", "langchain", "openai", "llm", "rag",
    "pinecone", "weaviate", "chroma",
    # Messaging
    "kafka", "rabbitmq", "celery", "sqs", "pubsub",
    # Tooling & Practices
    "git", "github", "gitlab", "bitbucket", "jira", "confluence",
    "agile", "scrum", "ci/cd", "linux", "bash", "shell", "sql", "nosql",
]
