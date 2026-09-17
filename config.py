from pathlib import Path

REVISION_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REVISION_ROOT.parent
SOURCE_ROOT = PROJECT_ROOT / "source"
WORK_ROOT = REVISION_ROOT / "work"
RESULT_ROOT = REVISION_ROOT / "results"
CLEAN_ROOT = WORK_ROOT / "clean_docs"
SENTENCE_ROOT = WORK_ROOT / "sentences"
PROFILE_ROOT = WORK_ROOT / "profiles"
TABLE_ROOT = RESULT_ROOT / "tables"
FIGURE_ROOT = RESULT_ROOT / "figures"
AUDIT_ROOT = RESULT_ROOT / "audit"

ARENAS = ("NYT", "WSJ", "arxiv")
ARENA_LABELS = {
    "NYT": "New York Times (public news)",
    "WSJ": "Wall Street Journal (business news)",
    "arxiv": "arXiv (academic)",
}

# One authoritative target list. All tables and figures load this tuple.
TARGETS = (
    "agent", "algorithm", "alignment", "autonomy", "bias", "capability",
    "control", "creativity", "decision", "fairness", "generation",
    "intelligence", "learning", "model", "network", "prediction",
    "reasoning", "representation", "risk", "safety", "system", "training",
    "understanding",
)

# May be changed to an absolute local snapshot directory. The pipeline never
# silently substitutes a sentence-embedding model for this masked-LM model.
MODEL_NAME = str(PROJECT_ROOT / "model" / "bert-base-uncased")
MAX_SENTENCES = 500
MIN_ELIGIBLE_SENTENCES = 100
MAX_LENGTH = 128
BATCH_SIZE = 32
TOP_K = 50
CONTROL_REL_TOLERANCE = 0.10
MAX_CONTROLS = 100
SMALL_CONTROL_WARN = 30
SEEDS = (11, 22, 33, 44, 55)

# A sentence is AI-related if an unambiguous cue occurs within this many tokens
# on either side of the target. Target terms themselves do not count as cues.
AI_WINDOW_TOKENS = 35
AI_CUES = {
    "ai", "a.i", "artificial intelligence", "machine learning", "deep learning",
    "neural", "neural network", "transformer", "language model", "large language model",
    "llm", "chatgpt", "openai", "anthropic", "gemini", "deepmind", "algorithmic",
    "computer vision", "natural language processing", "nlp", "robot", "robotic",
    "automation", "automated", "computer", "computational", "software", "data science",
    "generative", "foundation model", "reinforcement learning", "classifier", "dataset",
}

# Clear non-AI senses. A nearby strong AI cue takes precedence except where the
# phrase itself is explicitly listed as a veto.
NON_AI_PATTERNS = {
    "intelligence": (
        r"\b(intelligence agency|military intelligence|foreign intelligence|"
        r"intelligence service|intelligence officer|intelligence community|"
        r"central intelligence agency|human intelligence)\b",
    ),
    "agent": (r"\b(real estate|travel|literary|talent|federal|police|border) agent\b",),
    "model": (r"\b(fashion|runway|photographic) model\b",),
    "network": (r"\b(television|tv|broadcast|cable|rail|road) network\b",),
    "learning": (r"\b(distance|classroom|student|child|children) learning\b",),
    "training": (r"\b(athletic|military|fitness|marathon|employee) training\b",),
    "control": (r"\b(birth|traffic|pest|rent|border) control\b",),
    "alignment": (r"\b(wheel|spinal|political) alignment\b",),
}

# Multiword terms are replaced as a whole by one mask. This prevents a retained
# modifier (especially "artificial") from turning MLM prediction into phrase
# completion. These sentences are retained but counted as phrase-adjusted.
MASK_EXPANSIONS = {
    "intelligence": (r"\bartificial\s+intelligence\b",),
    "learning": (r"\b(machine|deep|reinforcement)\s+learning\b",),
    "model": (r"\b(large\s+language|language|foundation|generative)\s+model\b",),
    "network": (r"\b(neural|deep\s+neural)\s+network\b",),
    "agent": (r"\b(ai|artificial[- ]intelligence|autonomous)\s+agent\b",),
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "for", "from", "had", "has", "have", "he", "her", "hers", "him", "his",
    "i", "if", "in", "into", "is", "it", "its", "me", "my", "no", "not",
    "of", "on", "or", "our", "ours", "she", "so", "that", "the", "their",
    "theirs", "them", "they", "this", "those", "to", "too", "us", "was", "we",
    "were", "what", "when", "where", "which", "who", "will", "with", "would",
    "you", "your", "yours",
}

# Exact-token anchor sets for the revised two-dimensional map. Coordinates are
# expectations under each filtered top-50 GSP: temporality = future - past;
# ontology = beyond-machine - machine. Present anchors are reported as an
# auxiliary score but do not enter the signed temporality coordinate.
ANCHORS = {
    "past": ("past", "previous", "historical", "formerly", "earlier", "legacy"),
    "present": ("present", "current", "today", "now", "existing", "contemporary"),
    "future": ("future", "forthcoming", "prospective", "emerging", "next", "upcoming"),
    "machine": ("machine", "computer", "software", "system", "algorithm", "automated",
                "computational", "technical", "digital", "robot"),
    "beyond": ("human", "social", "society", "cultural", "ethical", "political",
               "creative", "consciousness", "agency", "meaning"),
}

# Expanded semantic anchors used with MiniLM cosine similarity. Unlike ANCHORS,
# these do not require exact token matches. Each GSP substitute is embedded, its
# cosine similarity to each normalized anchor centroid is probability-weighted,
# and signed coordinates are future-past and beyond-machine.
SEMANTIC_ANCHORS = {
    "past": ("past", "history", "historical", "previous", "earlier", "legacy",
             "traditional", "classical", "precedent", "prior"),
    "future": ("future", "forthcoming", "prospective", "emerging", "next",
               "upcoming", "potential", "anticipated", "projected", "transformation"),
    "machine": ("machine", "computer", "software", "system", "algorithm", "automated",
                "computational", "technical", "digital", "robot", "program", "app",
                "application", "technology", "hardware", "code", "tool", "model",
                "method", "approach", "technique", "process"),
    "beyond": ("human", "individual", "social", "society", "cultural", "ethical",
               "political", "creative", "consciousness", "agency", "meaning",
               "intelligence", "think", "understand", "reason", "autonomous",
               "sentient", "mind", "cognitive", "existential"),
}
MINILM_MODEL_PATH = PROJECT_ROOT / "model" / "all-MiniLM-L6-v2-local"

for path in (WORK_ROOT, RESULT_ROOT, CLEAN_ROOT, SENTENCE_ROOT, PROFILE_ROOT,
             TABLE_ROOT, FIGURE_ROOT, AUDIT_ROOT):
    path.mkdir(parents=True, exist_ok=True)
