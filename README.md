# AI Recommendation Logic 🎯

> Project 3 — DecodeLabs Industrial Training | Batch 2026

A content-based Tech Stack Recommender that maps user skills to the most relevant job roles using TF-IDF vectorization and Cosine Similarity.

---

## Results

| Input Skills | #1 Recommendation | Match Score |
|---|---|---|
| python, machine_learning, sql | Data Scientist | 56.9% |
| docker, kubernetes, aws | Cloud Architect | 65.1% |
| python, nlp, transformers | NLP Engineer | 68.3% |

---

## Project Structure

```
RECOMMENDER_PROJECT/
├── recommender.py     # Main recommendation engine
├── raw_skills.csv     # Job roles dataset
├── requirements.txt   # Dependencies
├── .gitignore         # Files to ignore
└── README.md          # Documentation
```

---

## Pipeline

```
INPUT              PROCESS                    OUTPUT
──────────         ──────────────────────     ──────────────
User Skills   →    TF-IDF Vectorization   →   Top 3 Job Roles
(min 3)            Cosine Similarity           Ranked by Score
                   Sorting & Filtering
```

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/your-username/recommender-project.git

# 2. Navigate to folder
cd recommender-project

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the recommender
python recommender.py
```

---

## Example Output

```
==================================================
   TOP RECOMMENDATIONS FOR YOU
==================================================

  #1 NLP Engineer
      Match Score : 68.3%
      Skills      : python nlp transformers huggingface pytorch

  #2 ML Engineer
      Match Score : 8.0%
      Skills      : python machine_learning deep_learning tensorflow

  #3 AI Research Scientist
      Match Score : 7.7%
      Skills      : python mathematics deep_learning research
==================================================
```

---

## Key Concepts

| Concept | Implementation |
|---------|---------------|
| Content-Based Filtering | Match user skills to item attributes |
| TF-IDF Vectorization | Convert skills to weighted vectors |
| Cosine Similarity | Measure angle between vectors |
| Top-N Filtering | Prevent choice overload |
| Cold Start Handling | Onboarding survey (min 3 skills) |

---

## Why Cosine Similarity over Euclidean Distance?

```
Euclidean → sensitive to vector magnitude (size)
Cosine    → measures orientation only (direction)

A user with 3 skills vs a job with 10 skills:
Euclidean → big distance (misleading)
Cosine    → correct match based on direction ✅
```

---

## Author

**Yahia** — NLP Engineer & AI Graduate
Faculty of Artificial Intelligence, Kafr El-Sheikh University (2023)

---

## License

MIT License
