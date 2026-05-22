import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
#--------------------------------------------------------------------
# load dataset

def load_dataset(filepath):
    """Load the dataset and return it as a DataFrame"""
    df = pd.read_csv(filepath)
    return df


def explore_dataset(df):
    """Print basic information about the dataset"""
    print("=" * 50)
    print("TECH STACK RECOMMENDER - DATASET")
    print("=" * 50)
    print(f"\n[ Total Job Roles ] : {len(df)}")
    print(f"\n[ Job Roles Available ]")
    for i, role in enumerate(df['job_role'], 1):
        print(f"  {i:2}. {role}")
    print(f"\n[ Sample Skills - Data Scientist ]")
    print(f"  {df[df['job_role']=='Data Scientist']['skills'].values[0]}")
    print("\n" + "=" * 50)

#----------------------------------------------------------------------------
# vectorization

def build_tfidf_matrix(df):
    """
    Convert job role skills into TF-IDF vectors.
    Returns the vectorizer and the matrix.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['skills'])

    print("[ TF-IDF Matrix Shape ]")
    print(f"  Rows    : {tfidf_matrix.shape[0]}  (job roles)")
    print(f"  Columns : {tfidf_matrix.shape[1]}  (unique skills)")

    return vectorizer, tfidf_matrix

#---------------------------------------------------------------------------
# user input

def get_user_skills():
    """
    Ask the user to enter at least 3 skills.
    Returns a cleaned string of skills.
    """
    print("\n[ Enter Your Skills ]")
    print("  Type at least 3 skills separated by commas.")
    print("  Example: python, machine_learning, sql\n")

    raw = input("  Your skills: ")
    skills = [s.strip().lower().replace(" ", "_")
              for s in raw.split(",")]

    print(f"\n  Skills detected: {skills}")
    return " ".join(skills)

#---------------------------------------------------------------------------
# scoring and ranking

def calculate_similarity(vectorizer, tfidf_matrix, user_skills):
    """
    Calculate cosine similarity between
    user skills and all job roles.
    Returns similarity scores as a list.
    """
    
    user_vector = vectorizer.transform([user_skills])

    scores = cosine_similarity(user_vector, tfidf_matrix)

    return scores.flatten()


def get_top_recommendations(df, scores, top_n=3):
    """
    Sort job roles by similarity score.
    Return top N recommendations.
    """
    df_copy = df.copy()
    df_copy['score'] = scores

    df_sorted = df_copy.sort_values('score', ascending=False)

    return df_sorted.head(top_n)


def display_recommendations(recommendations):
    """
    Display the top recommended job roles
    with their similarity scores.
    """
    print("\n" + "=" * 50)
    print("   TOP RECOMMENDATIONS FOR YOU")
    print("=" * 50)

    for rank, (_, row) in enumerate(recommendations.iterrows(), 1):
        score_percent = round(row['score'] * 100, 1)
        print(f"\n  #{rank} {row['job_role']}")
        print(f"      Match Score : {score_percent}%")
        print(f"      Skills      : {row['skills']}")

    print("\n" + "=" * 50)
#----------------------------------------------------------------------------
# entry point

def main():
    
    df = load_dataset('raw_skills.csv')
    explore_dataset(df)

    vectorizer, tfidf_matrix = build_tfidf_matrix(df)

    user_skills = get_user_skills()

    scores = calculate_similarity(vectorizer, tfidf_matrix, user_skills)

    recommendations = get_top_recommendations(df, scores, top_n=3)

    display_recommendations(recommendations)

if __name__ == "__main__":
    main()