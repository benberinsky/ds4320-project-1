# Check what columns the Kaggle files actually have
import pandas as pd

results = pd.read_csv("data/raw/MNCAATourneyDetailedResults.csv")
seeds = pd.read_csv("data/raw/MNCAATourneySeeds.csv")
teams = pd.read_csv("data/raw/MTeams.csv")

print("Results columns:", results.columns.tolist())
print("Seeds columns:", seeds.columns.tolist())
print("Teams columns:", teams.columns.tolist())
