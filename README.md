# ”DS 4320 Project 1: FILL IN TITLE HERE

### Executive Summary
Paragraph goes in here

<br>

<br>

---

---
| Spec | Value |
|:---|:---|
| Name | Benjamin Berinsky |
| NetID | tfu5hw |
| DOI | [ADD IN HERE](add valid link) |
| Press Release | [ADD IN HERE](add valid link) |
| Data | [Folder](https://1drv.ms/f/c/1A9305D23C7E8C8F/IgD3vwQQw6D5SaeZn0SAQzdtAVopiUO-ln5e1Mwty63JP0Q?e=vC3ilk) |
| Pipeline | [ADD IN HERE](add valid link) |
| License | [MIT](https://github.com/benberinsky/ds4320-project-1/blob/main/LICENSE.md) |

---
<br>

<br>

## Problem Definition

### General and Specific Problem
* **General Problem:** Can we predict upsets in the NCAA tournament?
* **Specific Problem:** What regular season metrics best characterize teams that upset higher seeds in March Madness, and do these patterns hold over time (allowing for upset forecasting before games take place) or are upsets due to complete random chance?

### Rationale
The general problem identifies the broad problem space, upset prediction in March Madness, but lacks specific, actionable direction. The specific problem addresses this by identifying regular season team metrics as the data source for analyzing upset probability. The specific problem also expands the scope beyond a single season/tournament, framing the analysis as a study over time to determine whether upset patterns are consistent rather than isolated occurrences. This connects directly back to the general problem of predicting game outcomes, as consistent patterns would enable pre-tournament forecasting. Finally, the specific problem  acknowledges the null hypothesis, that upsets may reflect pure randomness with no consistent predictive signal. Without this clause, confirmation bias may be introduced into analysis to find some sort of correlation over time.

### Motivation
Every year, March Madness rolls around and Americans across the country fill out their brackets. Even with millions of predictions, a perfect bracket has never been recorded, with the odds estimated at around 1 in 9.2 quintillion. While a perfect bracket may never be achievable, upsets remain the single biggest bracket-buster, with lower knocking out favorites before the first day of the tournament even concludes. This project aims to answer the question: is there a science to identifying March Madness upsets, or are they the result of pure random chance? If certain metrics prove to be strong predictors of upsets, that slight edge could be the difference between office pool glory and watching your coworker walk away with $100 that should've been yours.

### Press Release Headline and Link
[**Add Headline Here**](add link here)

## Domain Exposition

### Terminology
---

| Term | Description |
|:------|:-------------|
| Seed | Ranking from 1-16 within each of the four quadrants of the bracket|
| Upset | When the lower seeded team beats the higher seeded team |
| Cinderella | Low seeded team that makes a deep run in the tournament |
| Offensive efficiency  | Points scored per 100 offensive possessions |
| Pace/Tempo | An estimate of the number of possessions a team has per regulation (40 minutes) |
| Defensive efficiency | Statistic that calculates the points allowed per 100 defensive possessions |
| KPI | Ranking of how good/bad wins/losses are from -1 to +1 |
| POM | Rankings from Kenpom website that uses advanced statistics to rank teams |
| Strength of Schedule (SOS) | Measure of difficulty of a team's opponents|

---

### Domain Explanation
This project lives in the domain of men's college basketball as well as predictive sports analytics. March Madness is the hallmark of college basketball, a 68 team tournament beginning in March in which teams must go through multiple rounds to win the championship. March Madness brackets are incredibly popular, and even those who are not typically interested in sports often fill out brackets for social or work groups. With the growing popularity of sports betting, March Madness is becoming more and more lucrative. Although predictions are often made randomly/based on gut feeling, predictive sports analytics uses metrics from the past to forecast outcomes. It is often used by sports gamblers and sports books to both set the odds and find potential edges where money can be made from gambling and can be used to forecast upsets in March Madness. This project sits at the intersection of both domains, applying predictive analytics to one of the most unpredictable events in sports.

### Background Reading
[Folder here]([add valid link](https://github.com/benberinsky/ds4320-project-1/tree/main/background_reading))

### Reading Summary Table
---

| # | Title | Description | Link |
|:---|:-------|:-------------|:------|
| 1 | "Here's how to pick March Madness men's upsets, according to the data" | History of number of upsets in March Madness by year | [View Paper](https://github.com/benberinsky/ds4320-project-1/blob/main/background_reading/Here's%20how%20to%20pick%20March%20Madness%20men's%20upsets%2C%20according%20to%20the%20data%20_%20NCAA.com.pdf) |
| 2 | "March Madness Betting Projected to Hit Record of $4B, Even Without Prediction Markets" | March Madness sports betting growing to unprecedented level in 2026 | [View Paper](https://github.com/benberinsky/ds4320-project-1/blob/main/background_reading/March%20Madness%20Betting%20Projected%20to%20Hit%20Record%20of%20%244B%2C%20Even%20Without%20Prediction%20Markets%20-%20Yahoo%20Sports.pdf) |
| 3 | "How Predictive Analytics in Sports Transform Athletic Performance & Business Operations" | Overview of process and application of predictive sports analytics | [View Paper](https://github.com/benberinsky/ds4320-project-1/blob/main/background_reading/Predictive%20Analytics%20in%20Sports_%20Use%20Cases%20%26%20Future%20Trends.pdf) |
| 4 | "The math behind March Madness" | How sports analytics are used to select teams for the tournament, optimal bracket selection strategy | [View Paper](https://github.com/benberinsky/ds4320-project-1/blob/main/background_reading/The%20math%20behind%20March%20Madness%20_%20Penn%20Today.pdf) |
| 5 | "What is March Madness: The NCAA tournament explained" | Overview of March Madness, logistics, key terms | [View Paper](https://github.com/benberinsky/ds4320-project-1/blob/main/background_reading/What%20is%20March%20Madness_%20The%20NCAA%20tournament%20explained%20_%20NCAA.com.pdf) |

---

## Data Creation

### Data Acquisition (Provenance)
To build the dataset, I used two separate data sources. The first was the KenPom API, which hosts historical team-level data back to 2002, the year that KenPom data started being recorded. For the KenPom API, an API key is needed, which I had to provide when pulling in the data. This is not included in the script for security purposes but was input when run. This data includes advanced metrics such as efficiency, luck, tempo, and much more. To retrieve these stats, I looped through by year (from 2002–2025) and compiled the results into distinct dataframes based on API endpoint — rating, four-factors, height, and point distribution. I also pulled in historical information about team name, ID, and coach by year and saved these as well. I fitered the KenPom dataframes to only include relavent information (statistical rankings, not raw numbers). The dataframes were then saved as CSV and Parquet files.

The second was the 'March Machine Learning Mania 2026' Kaggle competition. The data in this competition has historical data for tournament games, teams, seeds, results, and much more. Once I pulled in many of these CSV files, I joined them and altered the structure to form two tables — tournament game results and historic tournament seeds. I had to parse the seed from raw string values, filter down to only games since 2002, merge winners and losers into the same row to represent one game, derive the round number from the order of game dates, and add a flag indicating whether or not games were upsets. I made the decision to encode upsets as a seed differential for 2+ with the lower seeded team beating the higher one. For tournament game information, I filtered the dataframe to only include wins/losses/score/team names because stats were not relevant to my question of interest. Because KenPom and Kaggle use different naming conventions, I built a mapping dictionary of about 90 team names to standardize across sources. The dataframe was then saved as CSV and Parquet files.

### Data Acquisition code

---

| File Name | Description | Link |
|:-----------|:-------------|:------|
| fetch_kenpom_stats.py | Fetches historical season-level stats for mens CBB teams, <br> creating and saving files to data/raw for ratings,  four-factors,<br> height, and point distribution (from KenPom API) | [Link](https://github.com/benberinsky/ds4320-project-1/blob/main/scripts/fetch_data/fetch_kenpom_stats.py) |
| fetch_kenpom_teams.py | Fetches team info from KenPom API including team name, <br> ID, conference, coach, and arena info for each season, saves <br> to data/raw| [Link](https://github.com/benberinsky/ds4320-project-1/blob/main/scripts/fetch_data/fetch_kenpom_teams.py) |
| clean.py | Cleans and standardizes raw data from KenPom and Kaggle <br>sources. Applies team name mapping, derives upset flags, <br>assigns tournament rounds, and outputs tables to data/clean/| [Link](https://github.com/benberinsky/ds4320-project-1/blob/main/scripts/fetch_data/clean.py) |
| utils.py | Provides logging, authentication, and save functions <br> used across all fetch and cleaning scripts. | [Link](https://github.com/benberinsky/ds4320-project-1/blob/main/scripts/fetch_data/utils.py) |

---

### Bias Identification
In the data collection process, most of the data was pulled directly from reliable online college basketball analytics sources without alteration. One subjective decision that I made was defining an upset. I determined that a game was an upset if a team that was more than one seed below a team won. This may introduce bias in results because if others define upsets differently because they may find different results in their analysis. Further, since the KenPom data for some statistics only goes back until 2010, while others go back until 2002, there may be some selection bias introduced that not does allow for holistic analysis of historical March Madness data. My decision to use rankings based off years rather than raw numbers as my main source of statistics was a judgement call that could be biased as well.

### Bias Mitigation
In my analysis and findings I will clearly document the definition of upsets. I defined it as a team seeded more than one line below their opponent winning. It is also important to document the process of cleaning data and mapping team names. Additionally, the clear documentation of handling of missing data rather than just dropping years/metrics is included.

### Critical Decision Rationale
The most crucial judgement call I made was to define upsets as a seed difference greater than one. I chose to do this because I felt that teams within one seed of each other are not ranked significantly differently. By looking only at multi-seed differences, we weed out games that were coin flips and not significantly upsets. It is possible that a 2 seed difference is not significant enough, and a 3+ seed difference distinction is necessary. In the process of my analysis, I will interpret my results with caution and this condition in mind.

I also chose to analyze data from 2002 onwards. This decision was made because of the data I have access to, but it notably excludes older historical trends. Since some tables only have information from 2010 onwards, my final analysis may be in the scope of 2010-2026. I dropped games from the play-in/first four, but these teams all have the same seed so no games would be classified as upsets. Rounds were assigned based on order of dates as I know that rounds progress in chronological order, but this assignment was still a subjective decision made by me. I selected national rankings for my analysis rather than raw statistics because trends over time change. I want to view teams relative to other teams in the tournament/NCAA in that specific year, not compare them to all other teams historically. Finally, I mapped team names to standardize and join across tables. This could potentially create uncertainty if any of the matches I identified were incorrect.

## Metadata

### ER Diagram
Add image in here

### Data
Update table below

---

| Spec | Value |
|---|---|
| Name | Benjamin Berinsky |
| NetID | tfu5hw |
| DOI | [ADD IN HERE](add valid link) |
| Press Release | [ADD IN HERE](add valid link) |
| Data | [ADD IN HERE](add valid link) |
| Pipeline | [ADD IN HERE](add valid link) |
| License | [ADD IN HERE](add valid link) |

---

### Data Dictionary
Update table below

---

| Spec | Value |
|---|---|
| Name | Benjamin Berinsky |
| NetID | tfu5hw |
| DOI | [ADD IN HERE](add valid link) |
| Press Release | [ADD IN HERE](add valid link) |
| Data | [ADD IN HERE](add valid link) |
| Pipeline | [ADD IN HERE](add valid link) |
| License | [ADD IN HERE](add valid link) |

---

### Uncertainty Quantification
Update table below

---

| Spec | Value |
|---|---|
| Name | Benjamin Berinsky |
| NetID | tfu5hw |
| DOI | [ADD IN HERE](add valid link) |
| Press Release | [ADD IN HERE](add valid link) |
| Data | [ADD IN HERE](add valid link) |
| Pipeline | [ADD IN HERE](add valid link) |
| License | [ADD IN HERE](add valid link) |

---
