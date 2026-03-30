# HEADLINE
- Something about finding statistical groupings to win march madness prediction pools

## HOOK
- Every year, millions of March Madness brackets are filled out across the country, whether it be just for fun or high-stakes gambling
- Often these brackets are filled out based off of instinct or 'gut feeling'. Whether you are playing for bragging rights or thousands of dollars, ...

## PROBLEM STATEMENT
- No proven models that capture randomness or can accurately predict all games
- Instead of predicting individual games, look at statistical profiles and see what metrics are most important
- Group underdogs based off statistical profiles to see how upset rate differs
- Look at what features are most important for underdogs being placed into respective groups


## SOLUTION DESCRIPTION
- Adjusted offense and defensive efficiency are highly important, capture a lot of stats 
- Teams that are explosive on offense have high potential to pull off upsets
  - Cluster with highest upset rate: Really good shooting %, great turnover margin, not as strong on defense
  - Effective field goal % very high up on feature importance (more than defensive)
  - If teams are really strong on offense but weak on defense they may be seeded lower but have strong upset potential
- Second highest upset rate: Teams with really hard strength of schedule, may be unlucky, worth considering
- Teams with drastically lower SOS seem to not perform well
- Lowest seeds (14/15/16) can be seen mainly grouped in "Weak Conference Winners", huge seed gap, 6.1% upset rate, not worth predicting them to win

## CHART
[Figure 1: Underdog Archetypes](https://github.com/benberinsky/ds4320-project-1/blob/main/figures/archetype_profiles.png)
[Figure 2: Underdog Metric Importance](https://github.com/benberinsky/ds4320-project-1/blob/main/figures/feature_importance.png)