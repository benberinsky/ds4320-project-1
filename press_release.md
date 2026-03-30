# Stop Guessing - A Data Driven Guide to Predicting March Madness Upsets

## March Madness Doesn't Need to be a Blind Gamble

Each year, around 100 million March Madness brackets are filled out across the country. Office pools, group chats with friends, professional gambling websites, you name it — almost everyone has a bracket. A vast majority of these brackets are filled out with the same process: gut feeling, favorite team bias, or simply random guessing. Whether you're playing for bragging rights or big money, you're probably making upset picks with zero strategy behind them. What if, instead of just randomly picking teams that feel right for a cinderella run you could look at what actually separates the underdogs that win from the ones that don't?


## The Problem: March Madness Is Unpredictable, but Not Random

Although many have tried to build March Madness prediction models, there is unfortunately no secret formula to find the perfect bracket. The tournament is built on chaos (or Madness), and the single-elimination format unexpected results are inevitable. But here's what most people miss: **not all underdogs are created equal.** An 11-seed from a power conference that played a grueling schedule all year is a completely different team than one that ran through a weak conference uncontested. Yet in most brackets, people treat them the same way.

Instead of trying to predict individual game outcomes, we took a different approach. Using 15 years of KenPom data (2010–2024), we grouped every underdog in the tournament based on their regular season statistical profile — offensive and defensive efficiency, shooting percentages, turnover rates, strength of schedule, experience, and more — all aiming to answer the question: **Which types of underdogs actually pull off upsets, and which ones almost never do?**

## What We Found: Not All Underdogs Are Created Equal

After grouping every underdog from the last 15 March Madnesses by their regular season stats, five distinct profiles emerged with distinct differences in upset rate between groups (findings referenced can be found in figure one)

**"High-Powered Offenses"** (41% upset rate): These teams score at a high rate. They shoot the ball well, take care of it, and get to the free throw line. Their defense is not ranked as highly which is likely why they were underdogs to begin. But when you can put up points against anyone, one hot shooting night is all it takes. Nearly half of these underdogs pull off the upset. Look for lower seeded teams with high effective field goal %, low turnover rates to beat higher seeded teams in your bracket.

**"Battle-Tested & Unlucky"** (30% upset rate): These are the teams that spent all season playing highly ranked opponents, likely members of tough conferences. Their stats and overall win % might not look the prettiest, but in the right matchup they cannot be underestimated. They've played the best of the best all year long and come into games without fear. Think mid-seeds from power conferences that feel underseeded. At least a couple of these teams pull off upsets every year, and they're strong candidates for your bracket. Look for teams that played tough schedules during the regular season, with decently efficient offenses and defenses, and tall lineups.

**Avoid at all costs: "Weak Conference Winners"** (6% upset rate): This is where Cinderella dreams go to die. These are your 14, 15, and 16 seeds — teams with great records built against weak opponents. They look good on paper, but when they face a real team in the tournament, they get exposed. Picking one of these teams is ill advised. Stay away from very low seeds with high regular season win percentages but super easy schedules.

The remaining two groups — defensive-minded underdogs (29%) and teams from soft schedules (10%) — fall somewhere in between. But across the board, the same pattern holds: the underdogs that win can score, and they've been tested.

### The Metrics That Actually Matter

So what separates these groups? We looked at which metrics were the biggest determinants of separating teams into their respective clusters. The biggest differentiators:

- **Offensive efficiency** — how well a team scores relative to their possessions. The single most important separator encapsulating offensive performance in one metric.
- **Seed gap** — how big the mismatch is on paper. Bigger gaps obviously make upsets harder, but it's not the whole story.
- **Defensive efficiency** and **strength of schedule** — round out the top four. Notably, offense trumps defense

(findings referenced can be found in figure two with numbers representing normalized importance)

What *doesn't* matter nearly as much? Bench depth, team height, experience, and tempo. These metrics barely move the needle when it comes to distinguishing the underdogs that win from the ones that don't. Unfortunately there's no hidden "secret stat" - the fundamentals tell you what you need to know.

### Your Cheat Sheet for Next March** 

When you're filling out your bracket and deciding where to pick upsets, keep it simple:

1. **Look for offense**: Find the underdog that can shoot and doesn't turn the ball over. A team that's strong on offense but shaky on defense is exactly the kind of lower seed that pulls off upsets.

2. **Respect the schedule**: A team that battled through Big East, Big 12, or SEC play all season is tournament-ready. A team that went 27-4 against a weak conference is not.

3. **Don't chase the Cinderella**: That 15-over-2 upset makes SportsCenter, but it almost never happens. The data says those matchups go to the favorite about 94% of the time. Spend your upset picks where they count, don't waste them on chasing a miracle.

## Charts:
![Figure 1: Underdog Archetypes](https://github.com/benberinsky/ds4320-project-1/blob/main/figures/archetype_profiles.png)
*Each panel shows one underdog type. Green bars (left) indicate areas where the underdog is closer to the favorite — a relative strength. Orange bars (right) indicate where the gap is largest — a weakness.*

![Figure 2: Underdog Metric Importance](https://github.com/benberinsky/ds4320-project-1/blob/main/figures/feature_importance.png)
*Features are ranked by how much they vary across the five underdog clusters. Higher values mean that metric plays a bigger role in defining which archetype a team falls into. Red = primary separator, orange = secondary, light blue = minor.*