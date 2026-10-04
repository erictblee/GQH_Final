"""Every fixed setting of the strategy, in one place.

Each number here comes from HYPOTHESIS.md, which was committed before any
backtest code existed. No other file is allowed to hardcode a strategy
number: they all do `import config` and read the value from here.

DO NOT change a value to make results look better. If a value seems wrong,
change HYPOTHESIS.md first (as a dated amendment) and explain why.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Universe (HYPOTHESIS.md, "Universe")
# ---------------------------------------------------------------------------

# The three futures we trade. The order matters: every table and array in the
# code keeps the assets in exactly this order.
ASSETS = ["ES", "ZN", "GC"]

# Dollar value of a 1.00 move in the quoted price, for ONE contract.
# These are CME contract specifications, not choices of ours:
#   ES: $50 x S&P 500 index points
#   ZN: $1,000 per point (the contract is $100,000 face value, quoted per $100)
#   GC: 100 troy ounces, quoted in dollars per ounce
MULTIPLIER = {"ES": 50.0, "ZN": 1000.0, "GC": 100.0}
# Grains (used only by the ENSO strategy, see the end of this file). Each
# contract is 5,000 bushels and the price is quoted in CENTS per bushel, so a
# 1.00 (one cent) move is worth 5,000 x $0.01 = $50.
MULTIPLIER.update({"ZC": 50.0, "ZS": 50.0, "ZW": 50.0})

# Calendar months of the contracts we are allowed to hold (1 = January).
CONTRACT_MONTHS = {
    "ES": [3, 6, 9, 12],            # quarterly
    "ZN": [3, 6, 9, 12],            # quarterly
    "GC": [2, 4, 6, 8, 10, 12],     # liquid even months only
    "ZC": [3, 5, 7, 9, 12],         # corn: every listed month
    "ZS": [1, 3, 5, 7, 8, 9, 11],   # soybeans: every listed month
    "ZW": [3, 5, 7, 9, 12],         # wheat: every listed month
}

# ---------------------------------------------------------------------------
# Rolls (HYPOTHESIS.md, "Prices, contracts and rolls")
# ---------------------------------------------------------------------------

# We leave a contract this many business days before its "anchor" date.
ROLL_DAYS_BEFORE = 5

# The anchor date each asset's roll is counted back from.
#   "expiration"   : the contract's last trading day (ES is cash-settled).
#   "first_notice" : first notice day = last business day of the month before
#                    the contract month (ZN and GC can be physically delivered,
#                    so we must be out before notices start).
ROLL_ANCHOR = {"ES": "expiration", "ZN": "first_notice", "GC": "first_notice"}
# Grains can be physically delivered, so they use first notice day too.
ROLL_ANCHOR.update({"ZC": "first_notice", "ZS": "first_notice", "ZW": "first_notice"})

# ---------------------------------------------------------------------------
# Signal settings (HYPOTHESIS.md, "Strategy definitions")
# ---------------------------------------------------------------------------

TRADING_DAYS = 252          # trading days per year, used to annualize
MAIN_WINDOW = 60            # days of returns used for volatility / covariance
PLATEAU_WINDOWS = [40, 90, 120]   # alternative windows for prediction 6
WARMUP_DAYS = 120           # common start = 120 trading days after first data
VOL_TARGET = 0.10           # sigma*: target portfolio volatility, per year

# ---------------------------------------------------------------------------
# Rules applied to both versions (HYPOTHESIS.md, "Rules applied identically")
# ---------------------------------------------------------------------------

GROSS_CAP = 4.0             # maximum sum of target weights (4x leverage)
BRAKE_ON = 0.15             # brake switches ON when drawdown is above 15%
BRAKE_OFF = 0.075           # brake switches OFF when drawdown is below 7.5%
BRAKE_VOL_TARGET = 0.05     # sigma* while the brake is on
BAND = 0.20                 # trade only if |current - target| > 20% of target
COST_BPS = 2.0              # cost per side, in basis points of traded notional
STRESS_COST_BPS = 4.0       # cost used in the stress test
CAPITAL = 10_000_000.0      # starting equity in dollars

# ---------------------------------------------------------------------------
# Benchmarks (HYPOTHESIS.md, "Benchmarks")
# ---------------------------------------------------------------------------

EQUAL_WEIGHT = {"ES": 1 / 3, "ZN": 1 / 3, "GC": 1 / 3}
SIXTY_FORTY = {"ES": 0.60, "ZN": 0.40}

# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

# First day of Databento's GLBX.MDP3 history. Data starts here.
DATA_START = "2010-06-06"

# Last day of the IN-SAMPLE period. Everything after this date is the
# out-of-sample holdout and is only ever loaded with an explicit --final flag.
IS_END = "2024-09-30"

# First day of the OUT-OF-SAMPLE period (HYPOTHESIS.md, "Sample periods").
OOS_START = "2024-10-01"

# ---------------------------------------------------------------------------
# Diagnostics (HYPOTHESIS.md, Amendment 1). None of these affect trading.
# ---------------------------------------------------------------------------

# Smallest price step of each contract (CME contract specifications).
TICK_SIZE = {"ES": 0.25, "ZN": 1 / 64, "GC": 0.10}
# Assumed exchange + clearing + broker fee, dollars per contract per side.
FEE_PER_CONTRACT = 2.50

# Capacity test: fund sizes in dollars, and the share of B's Sharpe ratio a
# fund size must keep to count as "within capacity".
FUND_SIZES = [10e6, 100e6, 500e6, 1e9, 5e9, 10e9, 25e9]
CAPACITY_SHARPE_KEPT = 0.75
IMPACT_COEFFICIENT = 1.0        # impact = 1.0 x daily vol x sqrt(traded / volume)
IMPACT_VOL_WINDOW = 60          # days of returns before the trade
IMPACT_VOLUME_WINDOW = 20       # days of cleared volume before the trade

# Stress windows, fixed in Amendment 1 from general market history.
STRESS_WINDOWS = {
    "2011 US downgrade": ("2011-07-22", "2011-10-03"),
    "2013 taper tantrum": ("2013-05-22", "2013-09-05"),
    "2015 China devaluation": ("2015-08-10", "2015-08-25"),
    "2018 Volmageddon": ("2018-01-26", "2018-02-08"),
    "2018 Q4 selloff": ("2018-09-20", "2018-12-24"),
    "2020 COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 inflation shock": ("2022-01-03", "2022-10-12"),
    "2023 regional banks": ("2023-03-08", "2023-03-24"),
    "2024 August volatility spike": ("2024-07-16", "2024-08-05"),
    "2025 tariff selloff": ("2025-02-19", "2025-04-08"),     # out of sample
}

# Stationary block bootstrap (Politis & Romano 1994).
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_MEAN_BLOCK = 20       # average block length in days
BOOTSTRAP_SEED = 12345          # fixed, so the numbers can be reproduced

# FRED series used for the regime diagnostics and the total-return figure.
FRED_SERIES = ["T10YIE", "VIXCLS", "DFII10", "BAMLH0A0HYM2", "DTB3"]

# CME initial margins, dollars per contract, with the date they were read.
# Left empty: the margin diagnostic is skipped until the team enters them.
INITIAL_MARGIN = {}
INITIAL_MARGIN_DATE = None

# ---------------------------------------------------------------------------
# ENSO strategy on corn, soybeans and wheat (ENSO_SPEC.md)
#
# A SECOND strategy, separate from the ES/ZN/GC risk parity above. Every
# number below was written down in ENSO_SPEC.md before any backtest was run.
# It reuses from above: TRADING_DAYS, VOL_TARGET (10%), GROSS_CAP (4x),
# COST_BPS (2), STRESS_COST_BPS (4), CAPITAL, IS_END and OOS_START.
# ---------------------------------------------------------------------------

AG_ASSETS = ["ZC", "ZS", "ZW"]      # corn, soybeans, wheat (CME / CBOT)

# Sample periods. Five years of futures data in total, never more:
#   development   2021-10-01 to 2023-09-30   (build and check the code)
#   validation    2023-10-01 to 2024-09-30   (look at design choices)
#   out-of-sample 2024-10-01 to 2026-09-30   (locked; final run only)
AG_DATA_START = "2021-10-01"
AG_DEV_END = "2023-09-30"
AG_OOS_END = "2026-09-30"
AG_WARMUP_DAYS = 61                 # days of data needed before the first trade

# Signal (ENSO_SPEC.md sections 6 to 9)
MOM_LOOKBACK = 60                   # momentum lookback, trading days
ALPHA_W_MOM = 1.0                   # weight of momentum in the alpha score
ALPHA_W_ENSO = 1.0                  # weight of the ENSO effect in the alpha score
ENSO_Z_MONTHS = 360                 # 30 years of monthly values for the z-score
ENSO_THRESHOLD = 0.5                # z >= 0.5: El Nino.  z <= -0.5: La Nina
ENSO_RELEASE_DAY = 15               # month M's value is usable from the 15th of month M+1
# What each ENSO state means for each price: +1 bullish, -1 bearish, 0 no view.
# From crop yields in Iizumi et al. (2014, Nature Communications 5:3712):
# La Nina lowers the world yield of all three crops (less grain = higher
# price); El Nino raises soybean yields and has a mixed effect on maize and wheat.
ENSO_EFFECT = {
    "ZC": {"el_nino": 0.0, "neutral": 0.0, "la_nina": 1.0},
    "ZS": {"el_nino": -1.0, "neutral": 0.0, "la_nina": 1.0},
    "ZW": {"el_nino": 0.0, "neutral": 0.0, "la_nina": 1.0},
}

# Risk and portfolio construction (sections 10 to 16)
AG_WINDOW = 60                      # days of returns for volatility and covariance
SHRINKAGE = 0.5                     # lambda: 0.5 x rolling covariance + 0.5 x diagonal
TILT_K = 1.0                        # k: how strongly alpha tilts the risk budgets
MAX_RISK_SHARE = 0.70               # one asset may carry at most 70% of the risk
MARGIN_CAP = 0.50                   # at most half of equity posted as margin
# CME initial margins, dollars per contract. Left empty: the margin limit is
# skipped (and reported as skipped) until the team enters them.
AG_INITIAL_MARGIN = {}

# Robustness tests, fixed in advance (sections 6, 12, 13 and 20). Reported, never selected from.
MOM_ROBUST_LOOKBACKS = [20, 40]
SHRINKAGE_ROBUST = [0.0, 1.0]
TILT_K_ROBUST = [2.0]

# ---------------------------------------------------------------------------
# Cross-asset signal-weighted risk budgeting (XA_SPEC.md)
#
# The THIRD strategy: long/short trend on eleven futures in five groups, with
# the ENSO adjustment on the three grains only. Every number below was
# written down in XA_SPEC.md before any backtest of it was run. It reuses
# from above: TRADING_DAYS, VOL_TARGET, GROSS_CAP, MAX_RISK_SHARE, COST_BPS,
# STRESS_COST_BPS, CAPITAL, ENSO_* and IS_END / OOS_START.
# ---------------------------------------------------------------------------

XA_GROUPS = {
    "equity": ["ES", "NQ"],             # S&P 500, Nasdaq 100
    "bonds": ["ZN", "ZB"],              # 10-year and 30-year US Treasuries
    "energy": ["CL", "NG"],             # crude oil, natural gas
    "metals": ["GC", "HG"],             # gold, copper
    "agriculture": ["ZC", "ZS", "ZW"],  # corn, soybeans, wheat
}
XA_ASSETS = [asset for group in XA_GROUPS.values() for asset in group]

# CME contract specifications of the five markets that are new here.
#   NQ: $20 x index.  ZB: $1,000 per point.  CL: 1,000 barrels, $ per barrel.
#   NG: 10,000 mmBtu, $ per mmBtu.  HG: 25,000 pounds, $ per pound.
MULTIPLIER.update({"NQ": 20.0, "ZB": 1000.0, "CL": 1000.0, "NG": 10000.0, "HG": 25000.0})
CONTRACT_MONTHS.update({
    "NQ": [3, 6, 9, 12],
    "ZB": [3, 6, 9, 12],
    "CL": list(range(1, 13)),           # every month
    "NG": list(range(1, 13)),           # every month
    "HG": [3, 5, 7, 9, 12],             # the liquid months
})
ROLL_ANCHOR.update({"NQ": "expiration", "ZB": "first_notice", "CL": "first_notice",
                    "NG": "first_notice", "HG": "first_notice"})
# Crude oil and natural gas stop trading BEFORE the month before delivery
# ends, so for them the anchor is the last business day TWO months before
# the delivery month (one month for everything else).
ROLL_MONTHS_BEFORE = {"CL": 2, "NG": 2}

# Sample periods (data starts at DATA_START, Databento's first day).
#   development  first decision (mid 2011) to 2016-12-31
#   validation   2017-01-01 to 2019-12-31
#   test         2020-01-01 to 2024-09-30   locked: needs `--test`
#   final        2024-10-01 onward          locked: needs `--final`, not downloaded
XA_DEV_END = "2016-12-31"
XA_VALIDATION_END = "2019-12-31"
XA_TEST_START = "2020-01-01"

XA_MOM_LOOKBACK = 252               # 12-month trend, in trading days
XA_HALF_LIFE = 60                   # half-life of the EWMA volatility / covariance, days
XA_COV_WINDOW = 252                 # days of returns the EWMA looks back over
XA_WARMUP_DAYS = 253                # days of data needed before the first trade
XA_MIN_SIGNAL = 0.25                # no position if |alpha| is below this
XA_MAX_WEIGHT = 2.0                 # one asset's notional may not exceed 2x equity
XA_ROBUST_LOOKBACKS = [126]         # 6-month trend, as a robustness test only

# ---------------------------------------------------------------------------
# A/B/C/D attribution models (ABCD_PLAN.md)
#
# The same eleven futures, data, periods, roll rule, costs, weekly rebalance,
# trend lookback, EWMA covariance, 10% target and position limits as the
# cross-asset section above. Only what is new is listed here. Every number
# was written down before the first run of these models.
# ---------------------------------------------------------------------------

ABCD_EQUAL_GROSS = 1.0              # Model A: longs + shorts add up to 1x equity, no leverage
# The ENSO overlay as a multiplier on the grains' trend (plan section 4),
# tested separately from the additive form: lean in when ENSO agrees with
# the trend, lean out when it disagrees, never flip or close the position.
ABCD_ENSO_AGREE = 1.5
ABCD_ENSO_DISAGREE = 0.5

# Robustness of Model D, one setting changed at a time. Reported, never selected from.
ABCD_ROBUST_LOOKBACKS = [126, 189]  # 6- and 9-month trend (the main one is 12)
ABCD_ROBUST_HALF_LIVES = [40, 90]   # covariance half-life, days (the main one is 60)
ABCD_ROBUST_VOL_TARGETS = [0.08, 0.12]
ABCD_ROBUST_REBALANCE_WEEKS = [2]   # decide every second week instead of every week

# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------

REPO_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = REPO_DIR / "data" / "processed"   # input data (never committed)
RESULTS_DIR = REPO_DIR / "results"                # outputs (never committed)
EXTERNAL_DIR = REPO_DIR / "data" / "external"     # public FRED and Kenneth French files
AG_RAW_DIR = REPO_DIR / "data" / "raw_ag"             # grain downloads (never committed)
AG_PROCESSED_DIR = REPO_DIR / "data" / "processed_ag" # grain tables (never committed)
XA_RAW_DIR = REPO_DIR / "data" / "raw_xa"             # cross-asset downloads (never committed)
XA_PROCESSED_DIR = REPO_DIR / "data" / "processed_xa" # cross-asset tables (never committed)
