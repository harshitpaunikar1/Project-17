"""
Power user curve analysis for SaaS and consumer products.
Computes L30/L7/DAU metrics, retention curves, and power user identification.
"""
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class ActivityMetrics:
    """
    Computes daily, weekly, and monthly activity metrics from event logs.
    Activity logs expected: user_id, event_date columns.
    """

    def __init__(self, events_df: pd.DataFrame,
                 user_col: str = "user_id",
                 date_col: str = "event_date"):
        self.df = events_df.copy()
        self.user_col = user_col
        self.date_col = date_col
        self.df[date_col] = pd.to_datetime(self.df[date_col])

    def dau_series(self) -> pd.Series:
        """Daily Active Users over time."""
        return (self.df.groupby(self.date_col)[self.user_col]
                .nunique()
                .rename("dau"))

    def wau_series(self) -> pd.Series:
        """Weekly Active Users (rolling 7-day window)."""
        dau = self.dau_series().reset_index()
        dau = dau.set_index(self.date_col)
        return (self.df.groupby(pd.Grouper(key=self.date_col, freq="W"))[self.user_col]
                .nunique()
                .rename("wau"))

    def mau_series(self) -> pd.Series:
        """Monthly Active Users."""
        return (self.df.groupby(pd.Grouper(key=self.date_col, freq="MS"))[self.user_col]
                .nunique()
                .rename("mau"))

    def dau_mau_ratio(self) -> pd.DataFrame:
        """DAU/MAU stickiness ratio per month."""
        dau = self.dau_series()
        mau = self.mau_series()
        dau_df = dau.reset_index()
        dau_df["month"] = dau_df[self.date_col].dt.to_period("M")
        avg_dau = dau_df.groupby("month")["dau"].mean().reset_index()
        avg_dau.columns = ["month", "avg_dau"]
        mau_df = mau.reset_index()
        mau_df["month"] = mau_df[self.date_col].dt.to_period("M")
        merged = avg_dau.merge(mau_df[["month", "mau"]], on="month")
        merged["stickiness"] = (merged["avg_dau"] / merged["mau"].replace(0, np.nan)).round(4)
        return merged

    def l_n_distribution(self, window_days: int = 30) -> pd.Series:
        """
        L(N) distribution: fraction of users who were active on exactly N days
        in a rolling window_days period.
        """
        max_date = self.df[self.date_col].max()
        min_date = max_date - pd.Timedelta(days=window_days - 1)
        recent = self.df[self.df[self.date_col] >= min_date]
        user_days = recent.groupby(self.user_col)[self.date_col].nunique()
        distribution = user_days.value_counts().sort_index()
        total = distribution.sum()
        return (distribution / total).rename("fraction")


class PowerUserAnalyzer:
    """
    Identifies power users and computes the power user curve.
    Power users are those in the top percentile of activity frequency.
    """

    def __init__(self, events_df: pd.DataFrame,
                 user_col: str = "user_id",
                 date_col: str = "event_date",
                 power_user_percentile: float = 0.80):
        self.df = events_df.copy()
        self.user_col = user_col
        self.date_col = date_col
        self.power_user_percentile = power_user_percentile
        self.df[date_col] = pd.to_datetime(self.df[date_col])

    def user_activity_profile(self, window_days: int = 30) -> pd.DataFrame:
        """Compute per-user activity days in the last window_days."""
        max_date = self.df[self.date_col].max()
        cutoff = max_date - pd.Timedelta(days=window_days - 1)
        recent = self.df[self.df[self.date_col] >= cutoff]
        profile = recent.groupby(self.user_col)[self.date_col].nunique().reset_index()
        profile.columns = [self.user_col, "active_days"]
        profile["activity_pct"] = profile["active_days"] / window_days
        return profile.sort_values("active_days", ascending=False).reset_index(drop=True)

    def identify_power_users(self, window_days: int = 30) -> pd.DataFrame:
        """Return users in the top power_user_percentile of activity."""
        profile = self.user_activity_profile(window_days)
        threshold = profile["active_days"].quantile(self.power_user_percentile)
        profile["is_power_user"] = profile["active_days"] >= threshold
        return profile

    def power_user_curve(self, window_days: int = 30) -> pd.DataFrame:
        """
        The power user curve: for each activity level (1..window_days),
        cumulative fraction of users at or above that level.
        """
        profile = self.user_activity_profile(window_days)
        total_users = len(profile)
        records = []
        for n in range(1, window_days + 1):
            users_at_n = int((profile["active_days"] >= n).sum())
            records.append({
                "min_active_days": n,
                "users_at_or_above": users_at_n,
                "fraction": round(users_at_n / max(total_users, 1), 4),
            })
        return pd.DataFrame(records)

    def cohort_retention(self, cohort_window: str = "MS") -> pd.DataFrame:
        """
        Compute cohort-based retention table.
        Each cohort is defined by the first activity month.
        """
        df = self.df.copy()
        first_activity = df.groupby(self.user_col)[self.date_col].min().reset_index()
        first_activity.columns = [self.user_col, "cohort_date"]
        first_activity["cohort"] = first_activity["cohort_date"].dt.to_period(cohort_window)
        df = df.merge(first_activity[[self.user_col, "cohort"]], on=self.user_col)
        df["activity_period"] = df[self.date_col].dt.to_period(cohort_window)
        df["period_offset"] = (df["activity_period"] - df["cohort"]).apply(lambda x: x.n)
        cohort_sizes = first_activity.groupby("cohort")[self.user_col].count().rename("cohort_size")
        retention = df[df["period_offset"] >= 0].groupby(
            ["cohort", "period_offset"]
        )[self.user_col].nunique().reset_index()
        retention.columns = ["cohort", "period_offset", "active_users"]
        retention = retention.merge(cohort_sizes, on="cohort")
        retention["retention_rate"] = (
            retention["active_users"] / retention["cohort_size"]
        ).round(4)
        return retention

    def engagement_segments(self, window_days: int = 30) -> pd.DataFrame:
        """Segment users into power, regular, casual, and dormant."""
        profile = self.user_activity_profile(window_days)
        total = len(profile)
        p80 = profile["active_days"].quantile(0.80)
        p50 = profile["active_days"].quantile(0.50)
        p20 = profile["active_days"].quantile(0.20)

        def segment(days):
            if days >= p80:
                return "power"
            elif days >= p50:
                return "regular"
            elif days >= p20:
                return "casual"
            return "dormant"

        profile["segment"] = profile["active_days"].apply(segment)
        summary = profile.groupby("segment").agg(
            count=("segment", "count"),
            avg_active_days=("active_days", "mean"),
        ).round(2)
        summary["pct"] = (summary["count"] / total * 100).round(1)
        return summary.reset_index()


if __name__ == "__main__":
    np.random.seed(42)
    n_users = 500
    n_events = 15000
    user_ids = [f"U{i:04d}" for i in range(1, n_users + 1)]
    activity_weights = np.random.dirichlet(np.ones(n_users) * 0.5)
    sampled_users = np.random.choice(user_ids, n_events, p=activity_weights)
    dates = pd.date_range("2024-01-01", "2024-04-30", freq="D")
    sampled_dates = np.random.choice(dates, n_events)

    events_df = pd.DataFrame({
        "user_id": sampled_users,
        "event_date": pd.to_datetime(sampled_dates),
    })

    metrics = ActivityMetrics(events_df)
    dau = metrics.dau_series()
    print(f"Average DAU: {dau.mean():.0f}")
    ratio = metrics.dau_mau_ratio()
    print("\nDAU/MAU stickiness:")
    print(ratio.to_string(index=False))

    l30 = metrics.l_n_distribution(window_days=30)
    print(f"\nL30 distribution (days active -> fraction):")
    print(l30.head(10))

    analyzer = PowerUserAnalyzer(events_df, power_user_percentile=0.80)
    segments = analyzer.engagement_segments(window_days=30)
    print("\nEngagement segments:")
    print(segments.to_string(index=False))

    curve = analyzer.power_user_curve(window_days=30)
    print("\nPower user curve (first 10 rows):")
    print(curve.head(10).to_string(index=False))

    power_users = analyzer.identify_power_users(window_days=30)
    n_power = int(power_users["is_power_user"].sum())
    print(f"\nPower users (top 20%): {n_power} / {len(power_users)}")
