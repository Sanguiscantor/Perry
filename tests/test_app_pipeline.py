from datetime import datetime, timezone

import pandas as pd

import app


def test_should_regenerate_feature_dataset_when_raw_is_newer(tmp_path):
    raw = pd.DataFrame({"Datetime": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]})
    feature_path = tmp_path / "features.csv"
    pd.DataFrame({"Datetime": [pd.Timestamp("2024-01-01")] }).to_csv(feature_path, index=False)

    assert app._should_regenerate_feature_dataset(feature_path, raw) is True


def test_should_keep_existing_feature_dataset_when_already_fresh(tmp_path):
    raw = pd.DataFrame({"Datetime": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]})
    feature_path = tmp_path / "features.csv"
    pd.DataFrame({"Datetime": [pd.Timestamp("2024-01-02")] }).to_csv(feature_path, index=False)

    assert app._should_regenerate_feature_dataset(feature_path, raw) is False


def test_should_flag_stale_market_timestamp_when_it_lags_the_clock():
    latest = datetime(2026, 7, 4, 8, 0, tzinfo=timezone.utc)
    current = datetime(2026, 7, 4, 8, 30, tzinfo=timezone.utc)

    stale, detail = app._describe_timestamp_freshness(latest, current)

    assert stale is True
    assert "behind" in detail.lower()


def test_should_not_flag_recent_market_timestamp():
    latest = datetime(2026, 7, 4, 8, 30, tzinfo=timezone.utc)
    current = datetime(2026, 7, 4, 8, 35, tzinfo=timezone.utc)

    stale, detail = app._describe_timestamp_freshness(latest, current)

    assert stale is False
    assert detail == "fresh"
