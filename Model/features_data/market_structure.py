import numpy as np
import pandas as pd


def detect_swing_points(df, pivot_window=3):
    swing_df = df.copy()

    swing_high = pd.Series(True, index=swing_df.index)
    swing_low = pd.Series(True, index=swing_df.index)

    for offset in range(1, pivot_window + 1):
        swing_high &= swing_df["High"] > swing_df["High"].shift(offset)
        swing_high &= swing_df["High"] > swing_df["High"].shift(-offset)

        swing_low &= swing_df["Low"] < swing_df["Low"].shift(offset)
        swing_low &= swing_df["Low"] < swing_df["Low"].shift(-offset)

    swing_df["swing_high"] = swing_high.fillna(False).astype(int)
    swing_df["swing_low"] = swing_low.fillna(False).astype(int)

    return swing_df


def project_trendline(first_anchor, second_anchor, position):
    first_position, first_price = first_anchor
    second_position, second_price = second_anchor

    if first_position == second_position:
        return np.nan

    slope = (second_price - first_price) / (second_position - first_position)

    return second_price + slope * (position - second_position)


def _latest_valid_line(anchors, invalidated_pairs):
    if len(anchors) < 2:
        return None

    for anchor_index in range(len(anchors) - 1, 0, -1):
        first_anchor = anchors[anchor_index - 1]
        second_anchor = anchors[anchor_index]
        anchor_pair = (first_anchor[0], second_anchor[0])

        if anchor_pair not in invalidated_pairs:
            return first_anchor, second_anchor

    return None


def _confirmed_pivot(df, position, pivot_window, swing_column, price_column):
    pivot_position = position - pivot_window

    if pivot_position < 0:
        return None

    pivot_row = df.iloc[pivot_position]

    if not bool(pivot_row[swing_column]):
        return None

    return pivot_position, float(pivot_row[price_column])


def build_anchored_trendlines(
    df,
    pivot_window=3,
    confirmation_candles=2,
):
    structure_df = detect_swing_points(df, pivot_window=pivot_window).copy()

    support_values = np.full(len(structure_df), np.nan)
    resistance_values = np.full(len(structure_df), np.nan)
    support_anchor_1_positions = np.full(len(structure_df), np.nan)
    support_anchor_1_prices = np.full(len(structure_df), np.nan)
    support_anchor_2_positions = np.full(len(structure_df), np.nan)
    support_anchor_2_prices = np.full(len(structure_df), np.nan)
    resistance_anchor_1_positions = np.full(len(structure_df), np.nan)
    resistance_anchor_1_prices = np.full(len(structure_df), np.nan)
    resistance_anchor_2_positions = np.full(len(structure_df), np.nan)
    resistance_anchor_2_prices = np.full(len(structure_df), np.nan)

    confirmed_lows = []
    confirmed_highs = []
    invalidated_support_pairs = set()
    invalidated_resistance_pairs = set()

    active_support_line = None
    active_resistance_line = None
    support_break_count = 0
    resistance_break_count = 0

    for position in range(len(structure_df)):
        confirmed_low = _confirmed_pivot(
            structure_df,
            position,
            pivot_window,
            "swing_low",
            "Low",
        )
        confirmed_high = _confirmed_pivot(
            structure_df,
            position,
            pivot_window,
            "swing_high",
            "High",
        )

        if confirmed_low is not None:
            confirmed_lows.append(confirmed_low)

        if confirmed_high is not None:
            confirmed_highs.append(confirmed_high)

        if active_support_line is None:
            active_support_line = _latest_valid_line(
                confirmed_lows,
                invalidated_support_pairs,
            )
            support_break_count = 0

        if active_resistance_line is None:
            active_resistance_line = _latest_valid_line(
                confirmed_highs,
                invalidated_resistance_pairs,
            )
            resistance_break_count = 0

        close_price = structure_df["Close"].iloc[position]

        if active_support_line is not None:
            support_value = project_trendline(
                active_support_line[0],
                active_support_line[1],
                position,
            )
            support_values[position] = support_value
            support_anchor_1_positions[position] = active_support_line[0][0]
            support_anchor_1_prices[position] = active_support_line[0][1]
            support_anchor_2_positions[position] = active_support_line[1][0]
            support_anchor_2_prices[position] = active_support_line[1][1]

            if close_price < support_value:
                support_break_count += 1
            else:
                support_break_count = 0

            if support_break_count >= confirmation_candles:
                invalidated_support_pairs.add(
                    (
                        active_support_line[0][0],
                        active_support_line[1][0],
                    )
                )
                active_support_line = None
                support_break_count = 0

        if active_resistance_line is not None:
            resistance_value = project_trendline(
                active_resistance_line[0],
                active_resistance_line[1],
                position,
            )
            resistance_values[position] = resistance_value
            resistance_anchor_1_positions[position] = active_resistance_line[0][0]
            resistance_anchor_1_prices[position] = active_resistance_line[0][1]
            resistance_anchor_2_positions[position] = active_resistance_line[1][0]
            resistance_anchor_2_prices[position] = active_resistance_line[1][1]

            if close_price > resistance_value:
                resistance_break_count += 1
            else:
                resistance_break_count = 0

            if resistance_break_count >= confirmation_candles:
                invalidated_resistance_pairs.add(
                    (
                        active_resistance_line[0][0],
                        active_resistance_line[1][0],
                    )
                )
                active_resistance_line = None
                resistance_break_count = 0

    structure_df["anchored_support"] = support_values
    structure_df["anchored_resistance"] = resistance_values
    structure_df["support_anchor_1_position"] = support_anchor_1_positions
    structure_df["support_anchor_1_price"] = support_anchor_1_prices
    structure_df["support_anchor_2_position"] = support_anchor_2_positions
    structure_df["support_anchor_2_price"] = support_anchor_2_prices
    structure_df["resistance_anchor_1_position"] = resistance_anchor_1_positions
    structure_df["resistance_anchor_1_price"] = resistance_anchor_1_prices
    structure_df["resistance_anchor_2_position"] = resistance_anchor_2_positions
    structure_df["resistance_anchor_2_price"] = resistance_anchor_2_prices

    return structure_df


def add_anchored_structure_features(
    df,
    pivot_window=3,
    confirmation_candles=2,
):
    structure_df = build_anchored_trendlines(
        df,
        pivot_window=pivot_window,
        confirmation_candles=confirmation_candles,
    )

    structure_df["distance_to_anchored_support"] = (
        (structure_df["Close"] - structure_df["anchored_support"])
        / structure_df["Close"]
    )
    structure_df["distance_to_anchored_resistance"] = (
        (structure_df["anchored_resistance"] - structure_df["Close"])
        / structure_df["Close"]
    )

    above_resistance = structure_df["Close"] > structure_df["anchored_resistance"]
    below_support = structure_df["Close"] < structure_df["anchored_support"]

    structure_df["anchored_breakout"] = (
        above_resistance
        .rolling(window=confirmation_candles, min_periods=confirmation_candles)
        .sum()
        .eq(confirmation_candles)
        .astype(int)
    )
    structure_df["anchored_breakdown"] = (
        below_support
        .rolling(window=confirmation_candles, min_periods=confirmation_candles)
        .sum()
        .eq(confirmation_candles)
        .astype(int)
    )

    return structure_df
