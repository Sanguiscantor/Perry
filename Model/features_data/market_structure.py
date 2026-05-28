import numpy as np
import pandas as pd


MIN_PIVOT_DISTANCE = 8
MAX_ALLOWED_SLOPE = 0.0035
PIVOT_STRENGTH = 0.001
MINIMUM_TRENDLINE_SPAN = 16
MINIMUM_TOUCHES = 2
TOUCH_TOLERANCE = 0.0035
MAX_ANCHOR_LOOKBACK = 180
MIN_LINE_PERSISTENCE = 12
MAX_VERTICAL_MOVE_RATIO = 0.65
MIN_STRUCTURE_SCORE = 0.85
MAX_UNTOUCHED_CANDLES = 48
MAX_CLEAN_BREAK_CANDLES = 2
MIN_ACTIVE_LINE_SCORE = 30.0
REPLACEMENT_SCORE_MULTIPLIER = 1.18
RECENT_INTERACTION_WINDOW = 24


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
    swing_df["swing_high_score"] = 0.0
    swing_df["swing_low_score"] = 0.0

    return swing_df


def project_trendline(first_anchor, second_anchor, position):
    first_position, first_price = first_anchor[:2]
    second_position, second_price = second_anchor[:2]

    if first_position == second_position:
        return np.nan

    slope = (second_price - first_price) / (second_position - first_position)

    return second_price + slope * (position - second_position)


def _price_range(df, start_position, end_position):
    window = df.iloc[start_position:end_position + 1]

    return window["High"].max() - window["Low"].min()


def _local_volatility(df, position, lookback=20):
    start_position = max(position - lookback, 0)
    window = df.iloc[start_position:position + 1]

    if window.empty:
        return 0.0

    true_range = window["High"] - window["Low"]

    return true_range.mean() / max(abs(df.iloc[position]["Close"]), 1e-9)


def _nearby_rejections(df, position, side, lookback=24, tolerance=TOUCH_TOLERANCE):
    start_position = max(position - lookback, 0)
    end_position = min(position + lookback, len(df) - 1)
    pivot_row = df.iloc[position]
    window = df.iloc[start_position:end_position + 1]
    tolerance_value = pivot_row["Close"] * tolerance

    if side == "support":
        nearby = (window["Low"] - pivot_row["Low"]).abs() <= tolerance_value
        rejected = window["Close"] > window["Low"]
    else:
        nearby = (window["High"] - pivot_row["High"]).abs() <= tolerance_value
        rejected = window["Close"] < window["High"]

    return int((nearby & rejected).sum())


def _pivot_importance_score(
    df,
    position,
    side,
    pivot_window,
    previous_anchor=None,
):
    start_position = max(position - pivot_window, 0)
    end_position = min(position + pivot_window, len(df) - 1)
    pivot_row = df.iloc[position]
    local_window = df.iloc[start_position:end_position + 1]

    if side == "support":
        nearby_extreme = local_window.drop(df.index[position], errors="ignore")["Low"].min()
        strength = nearby_extreme - pivot_row["Low"]
    else:
        nearby_extreme = local_window.drop(df.index[position], errors="ignore")["High"].max()
        strength = pivot_row["High"] - nearby_extreme

    if pd.isna(strength):
        return 0.0

    normalized_strength = strength / max(abs(pivot_row["Close"]), 1e-9)
    volatility = max(_local_volatility(df, position), 1e-9)
    volatility_adjusted_strength = normalized_strength / volatility
    rejections = _nearby_rejections(df, position, side)

    if previous_anchor is None:
        separation_score = 1.0
    else:
        separation = position - previous_anchor[0]
        separation_score = min(separation / max(MINIMUM_TRENDLINE_SPAN, 1), 2.5)

    return (
        volatility_adjusted_strength * 0.75
        + min(rejections, 5) * 0.18
        + separation_score * 0.25
    )


def _line_interactions(df, first_anchor, second_anchor, side, touch_tolerance):
    start_position = first_anchor[0]
    end_position = second_anchor[0]
    touches = 0
    rejections = 0
    body_violations = 0

    for position in range(start_position, end_position + 1):
        projected_value = project_trendline(first_anchor, second_anchor, position)
        row = df.iloc[position]
        tolerance = row["Close"] * touch_tolerance

        if side == "support":
            touched = abs(row["Low"] - projected_value) <= tolerance
            rejected = touched and row["Close"] > projected_value
            violated = row["Close"] < projected_value - tolerance
        else:
            touched = abs(row["High"] - projected_value) <= tolerance
            rejected = touched and row["Close"] < projected_value
            violated = row["Close"] > projected_value + tolerance

        if touched:
            touches += 1

        if rejected:
            rejections += 1

        if violated:
            body_violations += 1

    return {
        "touches": touches,
        "rejections": rejections,
        "body_violations": body_violations,
    }


def _trendline_quality(
    df,
    first_anchor,
    second_anchor,
    side,
    max_allowed_slope,
    minimum_trendline_span,
    touch_tolerance,
    latest_position,
):
    span = second_anchor[0] - first_anchor[0]

    if span < minimum_trendline_span:
        return None

    persistence = latest_position - second_anchor[0]

    if persistence < MIN_LINE_PERSISTENCE:
        return None

    price_span = _price_range(df, first_anchor[0], second_anchor[0])

    if pd.isna(price_span) or price_span <= 0:
        return None

    raw_slope = (second_anchor[1] - first_anchor[1]) / span
    normalized_slope = abs(raw_slope) / max(abs(second_anchor[1]), 1e-9)

    if normalized_slope > max_allowed_slope:
        return None

    if abs(raw_slope) > price_span / max(span, 1):
        return None

    vertical_move_ratio = abs(second_anchor[1] - first_anchor[1]) / price_span

    if vertical_move_ratio > MAX_VERTICAL_MOVE_RATIO:
        return None

    interactions = _line_interactions(
        df,
        first_anchor,
        second_anchor,
        side,
        touch_tolerance,
    )
    touches = interactions["touches"]
    rejections = interactions["rejections"]
    body_violations = interactions["body_violations"]

    if touches < MINIMUM_TOUCHES:
        return None

    if rejections < MINIMUM_TOUCHES:
        return None

    violation_ratio = body_violations / max(span, 1)

    if violation_ratio > 0.25:
        return None

    first_score = first_anchor[2]
    second_score = second_anchor[2]
    structural_score = first_score + second_score

    if structural_score < MIN_STRUCTURE_SCORE:
        return None

    slope_direction_bonus = 0.0

    if side == "support" and raw_slope > 0:
        slope_direction_bonus = 35.0
    elif side == "resistance" and raw_slope < 0:
        slope_direction_bonus = 35.0

    persistence_score = min(persistence, 120) * 0.6
    span_score = min(span, 220) * 2.5
    touch_score = touches * 22.0 + rejections * 18.0
    slope_penalty = normalized_slope * 14000.0 + vertical_move_ratio * 45.0
    violation_penalty = body_violations * 8.0

    return (
        span_score
        + touch_score
        + persistence_score
        + structural_score * 30.0
        + slope_direction_bonus
        - slope_penalty
        - violation_penalty
    )


def _best_valid_line(
    df,
    anchors,
    invalidated_pairs,
    side,
    max_allowed_slope,
    minimum_trendline_span,
    touch_tolerance,
    max_anchor_lookback,
    score_cache=None,
):
    if len(anchors) < 2:
        return None

    latest_position = len(df) - 1
    candidate_anchors = [
        anchor
        for anchor in anchors
        if anchors[-1][0] - anchor[0] <= max_anchor_lookback
    ]
    best_line = None
    best_score = None

    for second_index in range(len(candidate_anchors) - 1, 0, -1):
        for first_index in range(second_index - 1, -1, -1):
            first_anchor = candidate_anchors[first_index]
            second_anchor = candidate_anchors[second_index]
            anchor_pair = (first_anchor[0], second_anchor[0])

            if anchor_pair in invalidated_pairs:
                continue

            cache_key = (side, anchor_pair)

            if score_cache is not None and cache_key in score_cache:
                score = score_cache[cache_key]
            else:
                score = _trendline_quality(
                    df,
                    first_anchor,
                    second_anchor,
                    side,
                    max_allowed_slope,
                    minimum_trendline_span,
                    touch_tolerance,
                    latest_position,
                )

                if score_cache is not None:
                    score_cache[cache_key] = score

            if score is None:
                continue

            if best_score is None or score > best_score:
                best_score = score
                best_line = first_anchor, second_anchor

    if best_line is None:
        return None

    return {
        "line": best_line,
        "score": best_score,
    }


def _line_key(line):
    if line is None:
        return None

    return line[0][0], line[1][0]


def _make_line_state(line, base_score, created_position):
    return {
        "line": line,
        "base_score": base_score,
        "score": base_score,
        "touches": 0,
        "rejections": 0,
        "clean_break_count": 0,
        "created_position": created_position,
        "last_interaction_position": created_position,
    }


def _row_line_interaction(row, line_value, side, touch_tolerance):
    tolerance = row["Close"] * touch_tolerance

    if side == "support":
        touched = abs(row["Low"] - line_value) <= tolerance
        rejected = touched and row["Close"] > line_value
        clean_break = row["Close"] < line_value - tolerance
    else:
        touched = abs(row["High"] - line_value) <= tolerance
        rejected = touched and row["Close"] < line_value
        clean_break = row["Close"] > line_value + tolerance

    return touched, rejected, clean_break


def _active_line_score(state, position):
    age = position - state["created_position"]
    untouched_age = position - state["last_interaction_position"]
    recency_bonus = max(RECENT_INTERACTION_WINDOW - untouched_age, 0) * 2.0
    persistence_bonus = min(age, 160) * 0.35
    interaction_bonus = state["touches"] * 18.0 + state["rejections"] * 24.0
    decay_penalty = max(untouched_age - RECENT_INTERACTION_WINDOW, 0) * 4.0
    break_penalty = state["clean_break_count"] * 55.0

    return (
        state["base_score"]
        + interaction_bonus
        + persistence_bonus
        + recency_bonus
        - decay_penalty
        - break_penalty
    )


def _update_line_state(df, state, position, side, touch_tolerance):
    line = state["line"]
    line_value = project_trendline(line[0], line[1], position)
    row = df.iloc[position]
    touched, rejected, clean_break = _row_line_interaction(
        row,
        line_value,
        side,
        touch_tolerance,
    )

    if touched:
        state["touches"] += 1
        state["last_interaction_position"] = position

    if rejected:
        state["rejections"] += 1

    if clean_break:
        state["clean_break_count"] += 1
    else:
        state["clean_break_count"] = 0

    state["score"] = _active_line_score(state, position)
    untouched_age = position - state["last_interaction_position"]

    expired = (
        state["clean_break_count"] >= MAX_CLEAN_BREAK_CANDLES
        or untouched_age > MAX_UNTOUCHED_CANDLES
        or state["score"] < MIN_ACTIVE_LINE_SCORE
    )

    return line_value, expired


def _confirmed_pivot(
    df,
    position,
    pivot_window,
    swing_column,
    price_column,
    side,
    pivot_strength,
    min_pivot_distance,
    existing_anchors,
):
    pivot_position = position - pivot_window

    if pivot_position < 0:
        return None

    pivot_row = df.iloc[pivot_position]

    if not bool(pivot_row[swing_column]):
        return None

    if existing_anchors and pivot_position - existing_anchors[-1][0] < min_pivot_distance:
        return None

    pivot_score = _pivot_importance_score(
        df,
        pivot_position,
        side,
        pivot_window,
        previous_anchor=existing_anchors[-1] if existing_anchors else None,
    )

    if pivot_score < pivot_strength:
        return None

    return pivot_position, float(pivot_row[price_column]), pivot_score


def build_anchored_trendlines(
    df,
    pivot_window=3,
    confirmation_candles=2,
    min_pivot_distance=MIN_PIVOT_DISTANCE,
    max_allowed_slope=MAX_ALLOWED_SLOPE,
    pivot_strength=PIVOT_STRENGTH,
    minimum_trendline_span=MINIMUM_TRENDLINE_SPAN,
    touch_tolerance=TOUCH_TOLERANCE,
    max_anchor_lookback=MAX_ANCHOR_LOOKBACK,
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
    candidate_score_cache = {}

    active_support_state = None
    active_resistance_state = None

    for position in range(len(structure_df)):
        confirmed_low = _confirmed_pivot(
            structure_df,
            position,
            pivot_window,
            "swing_low",
            "Low",
            "support",
            pivot_strength,
            min_pivot_distance,
            confirmed_lows,
        )
        confirmed_high = _confirmed_pivot(
            structure_df,
            position,
            pivot_window,
            "swing_high",
            "High",
            "resistance",
            pivot_strength,
            min_pivot_distance,
            confirmed_highs,
        )

        if confirmed_low is not None:
            confirmed_lows.append(confirmed_low)
            structure_df.iloc[
                confirmed_low[0],
                structure_df.columns.get_loc("swing_low_score"),
            ] = confirmed_low[2]

        if confirmed_high is not None:
            confirmed_highs.append(confirmed_high)
            structure_df.iloc[
                confirmed_high[0],
                structure_df.columns.get_loc("swing_high_score"),
            ] = confirmed_high[2]

        support_candidate = _best_valid_line(
            structure_df,
            confirmed_lows,
            invalidated_support_pairs,
            "support",
            max_allowed_slope,
                minimum_trendline_span,
                touch_tolerance,
                max_anchor_lookback,
                candidate_score_cache,
            )
        resistance_candidate = _best_valid_line(
            structure_df,
            confirmed_highs,
            invalidated_resistance_pairs,
            "resistance",
            max_allowed_slope,
                minimum_trendline_span,
                touch_tolerance,
                max_anchor_lookback,
                candidate_score_cache,
            )

        if support_candidate is not None:
            candidate_key = _line_key(support_candidate["line"])
            active_key = _line_key(active_support_state["line"]) if active_support_state else None

            if active_support_state is None or (
                candidate_key != active_key
                and support_candidate["score"]
                > active_support_state["score"] * REPLACEMENT_SCORE_MULTIPLIER
            ):
                active_support_state = _make_line_state(
                    support_candidate["line"],
                    support_candidate["score"],
                    position,
                )

        if resistance_candidate is not None:
            candidate_key = _line_key(resistance_candidate["line"])
            active_key = (
                _line_key(active_resistance_state["line"])
                if active_resistance_state else None
            )

            if active_resistance_state is None or (
                candidate_key != active_key
                and resistance_candidate["score"]
                > active_resistance_state["score"] * REPLACEMENT_SCORE_MULTIPLIER
            ):
                active_resistance_state = _make_line_state(
                    resistance_candidate["line"],
                    resistance_candidate["score"],
                    position,
                )

        if active_support_state is not None:
            support_value, support_expired = _update_line_state(
                structure_df,
                active_support_state,
                position,
                "support",
                touch_tolerance,
            )
            support_values[position] = support_value
            active_support_line = active_support_state["line"]
            support_anchor_1_positions[position] = active_support_line[0][0]
            support_anchor_1_prices[position] = active_support_line[0][1]
            support_anchor_2_positions[position] = active_support_line[1][0]
            support_anchor_2_prices[position] = active_support_line[1][1]

            if support_expired:
                invalidated_support_pairs.add(
                    (
                        active_support_line[0][0],
                        active_support_line[1][0],
                    )
                )
                active_support_state = None

        if active_resistance_state is not None:
            resistance_value, resistance_expired = _update_line_state(
                structure_df,
                active_resistance_state,
                position,
                "resistance",
                touch_tolerance,
            )
            resistance_values[position] = resistance_value
            active_resistance_line = active_resistance_state["line"]
            resistance_anchor_1_positions[position] = active_resistance_line[0][0]
            resistance_anchor_1_prices[position] = active_resistance_line[0][1]
            resistance_anchor_2_positions[position] = active_resistance_line[1][0]
            resistance_anchor_2_prices[position] = active_resistance_line[1][1]

            if resistance_expired:
                invalidated_resistance_pairs.add(
                    (
                        active_resistance_line[0][0],
                        active_resistance_line[1][0],
                    )
                )
                active_resistance_state = None

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
    min_pivot_distance=MIN_PIVOT_DISTANCE,
    max_allowed_slope=MAX_ALLOWED_SLOPE,
    pivot_strength=PIVOT_STRENGTH,
    minimum_trendline_span=MINIMUM_TRENDLINE_SPAN,
):
    structure_df = build_anchored_trendlines(
        df,
        pivot_window=pivot_window,
        confirmation_candles=confirmation_candles,
        min_pivot_distance=min_pivot_distance,
        max_allowed_slope=max_allowed_slope,
        pivot_strength=pivot_strength,
        minimum_trendline_span=minimum_trendline_span,
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
