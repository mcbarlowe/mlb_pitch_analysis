import os
import pickle
import pandas as pd
import lightgbm as lgb
import numpy as np


def munge_data_for_models(df):
    """
    this function was written by Nick Wan (@nickwan on twitter)
    if it breaks bug him about it cause I'm not touching this.
    """
    df["runs_scored"] = df["des"].str.count("score")
    df.loc[df["events"] == "home_run", "runs_scored"] = df.loc[
        df["events"] == "home_run", "runs_scored"
    ].add(1)
    pitch_strikes = ["swinging_strike", "called_strike", "swinging_strike_blocked"]
    df["qualified_strikes"] = 0
    df.loc[df["description"].isin(pitch_strikes), "qualified_strikes"] = 1
    df = df.merge(pd.get_dummies(df["events"]), left_index=True, right_index=True)
    for col in ["on_1b", "on_2b", "on_3b"]:
        df[f"is_{col}"] = df[col].fillna(0).astype(bool).astype(int)
    df = df.merge(
        df.loc[:, ["game_pk", "inning", "inning_topbot", "bat_score"]]
        .groupby(["game_pk", "inning", "inning_topbot"], as_index=False)
        .max()
        .rename(columns={"bat_score": "runs_scored_inning_max"})
    )
    df["runs_scored_inning_delta"] = df["runs_scored_inning_max"].sub(df["bat_score"])
    df["is_lefty"] = df["p_throws"].replace(["R", "L"], [0, 1])

    re24 = (
        df.loc[
            (df["balls"] < 4),
            [
                "outs_when_up",
                "is_on_1b",
                "is_on_2b",
                "is_on_3b",
                "runs_scored_inning_delta",
            ],
        ]
        .groupby(["outs_when_up", "is_on_1b", "is_on_2b", "is_on_3b"], as_index=False)
        .mean()
        .rename(columns={"runs_scored_inning_delta": "re24"})
    )

    re288 = (
        df.loc[
            (df["balls"] < 4),
            [
                "balls",
                "strikes",
                "outs_when_up",
                "is_on_1b",
                "is_on_2b",
                "is_on_3b",
                "runs_scored_inning_delta",
            ],
        ]
        .groupby(
            ["balls", "strikes", "outs_when_up", "is_on_1b", "is_on_2b", "is_on_3b"],
            as_index=False,
        )
        .mean()
        .rename(columns={"runs_scored_inning_delta": "re288"})
    )

    _re = re24.merge(re288)
    df = df.merge(_re)
    _df = (
        df.loc[
            :,
            [
                "game_pk",
                "inning",
                "inning_topbot",
                "at_bat_number",
                "pitch_number",
                "balls",
                "strikes",
                "outs_when_up",
                "is_on_1b",
                "is_on_2b",
                "is_on_3b",
                "bat_score",
                "re288",
                "runs_scored_inning_delta",
            ],
        ]
        .dropna()
        .sort_values(
            ["game_pk", "inning", "inning_topbot", "at_bat_number", "pitch_number"],
            ascending=[1, 1, 0, 1, 1],
        )
        .reset_index(drop=True)
        .copy()
    )
    _df["game_pk_post"] = _df["game_pk"].shift(-1)
    _df["re288_post"] = _df["re288"].shift(-1)
    _df["inning"] = _df["inning"].astype(int)
    _df["inning_post"] = _df["inning"].shift(-1)
    _df["inning_topbot_post"] = _df["inning_topbot"].shift(-1)
    _df["bat_score_post"] = _df["bat_score"].shift(-1)
    _df.loc[
        (_df["game_pk"] == 565677)
        & (_df["inning"] == 7)
        & (_df["at_bat_number"] == 63)
        & (_df["pitch_number"] == 4),
        "bat_score_post",
    ] = 1
    _df["bat_score_delta"] = _df["bat_score_post"].sub(_df["bat_score"])
    _df.loc[_df["game_pk"] != _df["game_pk_post"], "bat_score_delta"] = 0
    _df.loc[_df["inning_topbot"] != _df["inning_topbot_post"], "bat_score_delta"] = 0
    _df.loc[_df["inning_post"] != _df["inning"], "re288_post"] = 0
    _df["re288_delta"] = _df["re288_post"].sub(_df["re288"]).add(_df["bat_score_delta"])
    df_with_re = df.merge(_df)

    for col in ["plate_x", "plate_z"]:
        df_with_re[f"{col}_round"] = df_with_re[col].mul(4).round(0).div(4)

    _a = (
        df_with_re.loc[:, ["game_pk", "inning", "inning_topbot", "re288_delta"]]
        .groupby(["game_pk", "inning", "inning_topbot"], as_index=False)
        .count()
        .rename(columns={"re288_delta": "num_pitches"})
    )
    _b = (
        df_with_re.loc[:, ["game_pk", "inning", "inning_topbot", "re288_delta"]]
        .groupby(["game_pk", "inning", "inning_topbot"], as_index=False)
        .sum()
    )
    _c = (
        df_with_re.loc[
            :, ["game_pk", "inning", "inning_topbot", "runs_scored_inning_delta"]
        ]
        .groupby(["game_pk", "inning", "inning_topbot"], as_index=False)
        .max()
    )
    df_adj = _a.merge(_b).merge(_c)
    df_adj["inning_adj"] = (
        df_adj["runs_scored_inning_delta"].sub(df_adj["re288_delta"])
    ).div(df_adj["num_pitches"])
    df_adj = df_adj.drop(
        ["num_pitches", "re288_delta", "runs_scored_inning_delta"], axis=1
    )
    df_with_re = df_with_re.merge(df_adj)
    df_with_re["pitch_level_runs"] = df_with_re["re288_delta"].add(
        df_with_re["inning_adj"]
    )

    plot_data = df_with_re.copy()
    plot_data["plate_x"] = plot_data["plate_x"].mul(-1)
    qual_locs = (
        plot_data.loc[:, ["plate_x_round", "plate_z_round", "pitch_level_runs"]]
        .groupby(["plate_x_round", "plate_z_round"], as_index=False)
        .count()
    )
    qual_locs = qual_locs.loc[
        qual_locs["pitch_level_runs"] > qual_locs["pitch_level_runs"].median(),
        ["plate_x_round", "plate_z_round"],
    ]
    loc_vals = (
        plot_data.loc[:, ["plate_x_round", "plate_z_round", "pitch_level_runs"]]
        .groupby(["plate_x_round", "plate_z_round"], as_index=False)
        .mean()
    )
    loc_vals = loc_vals.merge(qual_locs.loc[:, ["plate_x_round", "plate_z_round"]])
    df_with_re = df_with_re.merge(
        loc_vals.rename(columns={"pitch_level_runs": "loc_vals"})
    )
    df_with_re["is_swinging_strike"] = 0
    df_with_re.loc[
        df_with_re["description"].isin(["swinging_strike", "swinging_strike_blocked"]),
        "is_swinging_strike",
    ] = 1
    return df_with_re, loc_vals


def csv_to_sql(
    file_location: str, connection_str: str, schema: str, table_name: str
) -> None:
    df = pd.read_csv(file_location)
    df = df.rename(columns={"pitcher.1": "pitcher_1", "fielder_2.1": "fielder_2_1"})
    # going by game_date in case I pass a really large data frame to it I
    # don't want the process to error out because of size
    for date in df["game_date"].unique():
        print(f"Inserting date: {date}")
        df1 = df[df["game_date"] == date]
        df1.to_sql(
            table_name,
            os.environ[connection_str],
            schema=schema,
            if_exists="append",
            method="multi",
            index=False,
        )


def df_to_sql(
    df: pd.DataFrame, connection_str: str, schema: str, table_name: str
) -> None:
    df = pd.read_csv(file_location)
    df = df.rename(columns={"pitcher.1": "pitcher_1", "fielder_2.1": "fielder_2_1"})
    # going by game_date in case I pass a really large data frame to it I
    # don't want the process to error out because of size
    for date in df["game_date"].unique():
        print(f"Inserting date: {date}")
        df1 = df[df["game_date"] == date]
        df1.to_sql(
            table_name,
            os.environ[connection_str],
            schema=schema,
            if_exists="append",
            method="multi",
            index=False,
        )


def predict_pitch_location(df: pd.DataFrame) -> pd.DataFrame:
    model_feats = [
        "is_lefty",
        "release_pos_x",
        "release_pos_z",
        "release_extension",
        "vx0",
        "vy0",
        "vz0",
        "ax",
        "ay",
        "az",
        "release_spin_rate",
    ]

    model_feats_pitch_loc = model_feats + [
        "pfx_x_pred",
        "pfx_z_pred",
        "release_speed_pred",
    ]

    # loading_models
    with open("models/20201212132459_pitch_release_models.model", "rb") as prm:
        pitch_release_models = pickle.load(prm)

    with open("models/20201212132503_pitch_location_models.model", "rb") as plm:
        pitch_location_models = pickle.load(plm)

    swstr_model = lgb.Booster(model_file="models/20201212135304_swstr_model.model")

    with open("models/20201212134057_pitch_type_model.model", "rb") as ptm:
        pitch_type_model = pickle.load(ptm)

    pt_lookup_df = pd.read_csv("models/20201212133351_pitch_type_model_lkup.csv")

    # data munging
    df_with_re, loc_vals = munge_data_for_models(df)

    # predicting pitch release models
    for model_name, model in pitch_release_models.items():
        model_data = df_with_re.loc[:, model_feats].dropna()
        model = pitch_release_models[model_name]
        x = pd.Series(model.predict(model_data), index=model_data.index)
        df_with_re[model_name.replace("_model", "_pred")] = x

    # predicting pitch location models
    for model_name, model in pitch_location_models.items():
        model_data = df_with_re.loc[:, model_feats_pitch_loc].dropna()
        model = pitch_location_models[model_name]
        x = pd.Series(model.predict(model_data), index=model_data.index)
        df_with_re[model_name.replace("_model", "_pred")] = x

    for col in ["plate_x_pred", "plate_z_pred"]:
        df_with_re[f"{col}_round"] = df_with_re[col].mul(4).round(0).div(4)

    df_with_re = df_with_re.merge(
        loc_vals.rename(
            columns={
                "plate_x_round": "plate_x_pred_round",
                "plate_z_round": "plate_z_pred_round",
                "pitch_level_runs": "vacuum_val",
            }
        )
    )
    # swing strike model
    swstr_model_feats = model_feats_pitch_loc + [
        "plate_x_pred",
        "plate_z_pred",
        "vacuum_val",
    ]

    df_with_re["is_swinging_strike_pred"] = pd.Series(
        swstr_model.predict(df_with_re.loc[:, swstr_model_feats]),
        index=df_with_re.index,
    )
    df_with_re["pitch_type_code"] = pd.Series(
        pitch_type_model.predict(df_with_re.loc[:, model_feats_pitch_loc]),
        index=df_with_re.index,
    )
    pt_map = (
        pt_lookup_df.loc[:, ["pitch_type", "pitch_type_code"]]
        .rename(columns={"pitch_type": "pitch_type_rf_pred"})
        .set_index("pitch_type_code")
        .to_dict()["pitch_type_rf_pred"]
    )
    df_with_re["pitch_type_rf_pred"] = df_with_re["pitch_type_code"].map(pt_map)
    # more Nick code hopefully don't have to debug it need to find out what its
    # doing
    _df_with_re = df_with_re.loc[
        :, ["is_lefty", "pitch_type", "plate_x", "plate_z"]
    ].copy()
    _df_with_re.loc[_df_with_re["is_lefty"] == 1, "plate_x"] = _df_with_re.loc[
        _df_with_re["is_lefty"] == 1, "plate_x"
    ].mul(-1)
    q25 = (
        _df_with_re.loc[:, ["pitch_type", "plate_x", "plate_z"]]
        .groupby("pitch_type", as_index=False)
        .quantile(0.25)
    )
    q75 = (
        _df_with_re.loc[:, ["pitch_type", "plate_x", "plate_z"]]
        .groupby("pitch_type", as_index=False)
        .quantile(0.75)
    )
    pitch_type_quantiles = q25.merge(q75, on="pitch_type", suffixes=("_25", "_75"))
    pitch_type_quantiles["d"] = np.sqrt(
        (pitch_type_quantiles["plate_x_75"] - pitch_type_quantiles["plate_x_25"]) ** 2
        + (pitch_type_quantiles["plate_z_75"] - pitch_type_quantiles["plate_z_25"]) ** 2
    )
    pitch_type_quantiles["r"] = pitch_type_quantiles["d"].div(2)
    df_with_re["r"] = df_with_re["pitch_type_rf_pred"].map(
        pitch_type_quantiles.loc[:, ["pitch_type", "r"]]
        .set_index("pitch_type")
        .to_dict()["r"]
    )
    swstr_val = df_with_re.loc[
        df_with_re["is_swinging_strike"] == 1, "pitch_level_runs"
    ].mean()
    df_with_re["swstr_val"] = df_with_re["is_swinging_strike_pred"].mul(swstr_val)
    df_with_re["actual_stuff_value"] = df_with_re["vacuum_val"].add(
        df_with_re["swstr_val"]
    )

    return df_with_re


if __name__ == "__main__":
    df = pd.read_csv("data/2018_2019_data.csv")
    new_df = predict_pitch_location(df)
    print(new_df.columns)
    print(new_df.head())
    new_df.to_csv("data/20182019_data_w_pred.csv")
