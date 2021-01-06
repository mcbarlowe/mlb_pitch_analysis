"""
This will be the location of the main download functions for dowloading and
the data from the statcast API
"""
from typing import Union
import pybaseball


def dl_statcast_date_range(start_date: str, end_date: str) -> "pandas.DataFrame":
    """
    given a start and end date this downloads the statcast pitch data from
    the statcast API

    Inputs:
    start_date - date string in form "YYYY-MM-DD"
    end_date - date string in form "YYYY-MM-DD"

    Outputs:
    df  - pandas dataframe of raw pitch data
    """
    df = pybaseball.statcast(start_date, end_date)
    return df


def dl_statcast_single_game(game_pk: Union[str, int]) -> "pandas.DataFrame":
    """
    returns pitches for a single game from statcast

    Inputs:
    game_pk  - Game id

    Outputs:
    df  - pandas dataframe of raw pitch data from game_pk
    """
    df = pybaseball.statcast_single_game(game_pk)
    return df


if __name__ == "__main__":
    df = dl_statcast_date_range("2017-01-01", "2017-12-31")
    print(df.head())
    df.to_csv("data/2017_data.csv")
