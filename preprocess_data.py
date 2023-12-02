#!/usr/bin/env python3

from datetime import date
from enum import Enum
import itertools
import os
from pathlib import Path
import pickle

import networkx as nx
import pandas as pd

# Matching dictionary for team names between events and tracking data
TEAM_NAMES = {
    "Canada": "Olympic (Women) - Canada",
    "USA": "Olympic (Women) - United States",
    "Finland": "Olympic (Women) - Finland",
    "ROC": "Olympic (Women) - Olympic Athletes from Russia",
    "Switzerland": "Olympic (Women) - Switzerland",
}
TRACKING_DIR = "external/Big-Data-Cup-2022/data"
PBP_DIR = "external/Big-Data-Cup-2022/data"
PLAY_BY_PLAY_DATA_FILE = "pxp_womens_oly_2022_v2.csv"
POWER_PLAY_INFO_FILE = "pp_info.csv"
MAX_PENALTY_NUMBER = 8  # Determined from file.
GAMES = [
    "2022-02-08 Canada at USA",
    "2022-02-08 ROC at Finland",
    "2022-02-12 Switzerland at ROC",
    "2022-02-14 Switzerland at Canada",
    "2022-02-14 USA at Finland",
    "2022-02-16 Switzerland at Finland",
]


def main():
    ################################################################################
    # Actual Main code                                                             #
    ################################################################################

    # Count the various combinations encountered below.
    empty_networks = 0
    disconnected_networks = 0
    disconnected_networks_pp = 0
    connected_networks = 0
    pp_count = 0
    total_posibilities = 0

    dir = Path("./data/networks")
    dir.mkdir(parents=True, exist_ok=True)  # Create directory if it does not exist

    # Generate some combinations of potential networks.
    # Some of them are empty,
    # e.g. Venue.HOME, Situation.POWER_PLAY, PowerPlay(1) but Venue.HOME is PENALTY_KILL in that Power Play.
    for game, venue, (situation, pp) in itertools.product(
        GAMES,
        [Venue.HOME, Venue.AWAY],
        [(Situation.REGULAR, None)]
        + [(Situation.POWER_PLAY, PowerPlay(penalty_no)) for penalty_no in range(1, MAX_PENALTY_NUMBER + 1)],
    ):
        res = build_networks(Game(game), venue=venue, situation=situation, pp=pp)
        if res is None:
            # Happens if there is for example no 7th PP in a game.
            continue

        total_posibilities += 1

        if pp is not None:
            pp_count += 1

        if res["position_pass_network"].number_of_nodes() == 0:
            empty_networks += 1
            continue

        if res["position_pass_network"].number_of_nodes() < 5:
            disconnected_networks += 1
            if pp is not None:
                disconnected_networks_pp += 1
            continue

        connected_networks += 1

        # write to disk
        file = Path(f"{game}_{venue.value}{f'_pp{pp.penalty_no}' if pp is not None else ''}.pickle")
        with open(dir / file, "wb") as f:
            pickle.dump(res, f)

    print(f"Connected: {connected_networks} / {total_posibilities}")
    print(f"Empty Networks: {empty_networks} / {total_posibilities}")
    print(f"Disconnected Networks: {disconnected_networks} / {total_posibilities}")
    print(f"Empty Networks out of PP situations: {empty_networks} / {pp_count}")
    print(f"Disconnected Networks out of PP situations: {disconnected_networks_pp} / {pp_count}")


def directed_to_undirected(digraph):
    # From: https://stackoverflow.com/questions/56169907/networkx-change-weighted-directed-graph-to-undirected
    # Yes, I couldn't be bothered to write this simple loop myself
    graph = digraph.to_undirected()
    for node in digraph:
        for ngbr in nx.neighbors(digraph, node):
            if node in nx.neighbors(digraph, ngbr):
                graph.edges[node, ngbr]["n_passes"] = (
                    digraph.edges[node, ngbr]["n_passes"] + digraph.edges[ngbr, node]["n_passes"]
                )
    return graph


class Venue(Enum):
    HOME = "home"
    AWAY = "away"


class Situation(Enum):
    REGULAR = "5 on 5"
    POWER_PLAY = "5 on 4"
    PENALTY_KILL = "4 on 5"


class Game:
    def __init__(self, game):
        [dt, away, _, home] = game.split(" ")
        self.game = game
        self.home = TEAM_NAMES[home]
        self.away = TEAM_NAMES[away]
        self.date = date.fromisoformat(dt)

    @property
    def game_date(self):
        return f"{self.date.day}/{self.date.month}/{self.date.year}"

    @property
    def roster_file(self):
        return os.path.join(TRACKING_DIR, self.game, f"{self.game} roster.csv")


class PowerPlay:
    def __init__(self, penalty_no):
        self.penalty_no = penalty_no


def build_networks(
    game: Game,
    venue: Venue = Venue.HOME,
    situation: Situation | None = Situation.REGULAR,
    pp: PowerPlay | None = None,
    play_by_play_data=pd.read_csv(os.path.join(TRACKING_DIR, PLAY_BY_PLAY_DATA_FILE)),
    power_play_info=pd.read_csv(os.path.join(PBP_DIR, POWER_PLAY_INFO_FILE), comment="#"),
):
    """Parameters allow for creating different pass networks.

    Default parameters create the passing network in 5 on 5 situations of the home team.

    """
    roster_info = pd.read_csv(game.roster_file, index_col=0)
    events = play_by_play_data[play_by_play_data["game_date"] == game.game_date]
    match venue:
        case venue.HOME:
            events = events[events["team_name"] == game.home]
        case venue.AWAY:
            events = events[events["team_name"] == game.away]

    passes = events[events["event"] == "Play"]

    if situation is not None:
        passes = passes[passes["situation_type"] == situation.value]

    if pp is not None:
        # Find the unique* row in power_play_info corresponding to the current game and penalty_number in pp.
        # If it does not exist; just return None because there is no network to build.
        try:
            pp_df = power_play_info[
                (power_play_info["game_name"] == game.game) & (power_play_info["penalty_number"] == pp.penalty_no)
            ].iloc[0]
        except IndexError:
            # TODO Return something more graceful.
            return None

        # NOTE: Wrote custom logic to determine plays that happen as part of a PP because the time calculation stuff from the Data_Clean.ipynb notebook did not seem to work correctly; maybe I just made a mistake though.
        if pp_df["start_period"] != pp_df["end_period"]:
            # Take passes that are either in the start_period and the clock is below the PP (clock is counting down)
            passes = passes[
                (
                    (passes["period"] == pp_df["start_period"])
                    & (pp_df["start_game_clock_seconds"] >= passes["clock_seconds"])
                )
                # or passes in the end_period with the clock above the end time of the PP
                | (
                    (passes["period"] == pp_df["end_period"])
                    & (passes["clock_seconds"] >= pp_df["end_game_clock_seconds"])
                )
            ]

        else:
            # Take passes in the same period (start_period == end_period except for the one exception)
            # and PP-Start above current time and PP-END below current time
            passes = passes[
                (passes["period"] == pp_df["start_period"])
                & (pp_df["start_game_clock_seconds"] >= passes["clock_seconds"])
                & (passes["clock_seconds"] >= pp_df["end_game_clock_seconds"])
            ]

    passes = passes.join(roster_info, on="player_name").join(roster_info, on="player_name_2", rsuffix="_2")
    passes = passes.loc[:, ["team_name", "player_name", "position", "player_name_2", "position_2"]]

    def create_graph(factor):
        # factor is either "position" or "player_name"
        passes_by_factor = passes.loc[:, [factor, f"{factor}_2"]].value_counts().reset_index()

        graph = nx.DiGraph(
            passes_by_factor.apply(
                lambda row: (
                    row[factor],
                    row[f"{factor}_2"],
                    {"n_passes": row["count"]},
                ),
                axis=1,
            ),
            name=f"{game.game}_{venue.value}_{f'pp{pp.penalty_no}' if pp is not None else 'regular'}",
        )
        return graph

    return {
        "position_pass_network": create_graph("position"),
        "player_pass_network": create_graph("player_name"),
    }


def calculate_time(
    game: Game,
    pps: list[PowerPlay] | None = None,
    play_by_play_data=pd.read_csv(os.path.join(TRACKING_DIR, PLAY_BY_PLAY_DATA_FILE)),
    power_play_info=pd.read_csv(os.path.join(PBP_DIR, POWER_PLAY_INFO_FILE), comment="#"),
):
    """Calculate total time in seconds passed in game during power_plays in pps.  If pps is None, return time during regular play."""
    total_time = 3 * 20 * 60  # 3 periods of 20min in seconds
    power_play_info = power_play_info[power_play_info["game_name"] == game.game]
    power_play_info = power_play_info.loc[:, ["penalty_number", "start_game_clock_seconds", "end_game_clock_seconds"]]

    if pps is not None:
        power_play_info = power_play_info[power_play_info["penalty_number"].isin([pp.penalty_no for pp in pps])]

    def pp_time(pp):
        start = pp["start_game_clock_seconds"]
        end = pp["end_game_clock_seconds"]
        # NOTE: Man könnte auch bei beiden noch -1 machen.
        if start >= end:
            return start - end
        else:
            return start + (20 * 60 - end)

    pp_total_time = power_play_info.apply(pp_time, axis="columns").sum()

    if pps is not None:
        return pp_total_time
    else:
        return total_time - pp_total_time  # regular_time


if __name__ == "__main__":
    main()
