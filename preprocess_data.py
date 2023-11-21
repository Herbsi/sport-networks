#!/usr/bin/env python3

import pandas as pd
import os
import networkx as nx
import itertools

from datetime import date
from enum import Enum

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
GAMES = [
    "2022-02-08 Canada at USA",
    "2022-02-08 ROC at Finland",
    "2022-02-12 Switzerland at ROC",
    "2022-02-14 Switzerland at Canada",
    "2022-02-14 USA at Finland",
    "2022-02-16 Switzerland at Finland",
]


def main():
    play_by_play_data = pd.read_csv(os.path.join(TRACKING_DIR, PLAY_BY_PLAY_DATA_FILE))

    def build_networks(game, venue=None, power_play=False):
        game = Game(game)
        roster_info = pd.read_csv(game.roster_file, index_col=0)

        events = play_by_play_data[play_by_play_data["game_date"] == game.game_date]
        if venue == Venue.HOME:
            events = events[events["team_name"] == game.home]
        elif venue == Venue.AWAY:
            events = events[events["team_name"] == game.away]

        # TODO Filter for power plays
        passes = events[events["event"] == "Play"]
        passes = passes.join(roster_info, on="player_name").join(
            roster_info, on="player_name_2", rsuffix="_2"
        )
        passes = passes.loc[
            :, ["team_name", "player_name", "position", "player_name_2", "position_2"]
        ]

        def create_graph(factor):
            # factor is either "position" or "player_name"
            passes_by_factor = (
                passes.loc[:, [factor, f"{factor}_2"]].value_counts().reset_index()
            )
            return nx.DiGraph(
                passes_by_factor.apply(
                    lambda row: (
                        row[factor],
                        row[f"{factor}_2"],
                        {"weight": row["count"]},
                    ),
                    axis=1,
                )
            )

        return {
            "position_pass_network": create_graph("position"),
            "player_pass_network": create_graph("player_name"),
        }

    for game, venue in itertools.product(GAMES, [Venue.HOME, Venue.AWAY]):
        print(f"Processing game {game}; venue {venue}.")
        build_networks(game, venue)


def directed_to_undirected(digraph):
    # From: https://stackoverflow.com/questions/56169907/networkx-change-weighted-directed-graph-to-undirected
    # Yes, I couldn't be bothered to write this simple loop myself
    graph = digraph.to_undirected()
    for node in digraph:
        for ngbr in nx.neighbors(digraph, node):
            if node in nx.neighbors(digraph, ngbr):
                graph.edges[node, ngbr]["weight"] = (
                    digraph.edges[node, ngbr]["weight"]
                    + digraph.edges[ngbr, node]["weight"]
                )
    return graph


class Venue(Enum):
    HOME = "home"
    AWAY = "away"


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


if __name__ == "__main__":
    main()
