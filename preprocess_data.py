#!/usr/bin/env python3

import itertools
import os
import pickle
from datetime import date
from enum import Enum
from pathlib import Path
from typing import List

import networkx as nx
import pandas as pd

from network import Weight

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

DATA_PATH = "./data/networks/passing"


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
    __match_args__ = "penalty_no"

    def __init__(self, penalty_no):
        self.penalty_no = penalty_no

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, PowerPlay):
            return self.penalty_no == other.penalty_no

        return False


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


def calculate_time(
    game: Game,
    pps: List[PowerPlay] | None = None,
    pp_info=pd.read_csv(os.path.join(PBP_DIR, POWER_PLAY_INFO_FILE), comment="#"),
):
    """Calculate total time in seconds passed in game during power_plays in pps. If pps is None, return time during regular play."""
    total_time = 3 * 20 * 60  # 3 periods of 20min in seconds
    pp_info = pp_info[pp_info["game_name"] == game.game]
    pp_info = pp_info.loc[:, ["penalty_number", "start_game_clock_seconds", "end_game_clock_seconds"]]

    if pps is not None:
        pp_info = pp_info[pp_info["penalty_number"].isin([pp.penalty_no for pp in pps])]

    def pp_time(pp):
        start = pp["start_game_clock_seconds"]
        end = pp["end_game_clock_seconds"]
        # NOTE: Man könnte auch bei beiden noch -1 machen.
        if start >= end:
            return start - end
        else:
            return start + (20 * 60 - end)

    pp_total_time = pp_info.apply(pp_time, axis="columns").sum()

    if pps is not None:
        return pp_total_time
    else:
        return total_time - pp_total_time  # regular_time


def read_networks(situation: Situation) -> list:
    match situation:
        case Situation.REGULAR:
            files = Path(DATA_PATH).glob("*reg*")
        case Situation.POWER_PLAY:
            files = Path(DATA_PATH).glob("*pp*")
        case Situation.PENALTY_KILL:
            files = Path(DATA_PATH).glob("*pk*")

    def files_gen():
        for f in files:
            with open(f, "rb") as f:
                yield pickle.load(f)  # all types (e.g. Game) in the pickle file need to be in scope for this to work

    return files_gen()


def process_graph_for_analysis(G: nx.Graph | nx.DiGraph, make_undirected: bool = False):
    # Do not make undirected by default
    # TODO: I am not sure what makes more sense for our analysis; Haka generally considered the graph as directed; so, put it behind a flag for now
    if make_undirected:
        G = directed_to_undirected(G)

    # Remove the Goalie as he is rarely involved in passes
    try:
        G.remove_node("Goalie")
    except nx.NetworkXError:
        pass

    # Remove loops because they conceptually do not make sense for our analyses
    for u in G.nodes:
        try:
            G.remove_edge(u, u)
        except nx.NetworkXError:
            pass

    # Now, after removing other stuff, normalise edge weights.
    total_passes = sum(w for (_, _, w) in G.edges.data("n_passes"))

    for u, v, n_passes in G.edges.data("n_passes"):
        # For convenience, add multiple "distance" measures to edge.
        G.edges[u, v][Weight.N_PASSES.value] = n_passes
        G.edges[u, v][Weight.REL_PASSES.value] = n_passes / total_passes
        # NOTE: Distance measures are arguably not very meaningful.
        G.edges[u, v][Weight.REC_DISTANCE.value] = total_passes / n_passes
        G.edges[u, v][Weight.PASS_FREQUENCY.value] = n_passes / G.time

    return G


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

    # This is the team we are interested in.
    venue_team = game.home if venue == Venue.HOME else game.away

    # Only take events of team we want.
    events = events[events["team_name"] == venue_team]

    # Filter for situations we want.
    events = events[events["situation_type"] == situation.value]

    if pp is not None:
        # Find the unique* row in power_play_info corresponding to the current game and penalty_number in pp.
        # If it does not exist; just return None because there is no network to build.
        try:
            pp_info = power_play_info[
                (power_play_info["game_name"] == game.game) & (power_play_info["penalty_number"] == pp.penalty_no)
            ].iloc[0]
        except IndexError:
            return None

        # This is the team in power play
        pp_team = pp_info["pp_team_name"]
        # NOTE: These do not get passed as integers for some reason; and that makes me a bit worried.

        # Two cases are valid:
        # - PP Network and (venue_team == pp_team)
        # - PK Network and (venue_team != pp_team)
        # Any other situation, return None
        match (situation, venue_team, pp_team):
            case (Situation.POWER_PLAY, a, b) if a == b:
                pass
            case (Situation.PENALTY_KILL, a, b) if a != b:
                pass
            case _:
                return None

        # NOTE: Wrote custom logic to determine plays that happen as part of a PP because the time calculation stuff from the Data_Clean.ipynb notebook did not seem to work correctly; maybe I just made a mistake though.
        if pp_info["start_period"] != pp_info["end_period"]:
            # Take events that are either in the start_period and the clock is below the PP (clock is counting down)
            events = events[
                (
                    (events["period"] == pp_info["start_period"])
                    & (pp_info["start_game_clock_seconds"] >= events["clock_seconds"])
                )
                # or events in the end_period with the clock above the end time of the PP
                | (
                    (events["period"] == pp_info["end_period"])
                    & (events["clock_seconds"] >= pp_info["end_game_clock_seconds"])
                )
            ]

        else:
            # Take events in the same period (start_period == end_period except for the one exception)
            # and PP-Start above current time and PP-END below current time
            events = events[
                (events["period"] == pp_info["start_period"])
                & (pp_info["start_game_clock_seconds"] >= events["clock_seconds"])
                & (events["clock_seconds"] >= pp_info["end_game_clock_seconds"])
            ]

    passes = events[(events["event"] == "Play") & (events["event_successful"] == "t")]
    passes = passes.join(roster_info, on="player_name").join(roster_info, on="player_name_2", rsuffix="_2")
    passes = passes.loc[:, ["team_name", "player_name", "position", "player_name_2", "position_2"]]

    shots = events[events["event"] == "Shot"]

    def create_graph(factor):
        # factor is either "position" or "player_name"
        passes_by_factor = passes.loc[:, [factor, f"{factor}_2"]].value_counts().reset_index()
        match situation:
            case Situation.REGULAR:
                name = f"{game.game}_reg_{venue.value}"
            case Situation.POWER_PLAY:
                name = f"{game.game}_{pp.penalty_no}_pp_{venue.value}"
            case Situation.PENALTY_KILL:
                name = f"{game.game}_{pp.penalty_no}_pk_{venue.value}"

        g = nx.DiGraph(
            passes_by_factor.apply(
                lambda row: (
                    row[factor],
                    row[f"{factor}_2"],
                    {"n_passes": row["count"]},
                ),
                axis=1,
            )
        )
        g.name = name
        g.game = game
        g.venue = venue
        g.situation = situation
        g.pp = pp
        g.n_shots = len(shots)
        g.n_passes = g.size(weight=Weight.N_PASSES.value)
        g.time = calculate_time(game, pps=[pp]) if pp is not None else calculate_time(game)
        if situation == Situation.REGULAR:
            final_event = events.iloc[-1]
            g.home_score = final_event["goals_for"]
            g.away_score = final_event["goals_against"]
            if final_event["venue"] == "away":
                g.home_score, g.away_score = g.away_score, g.home_score
            g.winner = "home" if g.home_score > g.away_score else "away" if g.away_score > g.home_score else "tie"
            g.win = 1 if g.winner == venue.value else 0
        elif situation == Situation.POWER_PLAY:
            g.goal = {"t": 1, "f": 0}[pp_info["goal"].strip()]

        return g

    return {
        "position_pass_network": create_graph("position"),
        "player_pass_network": create_graph("player_name"),
    }


def main():
    dir = Path(DATA_PATH)
    dir.mkdir(parents=True, exist_ok=True)  # Create directory if it does not exist

    # Generate some combinations of potential networks.
    # Some of them do not make sense, but build_networks returns None in that case
    # e.g. Venue.HOME, Situation.POWER_PLAY, PowerPlay(1) but Venue.HOME is PENALTY_KILL in that Power Play.
    for game, venue, (situation, pp) in itertools.product(
        GAMES,
        [Venue.HOME, Venue.AWAY],
        [(Situation.REGULAR, None)]
        + [(Situation.POWER_PLAY, PowerPlay(penalty_no)) for penalty_no in range(1, MAX_PENALTY_NUMBER + 1)]
        + [(Situation.PENALTY_KILL, PowerPlay(penalty_no)) for penalty_no in range(1, MAX_PENALTY_NUMBER + 1)],
    ):
        res = build_networks(Game(game), venue=venue, situation=situation, pp=pp)
        if res is None:
            continue

        match (game, pp):
            # These two result in 6 on 4 situations instead of 5 on 4
            case ("2022-02-14 USA at Finland", PowerPlay(penalty_no=4)):
                continue
            case ("2022-02-08 Canada at USA", PowerPlay(penalty_no=7)):
                continue

        nw = res["position_pass_network"]
        nw = process_graph_for_analysis(nw)

        match situation:
            case Situation.REGULAR:
                file = Path(f"{game}_reg_{venue.value}.pickle")
            case Situation.POWER_PLAY:
                file = Path(f"{game}_{pp.penalty_no}_pp_{venue.value}.pickle")
            case Situation.PENALTY_KILL:
                file = Path(f"{game}_{pp.penalty_no}_pk_{venue.value}.pickle")

        with open(dir / file, "wb") as f:
            pickle.dump(nw, f)


if __name__ == "__main__":
    main()
