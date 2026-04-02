import pandas as pd
import numpy as np
import random
from tqdm import tqdm

from simulation import run_simulation, generate_new_players, update_retirement_status

def initialize_tournament(config_params: dict, 
                          start_year: int=-50, 
                          end_year: int=0
                          ) -> pd.DataFrame:

    """
    Generates an initial set of players by running a warm-up simulation from start_year to end_year, and then keeping only the active players at the end of the simulation.
    This allows us to have a realistic distribution of player ages and a stabilized number of active players at the start of the main simulation.

    Args:
        config_params (dict): Configuration dictionary containing all parameters.
        start_year (int): The starting year for the warm-up simulation. Default is -50.
        end_year (int): The ending year for the warm-up simulation. Default is 0.

    Returns:
        pd.DataFrame: A DataFrame containing the initialized players with their attributes, indexed by player_id.

    """
    
    # run the simulation to get a DataFrame of players 
    # = time for having players with different ages and stabilize the number of active players
    warm_up_data = run_simulation(start_year=start_year, end_year=end_year, config_params = config_params)

    # only keep players that are active the last year (and only keep data from the end year)
    end_year_data = warm_up_data[warm_up_data["current_year"] == end_year].copy()
    initialize_players_data = end_year_data[end_year_data["is_active"]].copy()

    # adding them columns useful for tracking the ATP points / number of consecutive weeks played / weeks of rest they need
    initialize_players_data["ATP_points_previous_year"] = ((10 ** initialize_players_data["current_log10_strength"])*100).astype(int)
    initialize_players_data["ATP_points_current_year"] = 0
    initialize_players_data["consecutive_weeks_played"] = 0
    initialize_players_data["weeks_of_rest_needed"] = 0
    # set to -1 to know that they are already present at the start of the simulation 
    initialize_players_data["enter_week"] = -1
    initialize_players_data["retire_week"] = -1

    initialize_players_data.set_index("player_id", inplace=True)

    return initialize_players_data

# ----------------------------------------------------------------------------

def get_available_players(players_data: pd.DataFrame) -> pd.DataFrame:

    """
    Filters players to keep only those who are active, have no weeks of rest needed, and have played less than 2 consecutive weeks.
    Then sorts the available players by their ATP points (current year and previous year) in descending order, and shuffles the players with no ATP points to randomize their order.

    Args:
        players_data (pd.DataFrame): DataFrame containing player information (including their activity status, weeks of rest needed, consecutive weeks played, and ATP points).  

    Returns:
        pd.DataFrame: A DataFrame of available players sorted by ATP points and shuffled for those with no ATP points, indexed by player_id.  
    """

    # keep only active players with no weeks of rest needed and less than 2 consecutive weeks played
    available_players = players_data[players_data["is_active"] &
                                       (players_data["weeks_of_rest_needed"] == 0) &
                                       (players_data["consecutive_weeks_played"] < 2)].copy()
    
    # sort them by their ATP points (current year and previous year) in descending order
    available_players.sort_values(by=["ATP_points_current_year", "ATP_points_previous_year"], ascending=False, inplace=True)

    # if they have no ATP points, shuffle them to randomize their order (instead of always having the same players with no ATP points at the end of the list)
    no_ATP_points_condition = (available_players["ATP_points_current_year"] == 0) & (available_players["ATP_points_previous_year"] == 0)
    players_no_ATP_points = available_players[no_ATP_points_condition]
    players_with_ATP_points = available_players[~no_ATP_points_condition]

    players_no_ATP_points_shuffled = players_no_ATP_points.sample(frac=1) # .sample(frac=1) shuffles the rows of the DF

    # concatenate the players with ATP points (sorted by points) and the players with no ATP points (shuffled)
    available_players_data = pd.concat([players_with_ATP_points, players_no_ATP_points_shuffled])

    return available_players_data

# ----------------------------------------------------------------------------

def assign_players_to_tournaments(available_players_data: pd.DataFrame, 
                                  tournaments_of_the_week: pd.DataFrame
                                  ) -> dict:
    
    """Distributes the available players to the tournaments of the week, depending on the tournament level and the players' ATP ranking.
    
    Args:
        available_players_data (pd.DataFrame): DataFrame containing the available players, sorted by their ATP points (current year and previous year) in descending order, and with players with no ATP points shuffled to randomize their order.
        tournaments_of_the_week (pd.DataFrame): DataFrame containing the tournaments of the week, with their level and capacity (number of players in the main draw and in the qualifiers).
    
    Returns:
        dict: A dictionary where the keys are the tournament IDs (index of the tournaments_of_the_week DataFrame) and the values are dictionaries with two keys: "main" and "qualif"
    """

    registration = {}

    pool_players = available_players_data.copy()
    pool_players["rank"] = range(1, len(pool_players) + 1) # add a column with the ATP ranking of the players 

    # group the tournaments by level
    for level, group_of_tournaments in tournaments_of_the_week.groupby("level", sort=False):

        # -- 1. Filtering the eligible players depending on their ATP ranking and tournament level --
        eligible_players = pool_players.copy()

        if level==10 or level==20:
            eligible_players = eligible_players[eligible_players["rank"] > 150]

        elif level < 250:
            eligible_players = eligible_players[eligible_players["rank"] > 50]

        elif level == 250:
            eligible_players = eligible_players[eligible_players["rank"] > 30]

        elif level == 500:
            top_30_players = eligible_players[eligible_players["rank"] <= 30]
            chosen_top_30_players = top_30_players.sample(frac=0.3)
            eligible_players = pd.concat([chosen_top_30_players, eligible_players[eligible_players["rank"] > 30]])

        # -- 2. Calculating the total capactiy of the tournaments of the week (direct acceptances and qualifiers players) --
        
        number_of_tournaments = len(group_of_tournaments)
        capacity = group_of_tournaments["players"].iloc[0]
        capacity_qualif = group_of_tournaments["qualif"].iloc[0]
        number_qualified = group_of_tournaments["num_qualified"].iloc[0]

        capacity_main = capacity - number_qualified
        total_directed_players = number_of_tournaments * capacity_main
        total_qualif_players = number_of_tournaments * capacity_qualif

        # -- 3. Selecting the players --

        selected_players_main = eligible_players.iloc[:total_directed_players]
        selected_players_qualif = eligible_players.iloc[total_directed_players:total_directed_players+total_qualif_players]

        selected_players_main_shuffled = selected_players_main.sample(frac=1)
        selected_players_qualif_shuffled = selected_players_qualif.sample(frac=1)

        # -- 4. Distributing the players in the different tournaments --

        index_main, index_qualif = 0, 0
        for i, tournament in group_of_tournaments.iterrows():
            main_players = selected_players_main_shuffled.iloc[index_main:index_main+capacity_main]
            qualif_players = selected_players_qualif_shuffled.iloc[index_qualif: index_qualif+capacity_qualif]

            registration[i] = {
                "main": main_players.index.tolist(),
                "qualif": qualif_players.index.tolist()
            }

            index_main += capacity_main
            index_qualif += capacity_qualif

        selected_players_tournament_level = pd.concat([selected_players_main_shuffled.iloc[:index_main], selected_players_qualif_shuffled.iloc[:index_qualif]])
        pool_players = pool_players.drop(selected_players_tournament_level.index) # remove the selected players from the pool of available players for the next tournament levels
                
    return registration

# ----------------------------------------------------------------------------

def play_tennis_game(player_id_A: str, 
                     player_id_B: str, 
                     log10_strength_A: float, 
                     log10_strength_B: float
                     ) -> tuple:

    """
    Simulates a tennis game between two players, given their log10 strength values, and returns the loser and winner.

    Args:
        player_id_A (str): The ID of player A.
        player_id_B (str): The ID of player B.
        log10_strength_A (float): The log10 strength value of player A.
        log10_strength_B (float): The log10 strength value of player B.
    
    Returns:
        tuple: A tuple containing the ID of the loser and the ID of the winner: (loser_id, winner_id).
    """

    # probability of player A beating player B using the BT probability (in log10)
    prob_A_beat_B = 1 / (1+10**(log10_strength_B-log10_strength_A))

    if np.random.random() < prob_A_beat_B:
        return (player_id_B, player_id_A)
    
    else: return (player_id_A, player_id_B) 

# ----------------------------------------------------------------------------

seed_positions = {
        16: [0, 8, 4, 12],
        32: [0, 16, 8, 24, 4, 12, 20, 28],
        64: [0, 32, 16, 48, 8, 24, 40, 56, 4, 12, 20, 28, 36, 44, 52, 60],
        128: [0, 64, 32, 96, 16, 48, 80, 112, 8, 24, 40, 56, 72, 88, 104, 120, 4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124]
}


def prepare_tournament_draw(selected_players_ids: list, 
                            current_week: int, 
                            players_data: pd.DataFrame, 
                            seeding:bool=True
                            ) -> list:
    """Prepares the tournament draw by selecting the players and calculate their seeding positions (if seeding is True) if they have one, else get them a random place.
    
    Args:
        selected_players_id (str): List of player IDs selected for the tournament.
        current_week (int): The current week of the year.
        players_data (pd.DataFrame): DataFrame containing player information(including ID, ATP points of current and previous year, ...) 
        seeding (bool): Whether to apply seeding or not. 
    
    Returns:
        list: A list of player IDs representing the tournament draw, where the index of each player corresponds to their position in the draw.
    """
    
    # get the data associated to the selected players
    selected_players_data = players_data.loc[selected_players_ids].copy()    

    if seeding:
        # calculate the ranking scores (by taking into account both the current and previous year's points
        # As we advance in the year, the current's year points become more important (current year + preivous year * (46-current week)/46))
        # avoids having really good players from the previous year that didn't play much in the current year being ranked very low
        selected_players_data["score_seeding"] = selected_players_data["ATP_points_current_year"] + selected_players_data["ATP_points_previous_year"]*(46-current_week)/46

        # sort the players by their ranking score
        selected_players_data.sort_values(by="score_seeding", ascending=False, inplace=True)

        # create a list for storing the ranking scores
        rankings = selected_players_data.index.tolist()

        # number of seeds depends on the number of players in the tournament   
        number_of_seeds = len(rankings) // 4
        seed_players = rankings[:number_of_seeds]
        other_players = rankings[number_of_seeds:]
        random.shuffle(other_players)

        tournament_draw = [None] * len(rankings)

        # seed process: the top 2 seeds are placed in specifi positions, and the others are shuffled to randomize their positions
        positions = seed_positions[len(rankings)]
        atp_steps = [(0,2), (2,4), (4,8), (8,16), (16,32)]

        for start, end in atp_steps:
            if end > number_of_seeds:
                break
            
            group_positions = positions[start:end]
            group_players = seed_players[start:end]

            if start > 0:
                random.shuffle(group_positions)
            
            for player, position in zip(group_players, group_positions):
                tournament_draw[position] = player
        
        # complete the draw with the non-seeded players
        for i in range(len(tournament_draw)):
            if tournament_draw[i] is None:
                tournament_draw[i] = other_players.pop(0)
                
    # if no seeding: just shuffle the players to randomize their positions
    else:
        tournament_draw = selected_players_data.index.tolist()
        random.shuffle(tournament_draw)


    return tournament_draw

# ----------------------------------------------------------------------------

rounds_dict = {128: "R128", 64: "R64", 32: "R32", 16: "R16", 8: "QF", 4: "SF", 2: "F"}


def play_main_tournament(tournament_draw: list,
                        tournament_id: int,
                        tournament_level: int,
                        current_week: int,
                        year: int,
                        players_data: pd.DataFrame
                        ) -> tuple:
    
    """
    Simulates the main draw of the tournament, given the list of players present in the tournament.
    
    Args:
        tournament_draw (list): List of player IDs present in the tournament.
        tournament_id (int): The ID of the tournament (index of the `tournaments_schedule` DataFrame).
        tournament_level (int): The level of the tournament.
        current_week (int): The current week of the tournament.
        year (int): The current year.
        players_data (pd.DataFrame): DataFrame containing the players data, including their current log10 strength values, indexed by player_id.
    
    Returns:
        tuple: A tuple containing the ID of the winner of the tournament and a list of dictionaries with the history of the games played in the tournament (tournament level, tournament id, week, round, winner, loser).
    """
    # list for storing the history of the games played in the tournament
    games_history = []

    # get the log10 strength values of the players in the tournament
    selected_players_data = players_data.loc[tournament_draw].copy()
    log10_strengths_tournament_players = selected_players_data["current_log10_strength"].to_dict()

    # simulate the games of the tournament until we get a winner
    while len(tournament_draw) > 1:

        # get the round of the tournament with the associated dictionary
        current_round = rounds_dict[len(tournament_draw)]

        winners_of_the_round = [] # store the players that won in the current round (to create the next round)
        for i in range(0, len(tournament_draw), 2):

            # simulate the games between every pair of players
            player_id_A = tournament_draw[i]
            player_id_B = tournament_draw[i+1]

            log10_strength_A = log10_strengths_tournament_players[player_id_A]
            log10_strength_B = log10_strengths_tournament_players[player_id_B]
            loser, winner = play_tennis_game(player_id_A, player_id_B, log10_strength_A, log10_strength_B)
        
            # add the game to the history of the games played in the tournament
            games_history.append({
                "tournament_level": tournament_level,
                "tournament_id": tournament_id,
                "week": current_week,
                "year": year,
                "round": current_round,
                "winner_id": winner,
                "loser_id": loser,
                  })

            winners_of_the_round.append(winner)

        # update the tournament draw for the next round with the winners of the current round
        tournament_draw = winners_of_the_round

    # the winner is the last remaining player in the list
    winner = tournament_draw[0]

    return winner, games_history

# ----------------------------------------------------------------------------

def play_qualifications(qualif_draw: list,
                        tournament_id: int, 
                        tournament_level: int, 
                        current_week: int,
                        year: int, 
                        num_qualified: int, 
                        players_data: pd.DataFrame
                        ) -> tuple:

    """
    Simulates the qualification rounds of the tournament, given the list of players in the qualifications and the final number of qualified players at the end of the qualifications.

    Args:
        qualif_draw (list): List of player IDs present in the qualifications.
        tournament_id (int): The ID of the tournament (index of the `tournaments_schedule` DataFrame).
        tournament_level (int): The level of the tournament.
        current_week (int): The current week of the tournament.
        year (int): The current year.
        num_qualified (int): The number of players that will qualify at the end of the qualifications.
        players_data (pd.DataFrame): DataFrame containing the players data, including their current log10 strength values, indexed by player_id.

    Returns:
        tuple: A tuple containing the list of qualified players and a list of dictionaries with the history of the games played in the qualifications (tournament level, tournament id, week, round, winner, loser).
    """

    # list for storing the history of the games played in the qualifications
    games_history = []

    # get the log10 strength values of the players in the qualifications
    selected_players_data = players_data.loc[qualif_draw].copy()
    log10_strengths_tournament_players = selected_players_data["current_log10_strength"].to_dict()

    # variable to know the tour of the qualifications (from 1 to 2 or 3)
    qualif_tour = 1

    # stop condition: when we have the required number of qualified for the main tournament
    while len(qualif_draw) > num_qualified:

        # names of the rounds: Q1, Q2, ...
        current_round = f"Q{qualif_tour}"

        winners_of_the_round = [] # store the players that won in the current round (to create the next round)

        for i in range(0, len(qualif_draw), 2):

            # simulate the games between every pair of players
            player_id_A = qualif_draw[i]
            player_id_B = qualif_draw[i+1]

            log10_strength_A = log10_strengths_tournament_players[player_id_A]
            log10_strength_B = log10_strengths_tournament_players[player_id_B]
            loser, winner = play_tennis_game(player_id_A, player_id_B, log10_strength_A, log10_strength_B) 
    
            # add the game to the history of the games played in the tournament
            games_history.append({
                "tournament_level": tournament_level,
                "tournament_id": tournament_id,
                "week": current_week,
                "year": year,
                "round": current_round,
                "winner_id": winner,
                "loser_id": loser,
                  })
            
            winners_of_the_round.append(winner)

        # update the tournament draw for the next round with the winners of the current round
        qualif_draw = winners_of_the_round
        # add 1 to the qualif_tour
        qualif_tour += 1

    # trick for giving the qualified players their points 
    # in my function giving points to players: only the loser gets the points
    # here: put the qualified as loser to get their qualified points
    for qualifier_id in qualif_draw:
         games_history.append({
                "tournament_level": tournament_level,
                "tournament_id": tournament_id,
                "week": current_week,
                "year": year,
                "round": "Q",
                "winner_id": None,
                "loser_id": qualifier_id,
                  })
            
    return qualif_draw, games_history

# ----------------------------------------------------------------------------

def update_atp_points(games_history: list,
                    tournaments_points: pd.DataFrame,
                    players_data: pd.DataFrame
                    ) -> pd.DataFrame :

    """ 
    Update the ATP points of the players based on the history of the games played in the tournament and the points associated to each round and tournament level.
    Note: only the loser of each game gets the points, except for the final.
    
    Args: 
        games_history (pd.DataFrame): DataFrame containing the history of the games played in the tournament (tournament level, tournament id, week, round, winner, loser).
        tournaments_points (pd.DataFrame): DataFrame containing the points associated to each round and tournament level.
        players_data (pd.DataFrame): DataFrame containing the players data, including their current ATP points, indexed by player_id.   
    
    Returns:
        pd.DataFrame: Updated players_data DataFrame with the new ATP points (current year) after the tournament.
    """

    # dictionary containing the points associated to each round and tournament level
    points_dict = tournaments_points.set_index("level").to_dict("index")

    # loop over all games in the history and update points
    for game in games_history:

        # get the tournament level, round, and loser of the game
        tournament_level = game["tournament_level"]
        round = game["round"]
        loser_id = game["loser_id"]

        # get him the points associated in the ATP points of current year
        loser_points = points_dict[tournament_level][round]
        players_data.at[loser_id, "ATP_points_current_year"] += loser_points

        # for the final (F): the winner also gets points
        if round == "F":
            winner_id = game["winner_id"]
            winner_points = points_dict[tournament_level]["W"]
            players_data.at[winner_id, "ATP_points_current_year"] += winner_points

    return players_data

# ----------------------------------------------------------------------------

def update_players_fatigue(players_data: pd.DataFrame, 
                           games_history: list
                           ) -> pd.DataFrame:
    
    """Update the fatigue of the players based on the history of the games played in the tournament.
    
    Rules: 
    - If a player plays in a tournament in a given week: +1 to consecutive_weeks_played.
    - If a player has played 2 consecutive weeks: +1 to weeks_of_rest_needed, consecutive_weeks_played reset to 0.
    - If a player plays in a big tournament (1000 or 2000): +2 to weeks_of_rest_needed, consecutive_weeks_played reset to 0.
    - If a player doesn't play in a given week: -1 to weeks_of_rest_needed (min 0), consecutive_weeks_played reset to 0.

    Args: 
        players_data (pd.DataFrame): DataFrame containing the players data (including their fatigue parameters (consecutive_weeks_played and weeks_of_rest_needed).
        games_history (pd.DataFrame): DataFrame containing the history of the games played in the tournament (tournament level, tournament id, week, round, winner, loser).

    Returns:
        pd.DataFrame: Updated players_data DataFrame with the new fatigue parameters after the tournament.
    """
    # dictionnary to have players playing in the tournament with the associated tournament level
    players_playing_dict = {}

    # loop over all games in the history and update the players playing in the tournament and the tournament level
    for game in games_history:
        tournament_level = game["tournament_level"]
        loser = game["loser_id"]
        winner = game["winner_id"]
        players_playing_dict[loser] = tournament_level
        players_playing_dict[winner] = tournament_level

    # get the players playing and not playing in the tournament
    players_playing = set(players_playing_dict.keys())
    players_playing.discard(None) # remove the None element if any (comes from the winner of the "Q" round in the qualifications)

    players_not_playing = set(players_data.index) - players_playing 

    # for players not playing: reset their consecutive weeks played to 0 and decrease their weeks of rest needed by 1 (min 0)
    players_data.loc[list(players_not_playing), "consecutive_weeks_played"] = 0

    current_rest = players_data.loc[list(players_not_playing), "weeks_of_rest_needed"]
    players_data.loc[list(players_not_playing), "weeks_of_rest_needed"] = np.maximum(current_rest - 1, 0)

    # for players playing: add 1 to their consecutive weeks played
    players_data.loc[list(players_playing), "consecutive_weeks_played"] += 1

    # for players with more than 2 consecutive weeks: add 1 to their weeks of rest needed and reset their consecutive weeks played to 0
    players_with_2_consecutive_weeks = players_data[players_data["consecutive_weeks_played"] >= 2].index
    players_data.loc[players_with_2_consecutive_weeks, "weeks_of_rest_needed"] = 1
    players_data.loc[players_with_2_consecutive_weeks, "consecutive_weeks_played"] = 0

    # for players playing in big tournaments (1000 or 2000): add 2 to their weeks of rest needed and reset their consecutive weeks played to 0
    big_tournaments_playing_players = [player for player, tournament_level in players_playing_dict.items() if tournament_level in [1000, 2000]]

    if big_tournaments_playing_players: # to avoid problem in the case there is no such a tournament in the week
        players_data.loc[big_tournaments_playing_players, "weeks_of_rest_needed"] = 2
        players_data.loc[big_tournaments_playing_players, "consecutive_weeks_played"] = 0

    return players_data

# ----------------------------------------------------------------------------

def run_full_tournaments(years: int, 
                         config_params: dict,
                         tournaments_schedule: pd.DataFrame,
                         tournaments_points: pd.DataFrame,
                         seeding: bool=True,
                         ) -> tuple:
        """
        Performs the full simulation of the tournaments for a given number of years, by simulating each week of the year and updating the players' ATP points and fatigue accordingly.

        Args:
                years (int): The number of years to simulate.
                config_params (dict): Configuration dictionary containing all parameters.
                tournaments_schedule (pd.DataFrame): DataFrame containing the full schedule of tournaments with their levels and capacities.
                tournaments_points (pd.DataFrame): DataFrame containing the points associated to each round and tournament level.
                seeding (bool): Whether to apply seeding or not in the tournament draws. Default

        Returns:
                tuple: A tuple containing two DataFrames: the first one with the history of all games played in the tournaments (tournament level, tournament id, week, year, round, winner, loser), 
                and the second one with the yearly rankings of the players (player_id, age, current_log10_strength, log10_potential, category, ATP_points_current_year, year).

        Note: 
        The function initializes the tournament by running a warm-up simulation to have a stabilized number of active players at the start of the main simulation. 
        Then, for each year, it simulates each week by generating new players, updating retirement status, assigning players to tournaments, simulating the games, and updating ATP points and fatigue. 
        Finally, it compiles the history of games and yearly rankings into DataFrames.
        """
        # get the age-stratified retirement parameters from the config (for retirement status update)
        age_stratified_params = config_params["retirement_params"]["stratified_models"]
        
        # list for storing all games history and yearly rankings
        all_games = []
        yearly_rankings = []

        print("Initialising the tournament by running a warm-up simulation...")

        # initialize the tournament by running a warm-up simulation to have a stabilized number of active players at the start of the main simulation
        current_players = initialize_tournament(config_params)
        print("End of the initialisation. Starting the simulation of the tournaments...")

        pbar = tqdm(range(1, years + 1))
        # simulate each year
        for year in pbar:
                pbar.set_description(f"Simulating year {year}/{years}...")

                games_history_year = [] # contains all games played during the year

                # generate new players at the beginning of the year 
                new_players = generate_new_players(year=year, aging_model_choice="stratified", config_params = config_params)
                new_players.set_index("player_id", inplace=True)
                new_players.loc[:, "is_active"] = False # they will become active the week they enter the tour

                # get them an entry week (week where is_active = True) and initialize their ATP points to 0
                new_players["enter_week"] = np.random.randint(0, 47, size=len(new_players)) # assign them a random week of entry in the tour
                new_players["ATP_points_current_year"] = 0
                new_players["ATP_points_previous_year"] = 0

                # update the retirement status of the current players and get them a retirement week
                # axis = 1 to apply the function on each row (and not on each column)            
                retirement_status = current_players.apply(lambda player_info: update_retirement_status(player_info["age"], player_info["category"], age_stratified_params), axis=1)
                current_players.loc[retirement_status, "retire_week"] = np.random.randint(0, 47, size=retirement_status.sum())
                
                # get the full data of players for the year
                current_players = pd.concat([current_players, new_players])

                # simulate each week of the year
                for week in range(47):

                        games_history_week = [] # contains all games played during the week

                        # update the activity status of the players depending on their entry week and retirement week
                        current_players.loc[current_players["enter_week"] == week, "is_active"] = True 
                        current_players.loc[current_players["retire_week"] == week, "is_active"] = False

                        # get the players available for the tournaments of the week(active, no rest needed, less than 2 consecutive weeks played)
                        available_players = get_available_players(current_players)

                        # find the tournaments of the week and assign the available players to them depending on their ATP ranking and the tournament level
                        tournaments_of_the_week = tournaments_schedule[tournaments_schedule["week"] == week]
                        registration = assign_players_to_tournaments(available_players, tournaments_of_the_week)

                        # loop over all tournaments of the week
                        for tournament_id, tournament_players in registration.items():
                        
                                # get the tournament level and the number of qualified players for the tournament
                                tournament_level = tournaments_schedule.loc[tournament_id, "level"]
                                num_qualified = tournaments_schedule.loc[tournament_id, "num_qualified"]

                                # separate qualified and main draw players
                                registration_qualif = tournament_players["qualif"]
                                registration_main = tournament_players["main"]

                                # play the qualifications and get the qualified players
                                qualif_draw = prepare_tournament_draw(registration_qualif, week, current_players, seeding=seeding)
                                qualified_players, games_history_qualif = play_qualifications(qualif_draw, tournament_id, tournament_level, week, year, num_qualified, current_players)
                                
                                # play the main tournament
                                main_players = registration_main + qualified_players
                                main_draw = prepare_tournament_draw(main_players, week, current_players, seeding=seeding)
                                winner, games_history_tournament = play_main_tournament(main_draw, tournament_id, tournament_level, week, year, current_players)

                                # register all games (qualification and main tournament)
                                games_history_week.extend(games_history_qualif)
                                games_history_week.extend(games_history_tournament)

                        games_history_year.extend(games_history_week) # add the games of the week to the games of the year

                        # update the ATP points and fatigue based on the games history of the week
                        update_atp_points(games_history_week, tournaments_points, current_players)
                        update_players_fatigue(current_players, games_history_week)

                                
                # at the end of the year
                # save the yearly rankings of the players (player_id, age, current_log10_strength, log10_potential, category, ATP_points_current_year, year)
                year_rankings = current_players.reset_index()[["player_id", "age", "current_log10_strength", "log10_potential", "category", "ATP_points_current_year"]].copy()
                year_rankings["year"] = year
                year_rankings.sort_values(by=["ATP_points_current_year"], ascending=False, inplace=True)
                year_rankings.reset_index(drop=True, inplace=True)
                yearly_rankings.append(year_rankings)

                # update the players data:
                current_players["age"] += 1
                current_players["ATP_points_previous_year"] = current_players["ATP_points_current_year"]
                current_players["ATP_points_current_year"] = 0
                current_players["consecutive_weeks_played"] = 0
                current_players["weeks_of_rest_needed"] = 0

                # reset enter and retire week 
                current_players["enter_week"] = -1
                current_players["retire_week"] = -1

                all_games.extend(games_history_year) #

               
        return pd.DataFrame(all_games), pd.concat(yearly_rankings)