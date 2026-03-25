from scipy import stats
import numpy as np
import pandas as pd

def calculate_log10_strength(age: int, 
                             log10_potential: float, 
                             polynom_3_params: dict
                             ) -> float :
    
    """
    Calculate the current strength (in log10) of a player as a function of the age and potential.
    
    Args:
        age (int): Current age of the player.
        log10_potential (float): Maximum theoretical Potential of the player (in log10).
        polynom_3_params (dict): Dictionary containing the parameters of the aging curve.
    
    Returns: 
        log10_current_strength (float): Current computed strength of the player (in log10).
    """

    
    
    # Get the Parameters of the Aging Curve
    a = polynom_3_params["a"]
    b = polynom_3_params["b"]
    c = polynom_3_params["c"]
    d = polynom_3_params["d"]
    offset = polynom_3_params["offset"]

    # Calculate the log10 Difference from the Potential with the Aging Curve
    # Warning: do not forget the offset to reach the potential!
    log10_difference_from_potential = a*(age**3)+b*(age**2)+c*age+(d-offset)

    # Calculate log10 Current Strength by summing the Two Strengths
    log10_current_strength = log10_potential + log10_difference_from_potential 

    return log10_current_strength

# ----------------------------------------------------------------------------

def generate_new_players(year: int, 
                         config_params: dict):
    
    """
    Generates a new set of tennis players for a specific simulation year.

    Args:
        year (int): Current year of the simulation.
        config_params (dict): Configuration dictionary containing all parameters.
    
    Returns:
        new_players_data (pd.DataFrame): DataFram where each row is a new players with following columns: 
        `player_id`, `age`,` log10_potential`, `current_strength`, `is_active`
    """

    # --- 1. Extracting the Parameters from the Dictionary ---
    
    # Arrival Parameters
    # choices: stationary_model or dynamic_model
    arrival_params = config_params["arrival_params"]["stationary_model"] # stationary or dynamic

    arrival_mu = arrival_params["mu"]
    arrival_sigma = arrival_params["sigma"]

    # Potential (maximum strength) Parameters
    # choices: skewed_normal_single, normal_mixture or skewed_normal_mixture
    potential_params = config_params["potential_params"]["skewed_normal_mixture"]

    potential_mean1 = potential_params["mean1"]
    potential_sigma1 = potential_params["std1"]
    potential_shape1 = potential_params["shape1"]
    potential_mean2 = potential_params["mean2"]
    potential_sigma2 = potential_params["std2"]
    potential_shape2 = potential_params["shape2"]

    potential_w = potential_params["w"] 
    

    # Entry Age Parameters
    entry_age_params = config_params["entry_age_params"]

    entry_age = entry_age_params["fixed_start_age"]

    # Aging Curve Parameters
    aging_curve_params = config_params["aging_curve_params"]["polynom_3_params"]

    # Retirement Parameters
    # choices: global_model or stratified_models (with categories limit and bottom/middle/top))
    retirement_params = config_params["retirement_params"]["stratified_models"]

    retirement_categories_limits = retirement_params["categories_limit"]


    # --- 2. Creating the New Players ---

    # Number of New Players (has to be an integer!)
    nbr_new_players = int(np.random.normal(arrival_mu, arrival_sigma))

    # Creating them by defining an ID
    players_ids = [f"{year}_{player_nbr}" for player_nbr in range(nbr_new_players)]
    
    # Getting them an Age
    players_ages = [entry_age for player_nbr in range(nbr_new_players)]


    # --- 3. Attribute each Player a Potential + Current Strength + Category ---
    players_potentials=[]
    players_current_strengths=[]
    players_categories=[]

    for player_nbr in range(nbr_new_players):

        # Choosing the Distribution (1 or 2) based on the Probability w
        if np.random.random() < potential_w:
            player_potential = stats.skewnorm.rvs(loc = potential_mean1, scale = potential_sigma1, a = potential_shape1)
        else:
            player_potential = stats.skewnorm.rvs(loc = potential_mean2, scale = potential_sigma2, a = potential_shape2)
        
        players_potentials.append(player_potential)

        # calculating the Log10 current Strength based on the Age and log10 Potential
        player_current_strength = calculate_log10_strength(players_ages[player_nbr], player_potential, aging_curve_params)
        players_current_strengths.append(player_current_strength)
    
        # defining the category by looking at the potential
        if player_potential < retirement_categories_limits[0]:
            players_categories.append("bottom")

        elif player_potential > retirement_categories_limits[1]:
            players_categories.append("top")

        else:
            players_categories.append("middle")


    # --- 4. Storing the data in a DataFrame
    new_players = {"player_id": players_ids,
                   "start_year": year,
                   "age": entry_age,
                   "log10_potential": players_potentials,
                   "current_strength": players_current_strengths,
                   "category": players_categories,
                   "is_active": True}

    new_players_data = pd.DataFrame(new_players)

    return new_players_data

# ----------------------------------------------------------------------------

def update_retirement_status(age: int, 
                             category: str, 
                             stratified_params: dict
                             ) -> bool:
    
    """
    Determines whether a player will retire this year (based on probability).

    Args:
        age (int): The player's current age.
        category (str): The player's category (bottom, middle or top).
        stratified_params (dict): A dictionary containing the parameters for the retirement probabilities. 

    Returns:
        bool: True if the player retires, False if he remains active.
    """


    # Types of fit: linear / quadratic / exp
    bottom_fit_parameters = stratified_params["bottom"]["linear"]["params"]
    middle_fit_parameters = stratified_params["middle"]["linear"]["params"]
    top_fit_parameters = stratified_params["top"]["quadratic"]["params"]
    
    # Computing the Probability to Retire, depending on the Category

    if category == "bottom":
        prob_to_retire = bottom_fit_parameters[0]*(age-18)+bottom_fit_parameters[1]

    elif category == "middle":
        prob_to_retire = middle_fit_parameters[0]*(age-18)+middle_fit_parameters[1]

    else: 
        prob_to_retire = top_fit_parameters[0]*(age-18)**2+top_fit_parameters[1]*(age-18)+top_fit_parameters[2]
    
    # ensuring that the probability is between 0 and 1
    prob_to_retire = min(1, prob_to_retire)
    prob_to_retire = max(0, prob_to_retire)

    # Choosing if the player retires depending on the probability
    return np.random.random() < prob_to_retire

# ----------------------------------------------------------------------------

def run_simulation(start_year: int, 
                   end_year: int,
                    config_params: dict
                    ) -> pd.DataFrame:
    
    """
    Performs a complete simulation of the circuit over a given period (from start year to end year),
    accounting for aging, retirements and arrival of new players.

    Args:
        start_year (int): Starting year of the simulation.
        end_year (int): Ending year of the simulation.
        config_params (dict): Configuration dictionary containing all parameters.
    
    Returns:
        pd.DataFrame: DataFrame containg the complete history of all players year by year.
    """

    active_players = pd.DataFrame() # DataFrame containing data about All Players
    history_data = [] # keeps memory of all data, for all years

    # Get the Parameters needed from the Dictionary
    aging_curve_params = config_params["aging_curve_params"]["polynom_3_params"]
    stratified_params = config_params["retirement_params"]["stratified_models"]


    for year in range(start_year, end_year+1):
        
        if not active_players.empty: # to avoid an error for the 1st year (there is no data)

            # The Players are getting older (age +1)
            active_players["age"] += 1

            for index, player in active_players.iterrows():
                
                # Updating the Retirement Status
                if update_retirement_status(player["age"], player["category"], stratified_params):
                    active_players.loc[index, "is_active"] = False
                    active_players.loc[index, "current_strength"] = np.nan # retired players no longer have a strength
                    
                # Update of the Current Strength for the Active Players
                else:
                    new_strength = calculate_log10_strength(player["age"], player["log10_potential"], aging_curve_params)
                    active_players.loc[index, "current_strength"] = new_strength


        # Create New Generation of Players 
        new_players = generate_new_players(year, config_params)
        
        # Create the New DataFrame with all Players
        active_players = pd.concat([active_players, new_players], ignore_index=True)
        active_players["current_year"] = year # add a column with the current year

        # "Snapshot" of the data to save them 
        history_data.append(active_players.copy())

        # Remove All Retired Players from the DataFrame
        active_players = active_players[active_players["is_active"]==True]

    # get the final DataFrame (concatenation of the DataFrames in the list)
    final_data = pd.concat(history_data, ignore_index=True)

    return final_data.sort_values(["player_id", "current_year"]).reset_index(drop=True)  