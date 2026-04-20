import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx

def create_win_matrix(matches_data: pd.DataFrame, 
                    winner_id: str = "winner_id", 
                    loser_id: str = "loser_id",
                    mapping_index_to_players: bool=False
                    ) -> tuple:

    """
    Creates the win Matrix based on the results of the games.

    Args:
        matches_data (pd.DataFrame): DataFrame containing match results
        winner_id (str): Column name for winner IDs
        loser_id (str): Column name for loser IDs
        mapping_index_to_players (bool): If true, returns a dictionary where the keys are the indices of players and the values the associated IDs (Default: False).

    Returns:
        tuple: A tuple containing the list of players IDs, the win matrix (and a dictionary mapping indices with players IDs if requested)
    """ 
    
    # --- List of unique players (with their id) ---
    # Note: only players with at least 1 game
    players = set(matches_data[winner_id]) | set(matches_data[loser_id])
    players = sorted(list(players)) # sort players for ordering (+ set to list)
    id_to_index = {player: i for i,player in enumerate(players)} # map player id to index in list

    # --- Win Matrix ---
    wins = np.zeros((len(players), len(players)))  
    # wins[j, i] corresponds to w_ji (number of times player i beat player j)

    # get ID of winner and loser for each match
    winners = matches_data[winner_id].values
    losers = matches_data[loser_id].values

    # fill in the win matrix with match results
    for i in range(len(winners)): 
        # get winner and loser id
        w_id = winners[i]
        l_id = losers[i]

        # get index of winner and loser
        w_index = id_to_index[w_id] 
        l_index = id_to_index[l_id]

        # update win matrix
        wins[l_index,w_index] += 1
    
    if mapping_index_to_players:
        return players, wins, {index: player for player, index in id_to_index.items()}

    else:
        return players, wins
    
# ----------------------------------------------------------------------------

def compute_zermelo_strengths(matches_data: pd.DataFrame, 
                              winner_id: str = "winner_id", 
                              loser_id: str = "loser_id", 
                              max_iter: float = 100, 
                              tol: float = 1e-12, 
                              initial_strengths: dict = None,
                              leave: bool=True
                              ) -> dict :
        
    """
    Computes Zermelo strengths according to the iterative algorithm based on MLE.

    Args:
        matches_data (pd.DataFrame): DataFrame containing match results
        winner_id (str): Column name for winner IDs
        loser_id (str): Column name for loser IDs
        max_iter (float): Maximum number of iterations
        tol (float): Convergence threshold
        initial_strengths (dict): Initial strength values for players

    Returns:
        dict: Player IDs and their corresponding Zermelo strengths
    """

    # --- Get the players (note: the ones that played at least one game) and the win Matrix ---
    players, wins = create_win_matrix(matches_data, winner_id, loser_id)

    # --- Initial strengths ---
    pi = np.ones(len(players)) # default initial strengths: 1 (= mean strengh)

    if initial_strengths is not None: # get initial strengths from provided dict (if available)
        for i, player in enumerate(players):
            if player in initial_strengths: 
                pi[i]=initial_strengths[player]     

    # --- Iterative algorithm (Newman version, fast one) ---
    # See Newman (2023), Efficient computation of Rankings from Pairwise Comparisons
    # Section 5 (Equation 26)

    epsilon = 1e-12 # small cst used for numerical stability (avoiding division by zero)

    range_iterator = tqdm(range(int(max_iter)), leave=leave, desc=f"Computing Zermelo strengths ({len(matches_data)} matches and {len(players)} players)")

    for _ in range_iterator:
        pi_old = pi.copy()

        # update strengths for each player
        for i in range(len(players)):
            numerator, denominator = 0, 0

            # only iterating over opponents that played against player i
            opponents_index = np.where((wins[i,:]>0) | (wins[:,i]>0))[0]

            for j in opponents_index:
                if i==j: continue # skip self-loops

                # common term in numerator and denominator
                pi_sum = pi[i] + pi[j] + epsilon

                numerator +=  (wins[j,i] * pi[j]) / pi_sum
                denominator += wins[i,j] / pi_sum

            # The prior term represents the derivative of the log-prior.
            # It acts as if we add 1 win and 1 loss against a player with strength 1.
            # This avoids problems with undefeated or winless players.
            prior_term = 1 / (pi[i]+1)
            pi[i] = (prior_term+numerator) / (denominator + prior_term)

        # normalization for stability (mean strength = 1)
        pi = pi / np.mean(pi)

        # check if convergence is reached with L1 norm (sum of absolute differences)
        difference = np.linalg.norm(pi - pi_old, ord=1)

        range_iterator.set_postfix({"diff": f"{difference:.2e}"}) # just for displaying difference in tqdm

        if difference < len(players)*tol: 
            break
    
    else:
        print(f"Convergence not reached after {max_iter} iterations. Final L1 error: {difference:.2e}")

    # --- 5.  Final result as a dict ---
    final_pi = {player: pi[i] for i, player in enumerate(players)}

    return final_pi

# ----------------------------------------------------------------------------

def compute_pagerank_scores(matches_data: pd.DataFrame, 
                              winner_id: str = "winner_id", 
                              loser_id: str = "loser_id"
                              ) -> dict :
    
    """
    Computes PageRank scores according to the iterative algorithm.

    Args:
        matches_data (pd.DataFrame): DataFrame containing match results
        winner_id (str): Column name for winner IDs
        loser_id (str): Column name for loser IDs

    Returns:
        dict: Player IDs and their corresponding PageRank scores
    """
    
    # Get the win Matrix and transform it into a Weighted Directed Graph
    _, wins, mapping_index_players = create_win_matrix(matches_data, winner_id, loser_id, mapping_index_to_players=True)

    G = nx.from_numpy_array(wins, create_using=nx.DiGraph) #https://networkx.org/documentation/stable/reference/generated/networkx.convert_matrix.from_numpy_array.html
    G_relabeled = nx.relabel_nodes(G, mapping=mapping_index_players) # label the nodes with the players IDs

    return nx.pagerank(G_relabeled)

# ----------------------------------------------------------------------------

def compute_in_degree(matches_data: pd.DataFrame, 
                       winner_id: str = "winner_id", 
                       loser_id: str = "loser_id"
                       )-> dict:
    """
    Computes In-Degree scores according to the iterative algorithm.

    Args:
        matches_data (pd.DataFrame): DataFrame containing match results
        winner_id (str): Column name for winner IDs
        loser_id (str): Column name for loser IDs

    Returns:
        dict: Player IDs and their corresponding In-Degree scores
    """
        
    
     # --- Get the players (note: the ones that played at least one game) and the win Matrix ---
    players, wins = create_win_matrix(matches_data, winner_id, loser_id)

    # --- Get the in_degree by summing each column of the win Matrix ---
    in_degrees = map(int, np.sum(wins, axis=0))

    return dict(zip(players, in_degrees))

# ----------------------------------------------------------------------------

def get_all_rankings(games_data: pd.DataFrame,
                     rankings: pd.DataFrame
                     ) -> pd.DataFrame:
    
    """
    Compute the scores of the different metrics (in-degree, Zermelo and PageRank) based on the games played, and include them in the rankings DataFrame.

    Args:
        games_data (pd.DataFrame): DataFrame containing games results
        rankings (pd.DataFrame) : DataFrame containing yearly rankings of the players

    Returns:
        all_years_rankings (pd.DataFrame): DataFrame containing yearly rankings of the players, with their associated metrics score (in-degree, Zermelo and PageRank)
    """

    years = rankings["year"].unique()
    all_years_rankings = []

    last_year_strengths = None
       
    pbar = tqdm(years)
    for year in pbar:
        pbar.set_description(f"Computing metrics (year {year})")

        games_year = games_data[games_data["year"]==year].copy()
        rankings_year = rankings[rankings["year"]==year].copy()

        zermelo_strengths_year = compute_zermelo_strengths(games_year, max_iter=300, initial_strengths=last_year_strengths, leave=False)
        last_year_strengths = zermelo_strengths_year

        pagerank_scores_year = compute_pagerank_scores(games_year)

        in_degrees_year = compute_in_degree(games_year)


        rankings_year["zermelo_strength"] = rankings_year["player_id"].map(zermelo_strengths_year)
        rankings_year["pagerank_score"] = rankings_year["player_id"].map(pagerank_scores_year)
        rankings_year["in_degree"] = rankings_year["player_id"].map(in_degrees_year)
        rankings_year.rename(columns={"current_log10_strength": "log10_hidden_truth", "ATP_points_current_year": "ATP_points"}, inplace=True)

        all_years_rankings.append(rankings_year)
        
    return pd.concat(all_years_rankings, ignore_index=True)