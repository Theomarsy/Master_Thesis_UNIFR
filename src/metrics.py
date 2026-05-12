import pandas as pd
import numpy as np
from tqdm.auto import tqdm
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
                              show_tqdm: bool=True
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
    pi = np.ones(len(players))/len(players) # default initial strengths: 1/N 

    if initial_strengths is not None: # get initial strengths from provided dict (if available)
        for i, player in enumerate(players):
            if player in initial_strengths: 
                pi[i]=initial_strengths[player]     

    # --- Iterative algorithm (Newman version, fast one) ---
    # See Newman (2023), Efficient computation of Rankings from Pairwise Comparisons
    # Section 5 (Equation 26)

    epsilon = 1e-12 # small cst used for numerical stability (avoiding division by zero)

    range_iterator = tqdm(range(int(max_iter)), disable= not show_tqdm, desc=f"Computing Zermelo strengths ({len(matches_data)} matches and {len(players)} players)")

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
                     rankings: pd.DataFrame,
                     warm_up_years: int = 0,
                     ) -> pd.DataFrame:
    
    """
    Compute the scores of the different metrics (in-degree, Zermelo and PageRank) based on the games played, and include them in the rankings DataFrame.

    Args:
        games_data (pd.DataFrame): DataFrame containing games results
        rankings (pd.DataFrame) : DataFrame containing yearly rankings of the players
        warm_up_years (int): Number of years to skip (default: 0).

    Returns:
        all_years_rankings (pd.DataFrame): DataFrame containing yearly rankings of the players, with their associated metrics score (in-degree, Zermelo and PageRank)
    """

    years = sorted(rankings["year"].unique())
    all_years_rankings = []

    last_year_strengths = None

    if warm_up_years > 0:
        years = years[warm_up_years:]
        if not years:
            raise ValueError(f"Burn-in period of {warm_up_years} years is too long, no year left to compute the metrics.")
       
    pbar = tqdm(years)
    for year in pbar:
        pbar.set_description(f"Computing metrics (year {year})")

        games_year = games_data[games_data["year"]==year].copy()
        rankings_year = rankings[rankings["year"]==year].copy()

        zermelo_strengths_year = compute_zermelo_strengths(games_year, max_iter=300, initial_strengths=last_year_strengths, show_tqdm=False)
        last_year_strengths = zermelo_strengths_year

        pagerank_scores_year = compute_pagerank_scores(games_year)

        in_degrees_year = compute_in_degree(games_year)


        rankings_year["zermelo_strength"] = rankings_year["player_id"].map(zermelo_strengths_year)
        rankings_year["pagerank_score"] = rankings_year["player_id"].map(pagerank_scores_year)
        rankings_year["in_degree"] = rankings_year["player_id"].map(in_degrees_year)
        rankings_year.rename(columns={"current_log10_strength": "log10_hidden_truth", "ATP_points_current_year": "ATP_points"}, inplace=True)

        all_years_rankings.append(rankings_year)

    final_rankings_data = pd.concat(all_years_rankings, ignore_index=True)
        
    return final_rankings_data.sort_values(by=["year", "log10_hidden_truth"], ascending=[True, False]).reset_index(drop=True)

# ----------------------------------------------------------------------------

def get_correlations(ranking_metrics_data: pd.DataFrame,
                     warm_up_years: int=0,
                     return_all_years: bool=False,
                     coefficient: str="kendall"
                     ) -> pd.DataFrame:
    
    """
    Calculates the correlation between the log10 of the hidden truth and the different 
    ranking metrics (ATP points, zermelo strength, pagerank score, in-degree) for each year and each metric.

    Args:
        ranking_metrics_data (pd.DataFrame):  Contains the ranking metrics and the hidden truth for each player and each year.
        warm_up_years (int): Number of initial years to exclude from the correlation calculation (burn-in period). Default is 10.
        return_all_years (bool): Whether to return the correlation for each year or just the mean and std. Default is False.
        coefficient (str): The correlation coefficient to use ("kendall", "spearman", "pearson", "weighted_footrule"). Default is "kendall".

    Returns:
        pd.DataFrame: A DataFrame containing the correlation values for each metric and each year 
        (or just the mean and std if return_all_years is False).
    """

    # remove values with missing zermelo strength
    rankings_metrics_clean = ranking_metrics_data.dropna(subset=["zermelo_strength"]).copy()
    
    # get the years available
    years = sorted(rankings_metrics_clean["year"].unique())

    # get the years to consider and verifiy it is not greater than the total number of years available
    valid_years = years[warm_up_years:]
    if not valid_years:
        raise ValueError("No years available after burn-in period. Please reduce the number of warm-up years.")
        
    # get the columns name for the different metrics
    metrics = ["ATP_points", "zermelo_strength", "pagerank_score", "in_degree"]

    results_all_years = []
    metrics_scores_lists = {metric : [] for metric in metrics}

    for year in valid_years:

        rankings_metrics_year = rankings_metrics_clean[rankings_metrics_clean["year"]==year]
        corr_year = {"year": year} # store the correlation values for each metric for this year

        for metric in metrics:
            if coefficient in ["kendall", "spearman", "pearson"]:
                corr_metric_year = rankings_metrics_year[["log10_hidden_truth", metric]].corr(method=coefficient).iloc[0,1]
                corr_year[metric]=corr_metric_year # add the correlation for this metric to the dictionary
                metrics_scores_lists[metric].append(corr_metric_year)
            
            elif coefficient == "weighted_footrule":
                # compute the weighted footrule (see Who's #1? The Science of Rating and Ranking, Langville and Meyer)
                # method = min implies that tied players receive the lower rank of the groupe 
                # (if 2nd and 3rd are tied, they will both get 2nd in the ranking, and the next one will have the 4th place)
                truth_rank = rankings_metrics_year["log10_hidden_truth"].rank(ascending=False, method="min")
                corr_rank = rankings_metrics_year[metric].rank(ascending=False, method="min")

                numerator = np.abs(truth_rank-corr_rank)
                denominator = np.minimum(truth_rank, corr_rank) # returns the smaller value for each pair
                weighted_footrule_year = np.sum(numerator/denominator)

                # get the normalisation constant as a function of the length of the rankings
                N_year = len(truth_rank)
                ideal_ranks = np.arange(1,N_year+1)
                worst_ranks= np.arange(N_year, 0, -1)

                # get the worse possible weighted footrule with inversed rankings
                max_numerator = np.abs(ideal_ranks-worst_ranks)
                max_denominator = np.minimum(ideal_ranks, worst_ranks)
                max_weighted_footrule_year = np.sum(max_numerator/max_denominator)

                normalised_weighted_footrule_year = 1-weighted_footrule_year/max_weighted_footrule_year

                # save the results in both lists
                corr_year[metric] = normalised_weighted_footrule_year
                metrics_scores_lists[metric].append(normalised_weighted_footrule_year)

            else:
                raise ValueError(f"Coefficient {coefficient} unknown. Please choose between these options: kendall, spearman, pearson and weighted_footrule.")
            
        results_all_years.append(corr_year)

    # if we want all the data, each year (for graphical visualisation)
    if return_all_years:
        return pd.DataFrame(results_all_years)
    
    else:
        results = []
        for metric in metrics:
            results.append({"Metric": metric,
                            "Coefficient": coefficient,
                            "Mean correlation": np.mean(metrics_scores_lists[metric]),
                            "Std": np.std(metrics_scores_lists[metric])})
            
        return pd.DataFrame(results)

# ----------------------------------------------------------------------------

def get_fraction_of_N_players(ranking_metrics_data: pd.DataFrame,
                              N: int=100,
                              warm_up_years: int=0,
                              return_all_years: bool=False
                              ) -> pd.DataFrame:
    
    """
    Calculates the fraction of players in the top N of the hidden truth that are also in the top N of each ranking metric for each year and each metric.

    Args:
        ranking_metrics_data (pd.DataFrame):  Contains the ranking metrics and the hidden truth for each player and each year.
        N (int): The number of top players to consider for the calculation. Default is 100.
        warm_up_years (int): Number of initial years to exclude from the calculation (burn-in period). Default is 0.
        return_all_years (bool): Whether to return the fraction for each year or just the mean and std. Default is False.  

    Returns:
        pd.DataFrame: A DataFrame containing the fraction of players in the top N of the hidden truth that are also in the top N of each metric for each year 
        (or just the mean and std if return_all_years is False).
    """


    # remove values with missing zermelo strength
    rankings_metrics_clean = ranking_metrics_data.dropna(subset=["zermelo_strength"]).copy()
    # get the years available
    years = sorted(rankings_metrics_clean["year"].unique())

    # get the minimum number of players each year to have a limit on N
    min_players_in_year = rankings_metrics_clean.groupby("year")["player_id"].nunique().min()
    if N > min_players_in_year:
        raise ValueError(f"N is too large. The minimum number of players in a year is {min_players_in_year}. Please change N.")

    # get the years to consider and verifiy it is not greater than the total number of years available
    valid_years = years[warm_up_years:]
    if not valid_years:
        raise ValueError("No years available after burn-in period. Please reduce the number of warm-up years.")

    # get the columns name for the different metrics
    metrics = ["ATP_points", "zermelo_strength", "pagerank_score", "in_degree"]

    results_all_years = []
    metrics_fraction_lists = {metric : [] for metric in metrics}

    for year in valid_years:
        rankings_metrics_year = rankings_metrics_clean[rankings_metrics_clean["year"]==year] # get the data for this year

        # get the hidden truth: N best players
        top_N_players_truth = set(rankings_metrics_year.nlargest(N, "log10_hidden_truth", keep="all")["player_id"])

        fraction_year = {"year": year} # store the fraction values for each metric for this year
        
        for metric in metrics:
            # get the top N players for this metric and this year
            top_N_players_metric = set(rankings_metrics_year.nlargest(N, metric, keep="all")["player_id"])

            fraction_metric_year = len(top_N_players_truth.intersection(top_N_players_metric)) / N

            fraction_year[metric] = fraction_metric_year
            metrics_fraction_lists[metric].append(fraction_metric_year) # add the fraction for this metric to the dictionary
        
        results_all_years.append(fraction_year)

    # if we want all the data, each year (for graphical visualisation)
    if return_all_years:
        return pd.DataFrame(results_all_years)

    else:
        results = []
        for metric in metrics:
            results.append({"Metric": metric,
                            "Mean fraction": np.mean(metrics_fraction_lists[metric]),
                            "Std": np.std(metrics_fraction_lists[metric])})


        return pd.DataFrame(results)
    
# ----------------------------------------------------------------------------

