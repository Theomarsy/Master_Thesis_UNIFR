import pandas as pd
import numpy as np
from tqdm import tqdm

def compute_zermelo_strengths(matches_data: pd.DataFrame, 
                              winner_id: str = "winner_id", 
                              loser_id: str = "loser_id", 
                              max_iter: float = 100, 
                              tol: float = 1e-12, 
                              initial_strengths: dict = None,
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

    # --- 1. List of unique players (with their id) ---
    players = set(matches_data[winner_id]) | set(matches_data[loser_id])
    players = sorted(list(players)) # sort players for ordering (+ set to list)
    id_to_index = {player: i for i,player in enumerate(players)} # map player id to index in list


    # --- 2. Initial strengths ---
    pi = np.ones(len(players)) # default initial strengths: 1 (= mean strengh)

    if initial_strengths is not None: # get initial strengths from provided dict (if available)
        for i, player in enumerate(players):
            if player in initial_strengths: 
                pi[i]=initial_strengths[player]     


    # --- 3. Win matrices ---
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


    # --- 4. Iterative algorithm (Newman version, fast one) ---
    # See Newman (2023), Efficient computation of Rankings from Pairwise Comparisons
    # Section 5 (Equation 26)

    epsilon = 1e-12 # small cst used for numerical stability (avoiding division by zero)

    range_iterator = tqdm(range(int(max_iter)), desc=f"Computing Zermelo strengths ({len(matches_data)} matches and {len(players)} players)")

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