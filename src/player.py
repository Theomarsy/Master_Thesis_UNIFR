import numpy as np

class Player: 

    def __init__(self, player_id, start_age, potential):
        """Initialises a new player (ID, age at entry, potential)"""
        self.id = player_id
        self.age = start_age
        self.potential = potential

        # performance stats
        self.current_strength = 0
        self.previous_eta = 0 # store the noise from the previous year for calculating fluctuations 
        self.is_active = True # variable to know whether a player is still active or not

        # to track the evolution of strengths (for each age)
        self.strengths_history = {}
    
    def get_older(self):
        """Advance the player's age by 1 (if he hasn't retired)"""
        if self.is_active:
            self.age += 1
    
    def update_strength(self, aging_curve_params):
        """Calculates the player's strength for the urrent age using the aging curve with fluctuations"""
        if self.is_active:
     
            # extract the polynomial coefficients and get the age
            a = aging_curve_params["polynom_3_params"]["a"]
            b = aging_curve_params["polynom_3_params"]["b"]
            c = aging_curve_params["polynom_3_params"]["c"]
            d = aging_curve_params["polynom_3_params"]["d"]
            x = self.age

            # compute the relative strength
            from_curve_strength = a*(x**3)+b*(x**2)+c*x+d
            
            # calculate the fluctuations from the parameters 
            # current fluctuations = phi * previous_fluctuations + white_noise
            phi = aging_curve_params["fluctuations_params"]["phi"]
            sigma_eta = aging_curve_params["fluctuations_params"]["sigma_eta"]
            
            fluctuations = phi * self.previous_eta + np.random.normal(0, sigma_eta)
            self.previous_eta = fluctuations # update for next year

            # total strength = potential + aging effect + random fluctuations
            self.current_strength = (self.potential + from_curve_strength) + fluctuations 

            # record the strength in the history
            self.strengths_history[self.age] = self.current_strength

    
    def update_retirement_status(self, retirement_params):
        """Evaluates if the player retires this year using the retirement probability"""

        if self.is_active:
            
            # extract exponential parameters and age
            a = retirement_params["exp_params"]["a"]
            b = retirement_params["exp_params"]["b"]
            c = retirement_params["exp_params"]["c"]
            x = self.age

            # calculates the probability for current age
            retirement_prob = a*np.exp(b*(x-18))+c

            # if the number is smaller than the probability, the player retires
            # np.min() ensures that the probability <=1
            if np.random.random() < np.min([1,retirement_prob]):
                self.is_active = False


