import numpy as np
import matplotlib.pyplot as plt


### Function calculates and returns 
# converged value function and optimal capital level policy
def get_V_and_policy(adj_cost, discount, max_iter=1000, tolerance=1e-6):
    # Initial guesses
    V_new = np.zeros(K_points)
    V_old = np.zeros(K_points)
    policy_K_next = np.zeros(K_points)  
    
    exp_prod_level = np.exp(prod_level)
    
    for i in range(max_iter):  
        for j in range(K_points):
            investment = K_grid - K_grid[j]
            values = exp_prod_level * K_grid[j]**out_elasticity - k_unit_price*investment - (adj_cost/2)*investment**2 + discount*V_old
            V_max = np.max(values)
            V_new_index = np.argmax(values)
            V_new[j] = V_max
            policy_K_next[j] = K_grid[V_new_index] 

        if np.max(np.abs(V_old - V_new)) < tolerance:
            print(f"Converged after {i+1} iterations")
            break
        
        V_old = V_new.copy()

    return V_new, policy_K_next


### Function plots value function, investment and next period policies 
# in the subplots with initial capital grid as the x-axis
def plot_V_policies(K_grid, V, policy_investment, policy_K_next):
    plt.figure(figsize=(15, 10))

    plt.subplot(1, 3, 1)
    plt.plot(K_grid, V)
    plt.title('Value Function')
    plt.xlabel('Capital (K)')
    plt.ylabel('Value')

    plt.subplot(1, 3, 2)
    plt.plot(K_grid, policy_investment)
    plt.title('Investment Policy Function')
    plt.xlabel('Capital (K)')
    plt.ylabel('Investment')

    plt.subplot(1, 3, 3)
    plt.plot(K_grid, policy_K_next, label="Next Period Capital")
    plt.plot(K_grid, K_grid, 'r--', label="45-Degree Line") 
    plt.title("Next Period Capital Policy Function")
    plt.xlabel('Capital (K)')
    plt.ylabel("K next")
    plt.legend() 

    plt.tight_layout()
    plt.show()


### Function calculates and returns investment policy 
# with corresponding changed parameters
def analyze_sensitivity(param_name, param_values):
    results = []
    
    for value in param_values:
        if param_name == 'adj_cost':
            adj_cost_local, discount_local = value, discount
        elif param_name == 'discount':
            discount_local, adj_cost_local = value, adj_cost

        _, policy_K_next = get_V_and_policy(adj_cost_local, discount_local)
            
        investment = policy_K_next - K_grid
        results.append((value, investment))
    
    return results


### Function analyzes and plots investment policy sensitivity 
# to different adjustment costs and discount factors
def plot_sensitivities():
    adj_cost_values = [0.01, 0.1, 0.2, 0.3, 0.9]
    adj_cost_results = analyze_sensitivity('adj_cost', adj_cost_values)
    
    plt.subplot(2, 1, 1)
    for value, inv in adj_cost_results:
        plt.plot(K_grid, inv, label=f'γ={value}')
    plt.title('Investment Policy Function for different γ')
    plt.xlabel('Capital (K)')
    plt.ylabel('Investment')
    plt.legend()

    discount_values = [0.01, 0.5, 0.7, 0.95, 0.99]
    discount_results = analyze_sensitivity('discount', discount_values)

    plt.subplot(2, 1, 2)
    for value, inv in discount_results:
        plt.plot(K_grid, inv, label=f'β={value}')
    plt.title('Investment Policy Function for different β')
    plt.xlabel('Capital (K)')
    plt.ylabel('Investment')
    plt.legend()

    plt.tight_layout()
    plt.show()


### Function calculates and plots marginal and average Q
def calc_plot_Q():
    # Calculating marginal Q with the use of derivative definition
    # f'(x) = lim Δx->0 ((f(x+Δx) - f(x)) / Δx)
    marginal_Q = discount * (np.diff(V) / np.diff(K_grid))

    # Aligning marginal Q with capital interval midpoints for plotting
    K_midpoints = (K_grid[:-1] + K_grid[1:]) / 2

    average_Q = V / K_grid

    # Normalizing to plot in the same axis for comparison
    average_Q_normalized = average_Q / np.max(average_Q)
    marginal_Q_normalized = marginal_Q / np.max(marginal_Q)

    plt.figure(figsize=(10, 6))

    plt.plot(K_grid, average_Q_normalized, label="Normalized Average Q", color="purple", linewidth=2)
    plt.plot(K_midpoints, marginal_Q_normalized, label="Normalized Marginal Q", color="blue", linewidth=2)

    plt.xlabel("Current Capital (K)")
    plt.ylabel("Normalized Q")
    plt.title("Normalized Average Q and Marginal Q as a function of Current Capital")
    plt.legend()

    plt.tight_layout()
    plt.show()


### Program entry point
### While running the next plot will not appear 
# unless the current one is closed
if __name__ == '__main__':
    # Defining initial parameters for the model
    discount = 0.95
    out_elasticity = 0.7
    adj_cost = 0.2
    k_unit_price = 1.2
    prod_level = 1.5

    # Constructing  capital grid
    K_min, K_max = 30, 80
    K_points = 301
    K_grid = np.linspace(K_min, K_max, K_points)

    # Number of maximum iterations and convergence tolerance
    max_iter = 1000
    tolerance = 10**-8 

    V, policy = get_V_and_policy(adj_cost, discount, max_iter, tolerance)
    investment = policy - K_grid

    plot_V_policies(K_grid, V, investment, policy)

    calc_plot_Q()

    plot_sensitivities()




