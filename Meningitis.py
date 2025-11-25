#Packages 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import scipy as sp
from itertools import combinations
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import poisson

#Seed
np.random.seed(357948)

#==============================================================================
#=================================== POINT 3 ==================================
#==============================================================================

# MCMC Function
def mcmc_algorithm(mu, lb_mu, ub_mu, eta, B, y):
    """
    Function that recives 
    - mu = initial value, needed to start Metropolis
    - lb_mu = lower bound for mu
    - ub_mu = upper bound for mu
    - eta = initial value of the adaptive proposal
    - B = nu of simulations
    - y = dataset, whether as an array or pandas df
    """
    #1) initializatons - Metropolis step
    n = y.shape[0]
    k = y.shape[1] #number of columns
    mu_list = [mu]
    count_iter = 0 #count the number of iterations

    #parameters to update eta
    A = 500 #parameter to compute gamma_b
    D = 1000 #parameter to compute gamma_b
    c = 50 #every c steps --> update eta
    alpha_list = []
    alpha_star = 0.234

    #1bis) initializatons - Gibbs step
    lambda_current = np.ones(k) #initial values of lambdas
    storage_lambda = np.zeros((B, k)) #matrix where lambdas resulting from mcmc will be stored
    y_sum = np.sum(y, axis=0)
    
    #2) Algorithm
    for b in range(B):
        #draw mu* from the proposal distribution
        mu_star = np.random.normal(mu, eta)
        # if mu is out of the domain --> append the last previous value and skip to the next iter. Otherwhise, execture the algorithm
        if mu_star>= lb_mu and mu_star <= ub_mu: #i.e. indicator function

            #alpha (log version, in order to avoid huge numbers)
            log_ratio = ( k*(mu_star**2)*np.log(mu_star)- k*sp.special.gammaln(mu_star**2) + (mu_star**2)*np.sum(np.log(lambda_current)) - mu_star*np.sum(lambda_current)
                        ) - (
                k*(mu_list[-1]**2)*np.log(mu_list[-1])- k*sp.special.gammaln(mu_list[-1]**2) + (mu_list[-1]**2)*np.sum(np.log(lambda_current)) - mu_list[-1]*np.sum(lambda_current))
            
            alpha = np.exp(min(0, log_ratio))
            alpha_list.append(alpha)
            #draw a sample from Unif(0,1), then apply acceptance criterion
            u = np.random.uniform(0,1)
            if u<alpha:
                mu_list.append(mu_star)
            else:
                mu_list.append(mu_list[-1])
            count_iter = count_iter +1
            mu = mu_list[-1]
        else:
            mu = mu_list[-1]
            mu_list.append(mu)
            if len(alpha_list) > 0:
                alpha_list.append(alpha_list[-1])
            else:
                alpha_list.append(0)
            count_iter = count_iter +1

        # check if eta requires to be updated (update each c = 50 iter)
        if count_iter % c == 0 and count_iter > 0:
            gamma_b = A/(D+count_iter)
            #compute overline{\alpha}
            mean_alpha_batch = np.mean(alpha_list[-c:])
            #update eta
            eta_new = np.exp( np.log(eta) + gamma_b * (mean_alpha_batch - alpha_star) )
            eta = eta_new
            
        #GIBBS STEP (vectorized, in order to improve efficiency)
        lambda_current = np.random.gamma(mu**2 + y_sum, 1/(mu + n))
        storage_lambda[b, :] = lambda_current
    return storage_lambda, mu_list


# BURNIN & THINNING FUNCTIONS
def burnin(mu_list, storage_lambda, burn):
    """
    Function that performs burnin, i.e. discard first _burn_ sample
    - mu_list = output of mu samples from MCMC
    - storage_lambda = output of lambda samples from MCMC
    - burn = how many samples to discard 
    """
    #burnin
    mu_aftburnin = mu_list[burn:] #keep the mu from burn-th element of the list up to the end
    lambda_aftburnin = storage_lambda[burn: , :] #given the B x k array containing lambda_j, keep from the burn-th row up to the end
    return mu_aftburnin, lambda_aftburnin
 
def thin(mu_aftburnin, lambda_aftburnin, thin):
    """
    Function that performs thin, i.e. keeps on sample each _thin_ samples
    - mu_aftburnin = output of burnin
    - lambda_aftburnin = output of burnin
    - thin = step  
    """
    #thinning
    mu_final = mu_aftburnin[::thin] #keep 1 element each "thin" steps
    lambda_final = lambda_aftburnin[::thin , :] #keep 1 row each "thin" steps
    return mu_final, lambda_final



# SAMPLE FROM Y_small
y_small = pd.read_csv(r"C:\Users\MATTEO IENTILE\Desktop\LM ING. MATEMATICA - ANNO1 25_26\LM ANNO 1 25-26\1° SEMESTRE\MODELLI STATISTICI - STATISTICA COMPUTAZIONALE\STATISTICA COMPUTAZIONALE\Homeworks\Homework 1\HM1_y_small.csv") #df #each row --> hospital (i from 1 to 100), #each column --> year (k from 1 to 5)
print(f"{y_small.isna().sum()} missing values within the dataset") #check nan values
# Apply the functions
storage_lambda, mu_list = mcmc_algorithm(
    mu = 8,
    lb_mu = 5,
    ub_mu = 10,
    eta = 1,
    B = 22000,
    y = y_small.to_numpy()
)

#DECIDE Burnin & Thinning BASED ON Traceplot and ACF

#decide burnin
fig, ax = plt.subplots(figsize=(10, 5))
plt.plot(mu_list, alpha=0.8)
ax.set_title("Trace plot μ - Pre Burnin")
plt.show()

#do the burnin
mu_aftburnin, lambda_aftburnin = burnin(mu_list, storage_lambda, 2000)

#ACF plot after burnin to decide THIN
plot_acf(mu_aftburnin, lags=35)
plt.grid()
plt.xlabel("lag")
plt.ylabel("ACF")
plt.show()

# APPLY thin
mu_final, lambda_final = thin(mu_aftburnin, lambda_aftburnin, 10)

#VISUALIZE RESULTS AFTER BURNIN & THIN
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("How the chain looks after Burn-in & Thinning", fontsize=16)

# Traceplot mu after burnin
ax[0].plot(mu_final, alpha=0.7)
ax[0].set_title("Traceplot μ - Post burnin")
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("μ")
ax[0].grid(alpha=0.3)

# ACF plot after burnin
plot_acf(mu_final, lags=10, ax=ax[1])
ax[1].set_title("ACF of μ")
ax[1].set_xlabel("Lag")
ax[1].set_ylabel("ACF")

plt.tight_layout()
plt.show()

# POSTERIOR DISTRIBUTION PLOTS
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Posterior Distributions — μ and λ₁…λ₅", fontsize=18)

# Posterior distribution mu
sns.kdeplot(
    mu_final,
    fill=True,
    color="skyblue",
    alpha=0.4,
    linewidth=2,
    ax=ax[0]
)
ax[0].axvline(np.mean(mu_final), color="red", lw=2, label="Posterior mean")
ax[0].set_title("Posterior KDE of μ")
ax[0].set_xlabel("μ")
ax[0].set_ylabel("Density")
ax[0].legend()
ax[0].grid(alpha=0.3)


# Posterior distribution lambdas
for j in range(lambda_final.shape[1]):
    sns.kdeplot(lambda_final[:, j], 
                label=f"$\\lambda_{j+1}$", 
                fill=True, alpha=0.25, ax=ax[1])

ax[1].set_title("Posterior KDEs of λ₁…λ₅")
ax[1].set_xlabel("λ")
ax[1].set_ylabel("Density")
ax[1].legend()
ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()


#==============================================================================
#=================================== POINT 4 ==================================
#==============================================================================

#COMPUTE DIFFERENCES Function
def cintervals(lambda_final):
    """
    Function that compute the differences for each couple lmabda_l-lambda_h, 
    then returns plots and table with results. The difference is significant
    if 0 falls within the credible interval of the _difference distribution_
    """
    n_sample = lambda_final.shape[0] #number of samples
    k = lambda_final.shape[1] #number of years
    couples = list(combinations(range(k), 2)) #generate all combinations of (lambda_h, lambda_l)

    lambda_diff = np.zeros(shape=(n_sample, len(couples))) #create blank array to fill with differences

    #------ compute differences and CI
    i = 0 
    diff_colname = [] #void list to store label'names
    alpha = 0.05 #level for the CI

    # create a void dictionary to store the results
    statistics = { "couple": [],
                  "lower quantile": [],
                  "upper quantile" : [],
                  "median" : [],
                 "significant difference": []}

    for (l, h) in couples:
        lambda_diff[:,i] = lambda_final[:, l] - lambda_final[:, h] #compute the difference lambda_l - lambda_h
        lw = np.quantile(lambda_diff[:,i], alpha/2) #compute lower quantile
        up = np.quantile(lambda_diff[:,i], 1-alpha/2) #compute upper quantile
        median =np.median(lambda_diff[:,i]) # compute the median
        zero_excluded = (lw > 0) or (up < 0) #check if zero is within the 95% band --> if one OR other is satisfied 0 is not contained --> significant diff
        name = f"Lambda{l+1} - Lambda{h+1}" #create labels for columns
        #append elements
        statistics["couple"].append(name)
        statistics["lower quantile"].append(lw)
        statistics["upper quantile"].append(up)
        statistics["median"].append(median)
        statistics["significant difference"].append(zero_excluded)
        diff_colname.append(name) 
        i=i+1

    #convert arrays and dictionaries into df to manage it easiliy
    lambda_diff_df = pd.DataFrame(lambda_diff, columns=diff_colname)
    statistics_df = pd.DataFrame.from_dict(statistics)
    statistics_df = statistics_df.set_index("couple")
    
    #------ PLOT
    # Define grid dimensions based on the number of comparisons
    num_plots = len(diff_colname)
    cols = 5
    rows = math.ceil(num_plots / cols) 

    # Initialize the figure and axes
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10), constrained_layout=True)
    axes = axes.flatten() 

    for idx, col_name in enumerate(lambda_diff_df.columns):
        ax = axes[idx]
        
        # Retrieve data and statistics for the current difference
        data = lambda_diff_df[col_name]
        
        # Access statistics using the index (couple name)
        stats = statistics_df.loc[col_name] 
        
        lw = stats["lower quantile"]
        up = stats["upper quantile"]
        is_sig = stats["significant difference"]

        # Color logic: Red if significant (0 excluded), Green otherwise
        color_interval = 'firebrick' if is_sig else 'forestgreen'
        title_color = 'red' if is_sig else 'black'

        # Kernel Density Estimation (KDE) plot
        sns.kdeplot(data, ax=ax, fill=True, color='skyblue', alpha=0.4, linewidth=1.5)

        # Vertical lines for Zero and Credible Interval bounds
        ax.axvline(0, color='black', linewidth=2, linestyle='-', label='Zero')
        ax.axvline(lw, color=color_interval, linestyle='--')
        ax.axvline(up, color=color_interval, linestyle='--')
        
        # Shade the Credible Interval area
        ax.axvspan(lw, up, color=color_interval, alpha=0.1)

        # Convert string "LambdaX - LambdaY" to LaTeX "$\lambda_X - \lambda_Y$" for the title
        # We simply replace the text 'Lambda' with the LaTeX command '\lambda_'
        latex_title = r"$" + col_name.replace("Lambda", r"\lambda_") + r"$"

        # Plot aesthetics
        ax.set_title(latex_title, color=title_color, fontweight='bold', fontsize=14)
        ax.set_ylabel('') # Remove Y-axis label for cleanliness
        ax.set_xlabel('Difference')
        
        # Minimal legend on the first plot only to avoid clutter
        if idx == 0:
            ax.legend(["KDE", "Zero", "95% CI"], fontsize='x-small')

    # Remove empty subplots if total plots < grid size
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(r'Posterior Distributions of Differences (with 95% CI)', fontsize=16)
    plt.show()

    return statistics_df


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)
statistics_df = cintervals(lambda_final)
print(statistics_df)

#RIBBON PLOT Function
def ribbon_plot(lambda_final):
    """
    Function that generates a Ribbon Plot to visualize the trend of lambda over the years
    with 95% Credible Intervals
    """

    n_years = lambda_final.shape[1] # number of years (k)
    years = np.arange(1, n_years + 1)  # x-axis: 1, 2, ..., k
    
    # Posterior statistics 
    medians = np.median(lambda_final, axis=0)
    lower_bounds = np.quantile(lambda_final, 0.025, axis=0)
    upper_bounds = np.quantile(lambda_final, 0.975, axis=0)

    #PLOT
    fig, ax = plt.subplots(figsize=(12, 6))

    # ---------------- RIBBON (95% CI) -------------------
    # Soft pastel fill, thin border → ggplot style
    ax.fill_between(
        years,
        lower_bounds,
        upper_bounds,
        color="#a6cee3",        # pastel blue
        alpha=0.35,             # light transparency
        edgecolor="#1f78b4",    # muted blue edge
        linewidth=0.5,
        label="95% Credible Interval"
    )

    # ---------------- TREND LINE (Median) -------------------
    ax.plot(
        years,
        medians,
        color="#1f78b4",         # muted blue
        linewidth=2,             # elegant thin line
        marker="o",              # circular points
        markersize=5,
        markerfacecolor="white", # white center → clean look
        markeredgecolor="#1f78b4",
        label="Posterior Median"
    )

    # ============================================================
    # 3) Aesthetics (ggplot-like)
    # ============================================================

    # Title and axes labels
    ax.set_title(
        r"Trend $\lambda_j$ (95% CI)",
        fontsize=16, weight="bold"
    )
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel(r"$\lambda_j$", fontsize=14)

    # Grid: horizontal dashed & faint (typical ggplot look)
    ax.grid(axis='y', alpha=0.25, linestyle='--')

    # x-axis ticks at discrete years
    ax.set_xticks(years)

    # Remove top and right spines (ggplot minimalism)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Give left/bottom spines a slightly thicker line
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    # Legend without frame → cleaner
    ax.legend(frameon=False, fontsize=12)

    # Final layout and show
    plt.tight_layout()
    plt.show()

# show the ribbon plot for y_small
ribbon_plot(lambda_final)


#==============================================================================
#=================================== POINT 5 ================================== USE THE FULL DATASET "Y" 
#==============================================================================

#Load data
y_big = pd.read_csv(r"C:\Users\MATTEO IENTILE\Desktop\LM ING. MATEMATICA - ANNO1 25_26\LM ANNO 1 25-26\1° SEMESTRE\MODELLI STATISTICI - STATISTICA COMPUTAZIONALE\STATISTICA COMPUTAZIONALE\Homeworks\Homework 1\HM1_y.csv")
# the structure is the same, however an higher number hospitals are taken into account --> higher number of lambda samples
print(f"{y_big.isna().sum()} missing values within the dataset") #check nan values


#SIMULATION
storage_lambda_big, mu_list_big = mcmc_algorithm(
    mu = 8,
    lb_mu = 5,
    ub_mu = 10,
    eta = 1,
    B = 22000,
    y = y_big.to_numpy()
)

#decide burnin
fig, ax = plt.subplots(figsize=(10, 5))
plt.plot(mu_list_big)
ax.set_title("Trace plot μ - Pre Burnin")
plt.show()

#do the burnin if required
mu_aftburnin_big, lambda_aftburnin_big = burnin(mu_list_big, storage_lambda_big, 2000)

#ACF plot after burnin
plot_acf(mu_aftburnin_big, lags=50)
plt.grid(True)
plt.show()

# apply thin
mu_final_big, lambda_final_big = thin(mu_aftburnin_big, lambda_aftburnin_big, 10)

# POSTERIOR DISTRIBUTION PLOTS
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Posterior Distributions — μ and λ", fontsize=18)

# Posterior distribution mu
sns.kdeplot(
    mu_final_big,
    fill=True,
    color="skyblue",
    alpha=0.4,
    linewidth=2,
    ax=ax[0]
)
ax[0].axvline(np.mean(mu_final_big), color="red", lw=2, label="Posterior mean")
ax[0].set_title("Posterior KDE of μ")
ax[0].set_xlabel("μ")
ax[0].set_ylabel("Density")
ax[0].legend()
ax[0].grid(alpha=0.3)


# Posterior distribution lambdas
for j in range(lambda_final_big.shape[1]):
    sns.kdeplot(lambda_final_big[:, j], 
                label=f"$\\lambda_{j+1}$", 
                fill=True, alpha=0.25, ax=ax[1])

ax[1].set_title("Posterior KDEs of λ")
ax[1].set_xlabel("λ")
ax[1].set_ylabel("Density")
ax[1].legend()
ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

#compute credible intervals
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)
statistics_big = cintervals(lambda_final_big)
print(statistics_big)

#ribbon plot
ribbon_plot(lambda_final_big)

#plot mu_small vs mu_big
plt.figure(figsize=(10, 6))

sns.kdeplot(mu_final, fill=True, color='skyblue', alpha=0.4, linewidth=2,
            label=f'Small (N=100) | Mean: {np.mean(mu_final):.2f}')

sns.kdeplot(mu_final_big, fill=True, color='orange', alpha=0.4, linewidth=2,
            label=f'Full  (N=200) | Mean: {np.mean(mu_final_big):.2f}')

plt.title(r"Comparison of Posterior Distributions for $\mu$", fontsize=16)
plt.xlabel(r"$\mu$", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3, linestyle='--')
plt.show()

# Compute width reduction for credible intervals
def numerical_comparison(mu_small, mu_full, lambda_small, lambda_full):
    """
    Compares the the uncertainty of lambda between 
    the small dataset y_small and the full dataset y_big.
    """
    # Compute average CI width for the small dataset
    lb_small = np.quantile(lambda_small, 0.025, axis=0)
    ub_small = np.quantile(lambda_small, 0.975, axis=0)
    avg_width_small = np.mean(ub_small - lb_small)
    
    # Compute average CI width for the small dataset
    lb_full = np.quantile(lambda_full, 0.025, axis=0)
    ub_full = np.quantile(lambda_full, 0.975, axis=0)
    avg_width_full = np.mean(ub_full - lb_full)
    
    #Compute the reduction between uncertainties
    reduction = (1 - avg_width_full / avg_width_small) * 100
 
    print(f"Avg 95% CI Width (Small Dataset): {avg_width_small:.4f}")
    print(f"Avg 95% CI Width (Full Dataset):  {avg_width_full:.4f}")
    print(f"Uncertainty Reduction:          {reduction:.2f}%")

numerical_comparison(mu_final, mu_final_big, lambda_final, lambda_final_big)




#==============================================================================
#=================================== POINT 6 ================================== 
#==============================================================================

#Recall lambda_3 series from POINT 5 
lambda3 = lambda_final_big[:, 2]
# For each lambda_3 sample, simulate Y_{n+1,3} | lambda_3 \sim Poisson(lambda_3)
y_new3 = np.random.poisson(lambda3)


#==============================================================================
#=================================== POINT 6 ================================== 
#==============================================================================

#MONTE CARLO APPROXIMATION f(Y_{n+1,3} | \underline{y})
def montecarlo_approx(year_considered, lambda_final_big):    
    """
    Function that performs Monte Carlo approximation for the integral proposed. It takes inputs
    - year_considered = in which year the measure is added (3 in this case)
    - lambda_final_big = result from MCMC (after burnin and thin)
    """
    # take the lambda_j corrresponding to the year_considered
    lambda_arr_year = lambda_final_big[:, year_considered-1]
    # range of plausible values of y
    y_reasonable = np.arange(0, int(np.max(lambda_arr_year) + 4*np.sqrt(np.max(lambda_arr_year)))+1 ) # assumed as max reasonable value 0 <y< lambda_max + 4*std
    
    #Monte Carlo setting (vectorized, to speed up the process)
    y_r = y_reasonable.reshape(1, -1) # y must be reshaped (from 1D vector to 2D array) in order to perform the vectorized operation
    lambda_r = lambda_arr_year.reshape(-1, 1) #same as above
    
    #compute and store results inside a B X len(y_reasonable) Matrix
    prob_matrix = poisson.pmf(y_r, lambda_r)
    
    #1D array where each entry represent the corresponding probability for each y_reasonable
    estimate =  np.mean(prob_matrix, axis=0) # Rao-Blackwell estimator, i.e. probabiity for each y_reasonable
    
    #------- PLOT
    expected_val = np.sum(y_reasonable * estimate)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    
    # --- PLOT 1: Lollipop Chart (Stem Plot) ---
    # This represents the discrete PMF elegantly using vertical lines ending in a point.
    
    # Draw vertical lines (stems)
    ax[0].vlines(
        x=y_reasonable, 
        ymin=0, 
        ymax=estimate, 
        colors='#4285F4',      # Bright "Google Blue"
        linewidth=1.5,         # Thin, precise lines
        alpha=0.6
    )
    
    # Draw vertex points on top of lines
    ax[0].scatter(
        y_reasonable, 
        estimate, 
        color='#003366',       # Dark Navy Blue for contrast
        s=30,                  # Small dot size
        zorder=3,              # Ensure dots are on top
        label='Point Probability'
    )
    
    ax[0].set_title(r"Predictive PMF", fontsize=14, weight='bold', color='#333333')
    
    # --- PLOT 2: Scatter + Line (Lecture Note Style) ---
    # Replicates the style seen in the provided lecture notes (Page 18).
    # Thin red interpolation line + Empty black/blue circles.
    
    ax[1].plot(
        y_reasonable, 
        estimate, 
        color='#E31A1C',       # Bright Red
        linewidth=1.2,         # Very thin line
        alpha=0.8,
        label='Interpolation'
    )
    ax[1].scatter(
        y_reasonable, 
        estimate, 
        color='#003366', 
        s=25, 
        facecolors='none',     # Empty circles
        edgecolors='#003366',  # Dark borders
        zorder=5, 
        label='Rao-Blackwell Estimate'
    )
    ax[1].set_title(r"Predictive PMF", fontsize=14, weight='bold', color='#333333')

    # --- COMMON FORMATTING ---
    for axis in ax:
        # Baseline at y=0
        axis.axhline(0, color='black', linewidth=1, alpha=0.5)
        
        # Vertical line for Expected Value
        axis.axvline(expected_val, color='#333333', linestyle='--', linewidth=1.5, 
                     label=f'E[Y]={expected_val:.1f}')
        
        # Axis labels
        axis.set_xlabel("Number of Cases ($y$)", fontsize=12)
        axis.set_ylabel("Probability $P(Y=y)$", fontsize=12)
        
        # Minimalist Aesthetics (Remove box borders)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        # Offset axes slightly for a cleaner look
        axis.spines['left'].set_position(('outward', 10))
        axis.spines['bottom'].set_position(('outward', 10))
        
        # Light grid
        axis.grid(axis='y', color='gray', alpha=0.15, linestyle=':')
        
        # Legend
        axis.legend(frameon=False, fontsize=11)

    plt.suptitle(fr"Monte Carlo Estimate Distribution $f(Y_{{n+1, {year_considered}}} \mid \mathbf{{y}})$", fontsize=16)
    plt.show()
                           
    return y_reasonable, estimate, print(f"Sum of the integral: {sum(estimate)}")


#perform Monte Carlo 
montecarlo_approx(3, lambda_final_big)











































