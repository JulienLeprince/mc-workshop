import pandas as pd
import matplotlib.pyplot as plt
import pulp

## ===========================================================================
## Preprocessing -------------------------------------------------------------
path_in = r'C:/.../'
path_out = r'C:/.../'

df = pd.read_csv(path_in + 'data_dtu.csv', index_col='t', sep=";")
T_a = df['Ta']
u_solar = df['Ps']
dt = 5/60  # [1/h] sampling every 5 minutes

## ===========================================================================
#-------------Defining system parameters--------------------------------------
## Economic parameters
c_opr = 0.22  #euro/kWh of heat input
c_comf = 1    # euro/C comfort penalty

# Building model parameters
hp_capacity = 8      # Individual HP capacity limit kWh
C_blg = 10
R_blg = 3
Aw = 3

# Comfort set point
T_set = 20

# Problem Horizon
n = df.shape[0]

##============================================================================
#-------------Defining LP problem & variables --------------------------------
my_lp_problem = pulp.LpProblem("My_LP_Problem", pulp.LpMinimize)

#--- Continuous Variables-----------------------------------------------------
u_heat = pulp.LpVariable.dicts('u_heat', range(n), lowBound=0, cat='Continous')  # heat input
T_blg = pulp.LpVariable.dicts('T_blg', range(n+1), lowBound=0, cat='Continuous')  # Building inside temperature
cost_comf = pulp.LpVariable.dicts('comf_cos', range(n+1), lowBound=0, cat='Continuous')  # Building inside temperature

## ===========================================================================
# Objective function----------------------------------------------------------
# Operation costs
cost_opr = c_opr * sum(u_heat[t] for t in range(n))
# Comfort costs
for t in range(n+1):
    my_lp_problem += cost_comf[t] >= c_comf * (T_set - T_blg[t])

# Add it to the problem (cost function definition)
my_lp_problem += cost_opr + sum(cost_comf[t] for t in range(n+1))


## ===========================================================================
# System constraints ---------------------------------------------------------
# Building model
for t in range(n):
    my_lp_problem += T_blg[t+1] - T_blg[t] == (T_a.iloc[t]-T_blg[t])/(C_blg * R_blg)*dt + u_heat[t]*(1/C_blg)*dt \
                                                + (Aw/C_blg)*u_solar.iloc[t]*dt
    # HP
    my_lp_problem += u_heat[t] <= hp_capacity

# Initial conditions
my_lp_problem += T_blg[0] == df["yTi"].iloc[0]

## ===========================================================================
# Optimization ---------------------------------------------------------------
print('Problem constructed!')
start_time = time.time()
status = my_lp_problem.solve()
end_time = time.time() - start_time
print(str(pulp.LpStatus[status]) + ' computing time: ' + str(end_time))
print(pulp.LpStatus[status])
print('Optimal solution found is: ' + str(pulp.value(my_lp_problem.objective)))

##============================================================================
# Results extraction ---------------------------------------------------------
df_res = pd.DataFrame(columns=["heat", "T"])
for t in range(n):
    df_res.loc[t, "heat"] = pulp.value(u_heat[t])
    df_res.loc[t, "T"] = pulp.value(T_blg[t])

df_res.loc[0, "comfort_cost"] = pulp.value(cost_comf)
df_res.loc[0, "objective"] = pulp.value(my_lp_problem.objective)

##============================================================================
# Results visualization-------------------------------------------------------
fig, axs = plt.subplots(2, 1)
axs[0].plot(df_res.index, df_res["T"].values, c='b')
axs[0].axhline(y=T_set, color='red', linestyle='--')
axs[0].set(ylabel='Building inside Temp [C]',
        	title='Optimized control variables')
axs[0].grid()
axs[1].plot(df_res.index, df_res["heat"].values, c='red')
axs[1].set(xlabel='time steps (5min)', ylabel='Heat input [kw]')
axs[1].grid()
plt.show()

##============================================================================
#Saving results --------------------------------------------------------------
# Write and save results
df_res.to_csv(path_out+'buildingcontrol_res.csv')