import numpy as np
from scipy.optimize import minimize
import pandas as pd

#proposition 7.1 approx minimization weights, closed form
def get_var_weighted(relvar,trade,Delta=1):
    Q = trade.sum(axis=1)
    Hi = relvar.sum(axis=1)
    x = np.sqrt(Q/Hi)
    x = x/np.sum(x)
    return x

# expected cost under the 7.1 weights
def expected_cost_multi(relvar,trade,weight_func,Delta=1):
  Q = trade.sum(axis=1)
  w = weight_func(relvar,trade)
  Hw = w.T @ relvar @ w/2
  totalQ = np.triu(trade).sum()
  return np.sqrt(Delta * Hw * np.sum(Q/w))

# expected cost under status quo (prop 4.1)
def expected_cost_status_quo(relvar,trade,Delta=1):
  Q = trade.sum(axis=1)
  totalQ = np.triu(trade).sum()
  sigma2 = relvar['USD']
  return np.sum(np.sqrt(Delta * sigma2 * Q))

def get_trade_sub(trade,group):
    pool = ['USD'] + group
    others = trade.columns.difference(pool)
    trade_sub = trade.loc[pool,pool]
    #trades to currencies outside pool are routed through USD here so add the volume to ccy <-> usd volume
    effective_trades_out = trade.loc[pool,others].sum(axis=1)
    trade_sub['USD'] += effective_trades_out
    trade_sub.loc['USD'] += effective_trades_out
    trade_sub.loc['USD','USD'] = 0 #double check make sure this is zero
    np.fill_diagonal(trade_sub.values, 0) #double check make sure diagonal is zero
    return trade_sub

def get_relvar_sub(relvar,group):
    pool = ['USD'] + group
    relvar_sub = relvar.loc[pool,pool] #cov is unchanged
    return relvar_sub

def get_selection_cost_multi(groups,trade,relvar):
  pool_cost = 0
  for group in groups:
      if len(group) < 2:
          continue
      # #effective trade is for each pool currency, sum of all trades with other currencies go through usd
      trade_sub = get_trade_sub(trade,group)
      relvar_sub = get_relvar_sub(relvar,group)

      #calculate pool cost
      total_pool_trade = np.triu(trade_sub.values).sum()
      cost = expected_cost_multi(relvar_sub,trade_sub,get_var_weighted)
      pool_cost += cost

  #any ungrouped currency just does everything through USD
  pooled_currencies = [item for sublist in groups for item in sublist if len(sublist) >= 2]
  if len(pooled_currencies)+1 < len(trade.columns): #if there's something unpooled
    ungrouped = trade.columns.difference(pooled_currencies+['USD']).to_list()
    #any trades to pooled currencies already have other leg accounted for so add the volume to USD volume to account for this leg
    #otherwise it's just status quo
    trade_sub = get_trade_sub(trade,ungrouped)
    relvar_sub = get_relvar_sub(relvar,ungrouped)
    external_cost = expected_cost_status_quo(relvar_sub,trade_sub)
  else:
    external_cost = 0

  return pool_cost + external_cost

def load_data():
    relvar_train = pd.read_csv("data/partition_data/quarterly_relative_variance.csv",index_col=0)
    currency_trade = pd.read_csv("data/partition_data/quarterly_trade_avg.csv",index_col=0)                
    return relvar_train, currency_trade

def calculate_cost(mode="partition"):
    relvar,trade = load_data()
    if mode == "partition":
        def cost(pools):
            if not pools: # If no valid pools were formed
                return None
            selection_cost = get_selection_cost_multi(pools, trade, relvar)
            cost = selection_cost
            return cost
    else:
       def cost(pool):
          if not pool:
             return None
          pool = ['USD'] + pool
          relvar_sub = relvar.loc[pool,pool]
          trade_sub = trade.loc[pool,pool]
        #   print(relvar_sub)
        #   print(trade_sub)
          numerator = expected_cost_multi(relvar_sub,trade_sub,get_var_weighted)
          denominator = expected_cost_status_quo(relvar_sub,trade_sub)
          cost_ratio = numerator / (denominator+1e-8)
          return cost_ratio
    return cost