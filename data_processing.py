import numpy as np
import pandas as pd
import csv

IMF_NAMES = pd.read_csv("data/raw_data/imf_country_to_currency.csv",delimiter=":")
IMF_NAMES = dict(zip(IMF_NAMES['country'],IMF_NAMES['currency']))

CCY_CTRY_CODES = pd.read_csv("data/raw_data/non_pegged.csv")
CURRENCIES = CCY_CTRY_CODES['currency'].to_list()
# CURRENCIES = ['EUR','DKK','BGN']

RETURNS_FREQ = "ME"

START_YEAR = 2002
OUT_OF_SAMPLE_START = 2008
END_YEAR = 2023

################### data for partitioning #######################

#monthly average exchange rates -> quarterly covariances/correlations

fx_rates = pd.read_csv('data/raw_data/imf_exchange_rates.csv')
fx_rates['currency'] = fx_rates['COUNTRY'].map(IMF_NAMES)
fx_rates = fx_rates[fx_rates['currency'].isin(CURRENCIES)]

#IMF_NAMES is somewhat ad-hoc for the selected currencies so there may be nans that we don't care about
fx_rates = fx_rates.dropna(subset=["currency"])
fx_rates['Date'] = pd.to_datetime(fx_rates['TIME_PERIOD'], format='%Y-M%m')
fx_rates = fx_rates.drop_duplicates(subset=['currency','Date'])
#imf data is in format one row per month per currency so we pivot it to one row per month, one column per currency
fx_rates = fx_rates.pivot(index='Date', columns='currency', values='OBS_VALUE')

prices = pd.DataFrame(fx_rates).sort_index()
prices = prices.resample(RETURNS_FREQ).last()
prices = prices[(prices.index.year >= START_YEAR) & (prices.index.year <= END_YEAR)]

#log returns
# returns = np.log(prices).diff().dropna()
returns = prices.pct_change().dropna()
returns['USD'] = 0
returns = returns.reindex(columns=['USD']+CURRENCIES)

#get correlations -> scaleless
corr_train = returns[(returns.index.year >= START_YEAR) & (returns.index.year < OUT_OF_SAMPLE_START)]
corr_train = corr_train[CURRENCIES].corr() 
corr_train = corr_train.reindex(index=CURRENCIES,columns=CURRENCIES)

#get covariances -> monthly so we rescale to quarterly
cov_train = returns[(returns.index.year >= START_YEAR) & (returns.index.year < OUT_OF_SAMPLE_START)]
cov_train = cov_train[['USD']+CURRENCIES].cov() * 3
var_train = np.diag(cov_train)
#get relative variances
relative_variance_train = var_train[:, None] + var_train[None, :] - 2 * cov_train.values
relative_variance_train = pd.DataFrame(relative_variance_train,index=cov_train.index,columns=cov_train.columns)
relative_variance_train = relative_variance_train.reindex(index=['USD']+CURRENCIES,columns=['USD']+CURRENCIES)

corr_train.to_csv("data/partition_data/quarterly_correlations.csv")
relative_variance_train.to_csv("data/partition_data/quarterly_relative_variance.csv")

#trade in goods -> daily pairwise trade volumes

trade = pd.read_csv("data/raw_data/imf_trade_monthly.csv",encoding='utf-8')
# trade = trade[trade['FREQUENCY'] == 'Monthly']

trade['source_currency'] = trade['COUNTRY'].map(IMF_NAMES)
trade['destination_currency'] = trade['COUNTERPART_COUNTRY'].map(IMF_NAMES)
trade = trade[trade['source_currency'].isin(CURRENCIES+['USD'])]
trade = trade[trade['destination_currency'].isin(CURRENCIES+['USD'])]

# get the month columns (each have - in them)
month_cols = [col for col in trade.columns if "-" in col]

#get rid of trade btwn countries w same currency
trade = trade[(trade["source_currency"] != trade["destination_currency"])]

# group by source & destination currency
trade = (
    trade
    .groupby(["source_currency","destination_currency"])[month_cols]
    .sum()
    .reset_index()
)
#keep only columns we care about
trade = trade[['source_currency','destination_currency']+month_cols]

train_years = list(range(START_YEAR, OUT_OF_SAMPLE_START)) # 2002-2007
cols_train = [c for c in month_cols if any(c.startswith(str(year)+"-") for year in train_years)]

trade_train = trade.copy()
trade_train['monthly_avg'] = trade_train[cols_train].mean(axis=1) #monthly avg

#format is one row per source/destination pair, we pivot it to pairwise matrix
trade_matrix_train = trade_train.pivot(
    index="source_currency",
    columns="destination_currency",
    values="monthly_avg"
)
#add both directions of flow
trade_matrix_train = trade_matrix_train.add(trade_matrix_train.T, fill_value=0)
trade_matrix_train = trade_matrix_train.fillna(0)
trade_matrix_train = trade_matrix_train * 3 #quarterly scale

trade_matrix_train = trade_matrix_train.reindex(index=['USD']+CURRENCIES,columns=['USD']+CURRENCIES)

trade_matrix_train.to_csv("data/partition_data/quarterly_trade_avg.csv")

##################### data for evaluation: rolling windows #######################

window = 12 # go back 12 months

# returns -> quarterly 1 year rolling window relative variances

returns = returns.sort_index()
rolling_cov = returns.rolling(window).cov() * window/4 # quarterly covariances
rolling_cov = rolling_cov.dropna()

#starts of each quarter
quarter_starts = returns.groupby(returns.index.to_period('Q')).head(1).index

rolling_relvar_quarterly = {}
for date in quarter_starts:
    if date in rolling_cov.index.get_level_values(0):
        date_str = f"{date.year}Q{date.quarter}"
        #calculate relative var for the 12-month window ending at the start of the period
        cov = rolling_cov.loc[date]
        var = np.diag(cov)
        relative_var = var[:, None] + var[None, :] - 2 * cov.values
        temp = pd.DataFrame(relative_var,index=cov.index,columns=cov.columns)
        temp = temp.reindex(index=['USD']+CURRENCIES,columns=['USD']+CURRENCIES)
        rolling_relvar_quarterly[date_str] = temp

        if date.year >= OUT_OF_SAMPLE_START and date.year <= END_YEAR:
            rolling_relvar_quarterly[date_str].to_csv(f"data/rolling_windows/relvars/{date_str}.csv")


#trade -> quarterly 1 year rolling window bilateral trade volumes

df = trade.copy()
#melt to 1 row per month
df_long = df.melt(
    id_vars=["source_currency", "destination_currency"],
    value_vars=month_cols,
    var_name="Date",
    value_name="Trade"
)

# Convert 'Date' to datetime objects and extract quarter
df_long['Date'] = pd.to_datetime(df_long['Date'], format='%Y-M%m')
df_long['Quarter'] = df_long['Date'].dt.to_period('Q')

# Aggregate trade by quarter
df_quarter_agg = df_long.groupby(['source_currency', 'destination_currency', 'Quarter'])['Trade'].sum()
df_quarter_agg = df_quarter_agg.reset_index()

# Quarterly bilateral trade volumes
quarterly_trade = {}

for quarter, group in df_quarter_agg.groupby("Quarter"):
    date_str = f"{quarter.year}Q{quarter.quarter}"
    #pivot to pairwise matrix
    mat = group.pivot(
        index="source_currency",
        columns="destination_currency",
        values="Trade"
    ).fillna(0)
    # add both directions of flow
    mat = mat.reindex(index=['USD']+CURRENCIES, columns=['USD']+CURRENCIES, fill_value=0)
    mat = mat.add(mat.T, fill_value=0)
    np.fill_diagonal(mat.values, 0) #set diag to 0
    quarterly_trade[date_str] = mat
    if int(date_str[:4]) >= OUT_OF_SAMPLE_START and int(date_str[:4]) <= END_YEAR:
        quarterly_trade[date_str].to_csv(f"data/realized_volumes/{date_str}.csv")

#quarters that are out-of-sample---this is what we evaluate on
out_of_sample = [quarter for quarter in quarterly_trade.keys() if int(quarter[:4]) >= OUT_OF_SAMPLE_START and int(quarter[:4]) <= END_YEAR]
with open('data/quarters_list.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for item in out_of_sample:
        writer.writerow([item])

#rolling windows
quarters = 4 #go back 4 quarters
rolling_trade = {}
dates_trade = list(quarterly_trade.keys())
for i in range(quarters, len(dates_trade)):  # need 4 quarters
    window_dates = dates_trade[i-quarters:i]
    
    mats = [quarterly_trade[d] for d in window_dates]

    # avg them
    combined = mats[0].copy()
    for m in mats[1:]:
        combined = combined.add(m, fill_value=0)
    combined = combined/quarters #avg over quarters

    rolling_trade[dates_trade[i]] = combined
    if int(dates_trade[i][:4]) >= OUT_OF_SAMPLE_START and int(dates_trade[i][:4]) <= END_YEAR:
        rolling_trade[dates_trade[i]].to_csv(f"data/rolling_windows/trade/{dates_trade[i]}.csv")
    
