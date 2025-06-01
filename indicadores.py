# Construção e Deploy de API - Machine Learning Para Prever o Preço do Bitcoin
# Módulo de Indicadores

# Imports
import numpy as np
import pandas as pd

# Função de cálculo dos indicadores
def bitcoin_calcula_indicadores(bitcoin_dados) :

    bitcoin_dados = bitcoin_ind_williams_percent_r(bitcoin_dados,14)
    bitcoin_dados = bitcoin_ind_roc(bitcoin_dados,14)
    bitcoin_dados = bitcoin_ind_rsi(bitcoin_dados,7)
    bitcoin_dados = bitcoin_ind_rsi(bitcoin_dados,14)
    bitcoin_dados = bitcoin_ind_rsi(bitcoin_dados,28)
    bitcoin_dados = bitcoin_ind_macd(bitcoin_dados, 8, 21)
    bitcoin_dados = bitcoin_ind_bbands(bitcoin_dados,20)    
    bitcoin_dados = bitcoin_ind_ichimoku_cloud(bitcoin_dados)
    bitcoin_dados = bitcoin_ind_ema(bitcoin_dados, 3)
    bitcoin_dados = bitcoin_ind_ema(bitcoin_dados, 8)
    bitcoin_dados = bitcoin_ind_ema(bitcoin_dados, 15)
    bitcoin_dados = bitcoin_ind_ema(bitcoin_dados, 50)
    bitcoin_dados = bitcoin_ind_ema(bitcoin_dados, 100)
    bitcoin_dados = bitcoin_ind_adx(bitcoin_dados, 14)
    bitcoin_dados = bitcoin_ind_donchian(bitcoin_dados, 10)
    bitcoin_dados = bitcoin_ind_donchian(bitcoin_dados, 20)
    bitcoin_dados = bitcoin_ind_alma(bitcoin_dados, 10)
    bitcoin_dados = bitcoin_ind_tsi(bitcoin_dados, 13, 25)
    bitcoin_dados = bitcoin_ind_zscore(bitcoin_dados, 20)
    bitcoin_dados = bitcoin_ind_log_return(bitcoin_dados, 10)
    bitcoin_dados = bitcoin_ind_log_return(bitcoin_dados, 20)
    bitcoin_dados = bitcoin_ind_vortex(bitcoin_dados, 7)
    bitcoin_dados = bitcoin_ind_aroon(bitcoin_dados, 16)
    bitcoin_dados = bitcoin_ind_ebsw(bitcoin_dados, 14)
    bitcoin_dados = bitcoin_ind_accbands(bitcoin_dados, 20)
    bitcoin_dados = bitcoin_ind_short_run(bitcoin_dados, 14)
    bitcoin_dados = bitcoin_ind_bias(bitcoin_dados, 26)
    bitcoin_dados = bitcoin_ind_ttm_trend(bitcoin_dados, 5, 20)
    bitcoin_dados = bitcoin_ind_percent_return(bitcoin_dados, 10)
    bitcoin_dados = bitcoin_ind_percent_return(bitcoin_dados, 20)
    bitcoin_dados = bitcoin_ind_kurtosis(bitcoin_dados, 5)
    bitcoin_dados = bitcoin_ind_kurtosis(bitcoin_dados, 10)
    bitcoin_dados = bitcoin_ind_kurtosis(bitcoin_dados, 20)
    bitcoin_dados = bitcoin_ind_eri(bitcoin_dados, 13)    
    bitcoin_dados = bitcoin_ind_atr(bitcoin_dados, 14)
    bitcoin_dados = bitcoin_ind_keltner_channels(bitcoin_dados, 20)
    bitcoin_dados = bitcoin_ind_chaikin_volatility(bitcoin_dados, 10)
    bitcoin_dados = bitcoin_ind_stdev(bitcoin_dados, 5)
    bitcoin_dados = bitcoin_ind_stdev(bitcoin_dados, 10)
    bitcoin_dados = bitcoin_ind_stdev(bitcoin_dados, 20)
    bitcoin_dados = ta_vix(bitcoin_dados, 21)    
    bitcoin_dados = bitcoin_ind_obv(bitcoin_dados, 10)
    bitcoin_dados = bitcoin_ind_chaikin_money_flow(bitcoin_dados, 5)
    bitcoin_dados = bitcoin_ind_volume_price_trend(bitcoin_dados, 7)
    bitcoin_dados = bitcoin_ind_accumulation_distribution_line(bitcoin_dados, 3)
    bitcoin_dados = bitcoin_ind_ease_of_movement(bitcoin_dados, 14)
    
    return bitcoin_dados

# Williams %R
def bitcoin_ind_williams_percent_r(dados_bitcoin, window=14):
    highest_high = dados_bitcoin["High"].rolling(window=window).max()
    lowest_low = dados_bitcoin["Low"].rolling(window=window).min()
    dados_bitcoin["Williams_%R{}".format(window)] = -((highest_high - dados_bitcoin["Close"]) / (highest_high - lowest_low)) * 100
    return dados_bitcoin

# Rate of Change
def bitcoin_ind_roc(dados_bitcoin, window=14):
    dados_bitcoin["ROC_{}".format(window)] = (dados_bitcoin["Close"] / dados_bitcoin["Close"].shift(window) - 1) * 100
    return dados_bitcoin

# RSI
def bitcoin_ind_rsi(dados_bitcoin, window=14) : 
    delta = dados_bitcoin["Close"].diff(1)
    gains = delta.where(delta>0,0)
    losses = -delta.where(delta<0,0)
    avg_gain = gains.rolling(window=window, min_periods=1).mean()
    avg_loss = losses.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    dados_bitcoin["rsi_{}".format(window)] = 100 - (100 / (1 + rs))
    return dados_bitcoin

# MACD 
def bitcoin_ind_macd(dados_bitcoin, short_window=8, long_window=21, signal_window=9):
    short_ema = dados_bitcoin["Close"].ewm(span = short_window, adjust = False).mean()
    long_ema = dados_bitcoin["Close"].ewm(span = long_window, adjust = False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    dados_bitcoin["MACD_Line"] = macd_line
    dados_bitcoin["Signal_Line"] = signal_line
    dados_bitcoin["MACD_Histogram"] = macd_histogram
    return dados_bitcoin

# Bollinger Bands
def bitcoin_ind_bbands(dados_bitcoin, window=20, num_std_dev=2) :
    dados_bitcoin["midlle_band"] = dados_bitcoin["Close"].rolling(window=window).mean()
    dados_bitcoin["std"] = dados_bitcoin["Close"].rolling(window=window).std()
    dados_bitcoin["upper_band{}".format(window)] = dados_bitcoin["midlle_band"] + (num_std_dev * dados_bitcoin["std"])
    dados_bitcoin["lower_band{}".format(window)] = dados_bitcoin["midlle_band"] - (num_std_dev * dados_bitcoin["std"])
    dados_bitcoin.drop(["std"], axis=1, inplace=True)   
    return dados_bitcoin

# Ichimoku Cloud
def bitcoin_ind_ichimoku_cloud(dados_bitcoin, window_tenkan=9, window_kijun=26, window_senkou_span_b=52, window_chikou=26):
    tenkan_sen = (dados_bitcoin["Close"].rolling(window=window_tenkan).max() + dados_bitcoin["Close"].rolling(window=window_tenkan).min()) / 2
    kijun_sen = (dados_bitcoin["Close"].rolling(window=window_kijun).max() + dados_bitcoin["Close"].rolling(window=window_kijun).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(window_kijun)
    senkou_span_b = (dados_bitcoin["Close"].rolling(window=window_senkou_span_b).max() + dados_bitcoin["Close"].rolling(window=window_senkou_span_b).min()) / 2
    chikou_span = dados_bitcoin["Close"].shift(-window_chikou)
    dados_bitcoin["Tenkan_sen"] = tenkan_sen
    dados_bitcoin["Kijun_sen"] = kijun_sen
    dados_bitcoin["Senkou_Span_A"] = senkou_span_a
    dados_bitcoin["Senkou_Span_B"] = senkou_span_b
    dados_bitcoin["Chikou_Span"] = chikou_span
    return dados_bitcoin

# Moving Average (EMA)
def bitcoin_ind_ema(dados_bitcoin, window=8): 
    dados_bitcoin["ema_{}".format(window)] = dados_bitcoin["Close"].ewm(span=window, adjust=False).mean()
    return dados_bitcoin

# ADX
def bitcoin_ind_adx(dados_bitcoin, window=14): #14
    dados_bitcoin["TR"] = abs(dados_bitcoin["High"] - dados_bitcoin["Low"]).combine_first(abs(dados_bitcoin["High"] - dados_bitcoin["Close"].shift(1))).combine_first(abs(dados_bitcoin["Low"] - dados_bitcoin["Close"].shift(1)))
    dados_bitcoin["DMplus"] = (dados_bitcoin["High"] - dados_bitcoin["High"].shift(1)).apply(lambda x: x if x > 0 else 0)
    dados_bitcoin["DMminus"] = (dados_bitcoin["Low"].shift(1) - dados_bitcoin["Low"]).apply(lambda x: x if x > 0 else 0)
    dados_bitcoin["ATR"] = dados_bitcoin["TR"].rolling(window=window).mean()
    dados_bitcoin["DIplus"] = (dados_bitcoin["DMplus"].rolling(window=window).mean() / dados_bitcoin["ATR"]) * 100
    dados_bitcoin["DIminus"] = (dados_bitcoin["DMminus"].rolling(window=window).mean() / dados_bitcoin["ATR"]) * 100
    dados_bitcoin["DX"] = abs(dados_bitcoin["DIplus"] - dados_bitcoin["DIminus"]) / (dados_bitcoin["DIplus"] + dados_bitcoin["DIminus"]) * 100
    dados_bitcoin["ADX_{}".format(window)] = dados_bitcoin["DX"].rolling(window=window).mean()
    dados_bitcoin.drop(["TR", "DMplus", "DMminus", "ATR", "DIplus", "DIminus", "DX"], axis=1, inplace=True)
    return dados_bitcoin

# Donchian Channel
def bitcoin_ind_donchian(dados_bitcoin, window=10):
    highest_high = dados_bitcoin["Close"].rolling(window=window).max()
    lowest_low = dados_bitcoin["Close"].rolling(window=window).min()
    dados_bitcoin["Donchian_Upper_{}".format(window)] = highest_high
    dados_bitcoin["Donchian_Lower_{}".format(window)] = lowest_low
    return dados_bitcoin

# Arnaud Legoux Moving Average (ALMA)
def bitcoin_ind_alma(dados_bitcoin, window=10, sigma=6, offset=0.85):
    m = np.linspace(-offset*(window-1), offset*(window-1), window)
    w = np.exp(-0.5 * (m / sigma) ** 2)
    w /= w.sum()
    alma_values = np.convolve(dados_bitcoin["Close"].values, w, mode="valid")
    alma_values = np.concatenate([np.full(window-1, np.nan), alma_values])
    dados_bitcoin["ALMA_{}".format(window)] = alma_values
    return dados_bitcoin

# True Strength Index (TSI)
def bitcoin_ind_tsi(dados_bitcoin, short_period=13, long_period=25):
    price_diff = dados_bitcoin["Close"].diff(1)
    double_smoothed = price_diff.ewm(span=short_period, min_periods=1, adjust=False).mean().ewm(span=long_period, min_periods=1, adjust=False).mean()
    double_smoothed_abs = price_diff.abs().ewm(span=short_period, min_periods=1, adjust=False).mean().ewm(span=long_period, min_periods=1, adjust=False).mean()
    tsi_values = 100 * double_smoothed / double_smoothed_abs
    dados_bitcoin["TSI_{}_{}".format(short_period, long_period)] = tsi_values
    return dados_bitcoin

# Z-Score
def bitcoin_ind_zscore(dados_bitcoin, window=20):
    rolling_mean = dados_bitcoin["Close"].rolling(window=window).mean()
    rolling_std = dados_bitcoin["Close"].rolling(window=window).std()
    z_score = (dados_bitcoin["Close"] - rolling_mean) / rolling_std
    dados_bitcoin["Z_Score_{}".format(window)] = z_score
    return dados_bitcoin

# Log Return
def bitcoin_ind_log_return(dados_bitcoin, window=5):
    dados_bitcoin["LogReturn_{}".format(window)] = dados_bitcoin["Close"].pct_change(window).apply(lambda x: 0 if pd.isna(x) else x)
    return dados_bitcoin

# Vortex Indicator
def bitcoin_ind_vortex(dados_bitcoin, window=7): 
    high_low = dados_bitcoin["High"] - dados_bitcoin["Low"]
    high_close_previous = abs(dados_bitcoin["High"] - dados_bitcoin["Close"].shift(1))
    low_close_previous = abs(dados_bitcoin["Low"] - dados_bitcoin["Close"].shift(1))
    true_range = pd.concat([high_low, high_close_previous, low_close_previous], axis=1).max(axis=1)
    positive_vm = abs(dados_bitcoin["High"].shift(1) - dados_bitcoin["Low"])
    negative_vm = abs(dados_bitcoin["Low"].shift(1) - dados_bitcoin["High"])
    true_range_sum = true_range.rolling(window=window).sum()
    positive_vm_sum = positive_vm.rolling(window=window).sum()
    negative_vm_sum = negative_vm.rolling(window=window).sum()
    positive_vi = positive_vm_sum / true_range_sum
    negative_vi = negative_vm_sum / true_range_sum
    dados_bitcoin["Positive_VI_{}".format(window)] = positive_vi
    dados_bitcoin["Negative_VI_{}".format(window)] = negative_vi
    return dados_bitcoin

# Aroon Indicator
def bitcoin_ind_aroon(dados_bitcoin, window=16):
    high_prices = dados_bitcoin["High"]
    low_prices = dados_bitcoin["Low"]
    aroon_up = []
    aroon_down = []
    for i in range(window, len(high_prices)):
        high_period = high_prices[i - window:i + 1]
        low_period = low_prices[i - window:i + 1]
        high_index = window - high_period.values.argmax() - 1
        low_index = window - low_period.values.argmin() - 1
        aroon_up.append((window - high_index) / window * 100)
        aroon_down.append((window - low_index) / window * 100)
    aroon_up = [None] * window + aroon_up
    aroon_down = [None] * window + aroon_down
    dados_bitcoin["Aroon_Up_{}".format(window)] = aroon_up
    dados_bitcoin["Aroon_Down_{}".format(window)] = aroon_down
    return dados_bitcoin

# Elder"s Bull Power e Bear Power 
def bitcoin_ind_ebsw(dados_bitcoin, window=14):
    ema = dados_bitcoin["Close"].ewm(span=window, adjust=False).mean()
    bull_power = dados_bitcoin["High"] - ema
    bear_power = dados_bitcoin["Low"] - ema
    dados_bitcoin["Bull_Power_{}".format(window)] = bull_power
    dados_bitcoin["Bear_Power_{}".format(window)] = bear_power
    return dados_bitcoin

# Acceleration Bands
def bitcoin_ind_accbands(dados_bitcoin, window=20, acceleration_factor=0.02):
    sma = dados_bitcoin["Close"].rolling(window=window).mean()
    band_difference = dados_bitcoin["Close"] * acceleration_factor
    upper_band = sma + band_difference
    lower_band = sma - band_difference
    dados_bitcoin["Upper_Band_{}".format(window)] = upper_band
    dados_bitcoin["Lower_Band_{}".format(window)] = lower_band
    dados_bitcoin["Middle_Band_{}".format(window)] = sma
    return dados_bitcoin

# Short Run
def bitcoin_ind_short_run(dados_bitcoin, window=14):
    short_run = dados_bitcoin["Close"] - dados_bitcoin["Close"].rolling(window=window).min()
    dados_bitcoin["Short_Run_{}".format(window)] = short_run
    return dados_bitcoin

# Bias
def bitcoin_ind_bias(dados_bitcoin, window=26):
    moving_average = dados_bitcoin["Close"].rolling(window=window).mean()
    bias = ((dados_bitcoin["Close"] - moving_average) / moving_average) * 100
    dados_bitcoin["Bias_{}".format(window)] = bias
    return dados_bitcoin

# TTM Trend
def bitcoin_ind_ttm_trend(dados_bitcoin, short_window=5, long_window=20):
    short_ema = dados_bitcoin["Close"].ewm(span=short_window, adjust=False).mean()
    long_ema = dados_bitcoin["Close"].ewm(span=long_window, adjust=False).mean()
    ttm_trend = short_ema - long_ema
    dados_bitcoin["TTM_Trend_{}_{}".format(short_window, long_window)] = ttm_trend
    return dados_bitcoin

# Percent Return
def bitcoin_ind_percent_return(dados_bitcoin, window=1): 
    percent_return = dados_bitcoin["Close"].pct_change().rolling(window=window).mean() * 100
    dados_bitcoin["Percent_Return_{}".format(window)] = percent_return
    return dados_bitcoin

# Kurtosis
def bitcoin_ind_kurtosis(dados_bitcoin, window=20):
    dados_bitcoin["kurtosis_{}".format(window)] = dados_bitcoin["Close"].rolling(window=window).apply(lambda x: np.nan if x.isnull().any() else x.kurt())
    return dados_bitcoin

# Elder's Force Index (ERI)
def bitcoin_ind_eri(dados_bitcoin, window=13):
    price_change = dados_bitcoin["Close"].diff()
    force_index = price_change * dados_bitcoin["Volume"]
    eri = force_index.ewm(span=window, adjust=False).mean()
    dados_bitcoin["ERI_{}".format(window)] = eri
    return dados_bitcoin

# ATR
def bitcoin_ind_atr(dados_bitcoin, window=14):
    dados_bitcoin["High-Low"] = dados_bitcoin["High"] - dados_bitcoin["Low"]
    dados_bitcoin["High-PrevClose"] = abs(dados_bitcoin["High"] - dados_bitcoin["Close"].shift(1))
    dados_bitcoin["Low-PrevClose"] = abs(dados_bitcoin["Low"] - dados_bitcoin["Close"].shift(1))
    dados_bitcoin["TrueRange"] = dados_bitcoin[["High-Low", "High-PrevClose", "Low-PrevClose"]].max(axis=1)
    dados_bitcoin["atr_{}".format(window)] = dados_bitcoin["TrueRange"].rolling(window=window, min_periods=1).mean()
    dados_bitcoin.drop(["High-Low", "High-PrevClose", "Low-PrevClose", "TrueRange"], axis=1, inplace=True)
    return dados_bitcoin

# Keltner Channels
def bitcoin_ind_keltner_channels(dados_bitcoin, period=20, multiplier=2):
    dados_bitcoin["TR"] = dados_bitcoin.apply(lambda row: max(row["High"] - row["Low"], abs(row["High"] - row["Close"]), abs(row["Low"] - row["Close"])), axis=1)
    dados_bitcoin["ATR"] = dados_bitcoin["TR"].rolling(window=period).mean()
    dados_bitcoin["Middle Band"] = dados_bitcoin["Close"].rolling(window=period).mean()
    dados_bitcoin["Upper Band"] = dados_bitcoin["Middle Band"] + multiplier * dados_bitcoin["ATR"]
    dados_bitcoin["Lower Band"] = dados_bitcoin["Middle Band"] - multiplier * dados_bitcoin["ATR"]
    return dados_bitcoin

# Chaikin Volatility
def bitcoin_ind_chaikin_volatility(dados_bitcoin, window=10):
    daily_returns = dados_bitcoin["Close"].pct_change()
    chaikin_volatility = daily_returns.rolling(window=window).std() * (252 ** 0.5)
    dados_bitcoin["Chaikin_Volatility_{}".format(window)] = chaikin_volatility
    return dados_bitcoin

# Standard Deviation 
def bitcoin_ind_stdev(dados_bitcoin, window=1): 
    stdev_column = dados_bitcoin["Close"].rolling(window=window).std()
    dados_bitcoin["Stdev_{}".format(window)] = stdev_column
    return dados_bitcoin

# Volatility Index (VIX)
def ta_vix(dados_bitcoin, window=21):
    returns = dados_bitcoin["Close"].pct_change().dropna()
    rolling_std = returns.rolling(window=window).std()
    vix = rolling_std * np.sqrt(252) * 100  
    dados_bitcoin["VIX_{}".format(window)] = vix
    return dados_bitcoin

# On-Balance Volume (OBV)
def bitcoin_ind_obv(dados_bitcoin, window=10):
    price_changes = dados_bitcoin["Close"].diff()
    volume_direction = pd.Series(1, index=price_changes.index)
    volume_direction[price_changes < 0] = -1
    obv = (dados_bitcoin["Volume"] * volume_direction).cumsum()
    obv_smoothed = obv.rolling(window=window).mean()
    dados_bitcoin["OBV_{}".format(window)] = obv_smoothed
    return dados_bitcoin

# Chaikin Money Flow (CMF)
def bitcoin_ind_chaikin_money_flow(dados_bitcoin, window=10):
    mf_multiplier = ((dados_bitcoin["Close"] - dados_bitcoin["Close"].shift(1)) + (dados_bitcoin["Close"] - dados_bitcoin["Close"].shift(1)).abs()) / 2
    mf_volume = mf_multiplier * dados_bitcoin["Volume"]
    adl = mf_volume.cumsum()
    cmf = adl.rolling(window=window).mean() / dados_bitcoin["Volume"].rolling(window=window).mean()
    dados_bitcoin["CMF_{}".format(window)] = cmf
    return dados_bitcoin

# Volume Price Trend (VPT)
def bitcoin_ind_volume_price_trend(dados_bitcoin, window=10):
    price_change = dados_bitcoin["Close"].pct_change()
    vpt = (price_change * dados_bitcoin["Volume"].shift(window)).cumsum()
    dados_bitcoin["VPT_{}".format(window)] = vpt
    return dados_bitcoin

# Accumulation/Distribution Line
def bitcoin_ind_accumulation_distribution_line(dados_bitcoin, window=10):
    money_flow_multiplier = ((dados_bitcoin["Close"] - dados_bitcoin["Close"].shift(1)) - (dados_bitcoin["Close"].shift(1) - dados_bitcoin["Close"])) / (dados_bitcoin["Close"].shift(1) - dados_bitcoin["Close"])
    money_flow_volume = money_flow_multiplier * dados_bitcoin["Volume"]
    ad_line = money_flow_volume.cumsum()
    ad_line_smoothed = ad_line.rolling(window=window, min_periods=1).mean()
    dados_bitcoin["A/D Line_{}".format(window)] = ad_line_smoothed
    return dados_bitcoin

# Ease of Movement (EOM)
def bitcoin_ind_ease_of_movement(dados_bitcoin, window=14):
    midpoint_move = ((dados_bitcoin["High"] + dados_bitcoin["Low"]) / 2).diff(1)
    box_ratio = dados_bitcoin["Volume"] / 1000000 / (dados_bitcoin["High"] - dados_bitcoin["Low"])
    eom = midpoint_move / box_ratio
    eom_smoothed = eom.rolling(window=window, min_periods=1).mean()
    dados_bitcoin["EOM_{}".format(window)] = eom_smoothed
    return dados_bitcoin
    

