# Scénario haussier
bullish_data = {
    'prices': [1.0900, 1.0910, 1.0920, 1.0930, 1.0940, 1.0950],
    'volumes': [1000, 1200, 1500, 1800, 2000, 2200],
    'indicators': {
        'rsi': 65.0,  # Sur-acheté mais trending
        'macd': 0.005,
        'atr': 15.0,
        'adx': 40.0,  # Trend fort
        'bb_position': 0.7,
        'stoch': 80.0
    }
}

# Scénario baissier  
bearish_data = {
    'prices': [1.1000, 1.0990, 1.0980, 1.0970, 1.0960, 1.0950],
    'volumes': [1000, 1300, 1600, 1900, 2100, 2300],
    'indicators': {
        'rsi': 35.0,  # Sur-vendu mais trending
        'macd': -0.005,
        'atr': 18.0,
        'adx': 38.0,  # Trend fort
        'bb_position': 0.3,
        'stoch': 20.0
    }
}