"""
Téléchargement de données historiques depuis MetaTrader 5
"""

import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import os

def setup_mt5():
    """Initialise la connexion à MT5"""
    if not mt5.initialize():
        print("❌ Échec de l'initialisation de MT5")
        return False
    
    print("✅ MT5 initialisé avec succès")
    return True

def download_historical_data(symbol="EURUSD", timeframe=mt5.TIMEFRAME_H1, years=2):
    """Télécharge les données historiques"""
    print(f"📥 Téléchargement des données {symbol}...")
    
    # Calcul des dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    
    # Téléchargement des données
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    
    if rates is None:
        print(f"❌ Aucune donnée trouvée pour {symbol}")
        return None
    
    # Conversion en DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Renommage des colonnes
    df.rename(columns={
        'time': 'timestamp',
        'open': 'open',
        'high': 'high', 
        'low': 'low',
        'close': 'close',
        'tick_volume': 'volume'
    }, inplace=True)
    
    # Sélection des colonnes nécessaires
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    print(f"✅ {len(df)} bougies téléchargées")
    print(f"📅 Période: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    
    return df

def main():
    """Fonction principale"""
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║              TÉLÉCHARGEMENT DONNÉES MT5              ║
    ╚══════════════════════════════════════════════════════╝
    """)
    
    # Initialisation MT5
    if not setup_mt5():
        return
    
    # Symboles à télécharger
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
    timeframe = mt5.TIMEFRAME_H1  # 1 heure
    years = 2  # 2 ans de données
    
    for symbol in symbols:
        try:
            print(f"\n🔍 Traitement de {symbol}...")
            df = download_historical_data(symbol, timeframe, years)
            
            if df is not None:
                # Sauvegarde
                filename = f"{symbol.lower()}_data.csv"
                df.to_csv(filename, index=False)
                print(f"💾 Données sauvegardées: {filename}")
                
                # Aperçu
                print(f"   Première bougie: {df['timestamp'].iloc[0]}")
                print(f"   Dernière bougie: {df['timestamp'].iloc[-1]}")
                print(f"   Prix actuel: {df['close'].iloc[-1]:.5f}")
                
        except Exception as e:
            print(f"❌ Erreur avec {symbol}: {e}")
    
    # Fermeture MT5
    mt5.shutdown()
    print("\n✅ Téléchargement terminé")

if __name__ == "__main__":
    main()