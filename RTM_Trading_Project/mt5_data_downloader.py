"""
Téléchargeur avancé de données MT5 avec plus d'options
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import argparse
import sys

class MT5DataDownloader:
    def __init__(self):
        self.connected = False
        
    def connect(self):
        """Connexion à MT5"""
        try:
            if not mt5.initialize():
                print("❌ Échec de l'initialisation MT5")
                print("💡 Vérifiez que MetaTrader 5 est installé et ouvert")
                return False
            
            self.connected = True
            print("✅ Connecté à MetaTrader 5")
            print(f"📊 Version MT5: {mt5.version()}")
            return True
            
        except Exception as e:
            print(f"❌ Erreur connexion: {e}")
            return False
    
    def get_available_symbols(self):
        """Liste les symboles disponibles"""
        symbols = mt5.symbols_get()
        forex_symbols = [s.name for s in symbols if "USD" in s.name or "EUR" in s.name or "GBP" in s.name]
        return forex_symbols[:20]  # Premiers 20 symboles Forex
    
    def download_data(self, symbol, timeframe_str, years=1):
        """Télécharge les données pour un symbole"""
        # Mapping des timeframes
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1
        }
        
        timeframe = timeframe_map.get(timeframe_str.upper(), mt5.TIMEFRAME_H1)
        
        # Vérifier si le symbole existe
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"❌ Symbole {symbol} non trouvé")
            return None
        
        # Calcul des dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        print(f"📥 Téléchargement {symbol} ({timeframe_str}) - {years} an(s)")
        
        # Téléchargement
        rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        
        if rates is None or len(rates) == 0:
            print(f"❌ Aucune donnée pour {symbol}")
            return None
        
        # Conversion en DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Formatage
        df = df.rename(columns={
            'time': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low', 
            'close': 'close',
            'tick_volume': 'volume'
        })[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        print(f"✅ {len(df)} bougies téléchargées")
        return df
    
    def close(self):
        """Fermeture de la connexion"""
        if self.connected:
            mt5.shutdown()
            print("🔌 Déconnecté de MT5")

def main():
    parser = argparse.ArgumentParser(description='Téléchargeur de données MT5')
    parser.add_argument('--symbol', type=str, default='EURUSD', help='Symbole (ex: EURUSD)')
    parser.add_argument('--timeframe', type=str, default='H1', help='Timeframe (M1,M5,M15,M30,H1,H4,D1,W1)')
    parser.add_argument('--years', type=int, default=1, help='Nombre d\'années de données')
    parser.add_argument('--list-symbols', action='store_true', help='Lister les symboles disponibles')
    
    args = parser.parse_args()
    
    downloader = MT5DataDownloader()
    
    try:
        if not downloader.connect():
            sys.exit(1)
        
        if args.list_symbols:
            symbols = downloader.get_available_symbols()
            print("\n📋 Symboles Forex disponibles:")
            for i, symbol in enumerate(symbols, 1):
                print(f"   {i:2d}. {symbol}")
            return
        
        # Téléchargement des données
        df = downloader.download_data(args.symbol, args.timeframe, args.years)
        
        if df is not None:
            filename = f"{args.symbol.lower()}_{args.timeframe.lower()}_data.csv"
            df.to_csv(filename, index=False)
            
            print(f"\n💾 Fichier créé: {filename}")
            print(f"📈 Statistiques:")
            print(f"   - Période: {df['timestamp'].min()} -> {df['timestamp'].max()}")
            print(f"   - Bougies: {len(df)}")
            print(f"   - Prix moyen: {df['close'].mean():.5f}")
            print(f"   - Volatilité: {(df['high'] - df['low']).mean():.5f}")
            
            # Créer aussi le fichier forex_data.csv pour l'entraînement
            df.to_csv("forex_data.csv", index=False)
            print(f"📁 forex_data.csv créé pour l'entraînement RTM")
    
    finally:
        downloader.close()

if __name__ == "__main__":
    main()