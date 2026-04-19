"""
Test du serveur RTM sans MQL5
"""

import socket
import json
import time
from rtm_trading_system import RTMTradingSystem

def test_server():
    print("🧪 TEST DU SERVEUR RTM")
    print("=" * 50)
    
    # Démarrer le système sur un port différent
    system = RTMTradingSystem()
    system.bridge.port = 9092  # Utiliser un port différent
    
    try:
        # Démarrer le serveur dans un thread séparé
        import threading
        server_thread = threading.Thread(target=system.run, daemon=True)
        server_thread.start()
        
        print("⏳ Attente du démarrage du serveur...")
        time.sleep(2)
        
        # Tester la connexion
        print("🔌 Test de connexion au serveur...")
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(5)
        client.connect(('localhost', 9092))
        
        # Envoyer une requête de test
        test_request = {
            'prices': [1.0950, 1.0951, 1.0952, 1.0950, 1.0948],
            'volumes': [1000, 1200, 800, 1500, 900],
            'indicators': {
                'rsi': 45.0,
                'macd': 0.0012,
                'atr': 25.5,
                'adx': 28.0,
                'bb_position': 0.45,
                'stoch': 52.0
            }
        }
        
        client.send(json.dumps(test_request).encode('utf-8'))
        
        # Recevoir la réponse
        response = client.recv(4096).decode('utf-8')
        signal = json.loads(response)
        
        print("✅ Réponse du serveur reçue!")
        print(f"📊 Signal: {signal}")
        
        client.close()
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
    
    print("🧪 Test terminé")

if __name__ == "__main__":
    test_server()