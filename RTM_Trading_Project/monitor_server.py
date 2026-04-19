"""
Monitoring du serveur RTM
"""

import socket
import time
import threading

def monitor_server():
    """Affiche l'état du serveur en temps réel"""
    while True:
        try:
            # Test de connexion
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.settimeout(2)
            client.connect(('localhost', 9091))
            client.close()
            print(f"✅ Serveur ACTIF - Port 9091 ouvert - {time.strftime('%H:%M:%S')}")
        except:
            print(f"❌ Serveur INACCESSIBLE - {time.strftime('%H:%M:%S')}")
        
        time.sleep(5)  # Vérifier toutes les 5 secondes

def send_test_request():
    """Envoie une requête test périodiquement"""
    time.sleep(3)  # Attendre que le monitoring démarre
    
    test_data = {
        'prices': [1.0850, 1.0852, 1.0855],
        'volumes': [1000, 1200, 1500],
        'indicators': {'rsi': 55.0, 'macd': 0.002, 'atr': 20.0}
    }
    
    while True:
        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.connect(('localhost', 9091))
            client.send(json.dumps(test_data).encode('utf-8'))
            response = client.recv(4096).decode('utf-8')
            signal = json.loads(response)
            print(f"🧪 Test auto: {signal['action']} ({signal['confidence']:.1%})")
            client.close()
        except Exception as e:
            print(f"🧪 Test échoué: {e}")
        
        time.sleep(30)  Tester toutes les 30 secondes

if __name__ == "__main__":
    print("📊 MONITORING SERVEUR RTM")
    print("=" * 40)
    
    # Démarrer le monitoring
    monitor_thread = threading.Thread(target=monitor_server, daemon=True)
    monitor_thread.start()
    
    # Démarrer les tests automatiques
    test_thread = threading.Thread(target=send_test_request, daemon=True)
    test_thread.start()
    
    # Maintenir le script actif
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Monitoring arrêté")