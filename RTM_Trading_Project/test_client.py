import socket
import json
import time

def test_rtm_server():
    print("🧪 TEST DU SERVEUR RTM EN TEMPS RÉEL")
    print("=" * 50)
    
    try:
        # Connexion au serveur
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(10)
        client.connect(('localhost', 9091))
        print("✅ Connecté au serveur RTM sur le port 9091")
        
        # Données de marché réalistes
        test_data = {
            'prices': [1.0850, 1.0852, 1.0855, 1.0858, 1.0860, 1.0862, 1.0865],
            'volumes': [1050, 1120, 1250, 1180, 1320, 1280, 1400],
            'indicators': {
                'rsi': 58.0,
                'macd': 0.0023,
                'atr': 22.5,
                'adx': 32.0,
                'bb_position': 0.62,
                'stoch': 65.0
            }
        }
        
        print("📊 Envoi des données de marché au serveur...")
        client.send(json.dumps(test_data).encode('utf-8'))
        
        # Réception du signal
        response = client.recv(4096).decode('utf-8')
        signal = json.loads(response)
        
        print("\n🎯 SIGNAL DE TRADING GÉNÉRÉ:")
        print("═" * 40)
        print(f"📍 Action: {signal.get('action', 'N/A')}")
        print(f"📈 Confiance: {signal.get('confidence', 0):.2%}")
        print(f"🛑 Stop Loss: {signal.get('stop_loss', 0):.1f} pips")
        print(f"🎯 Take Profit: {signal.get('take_profit', 0):.1f} pips")
        print(f"🧠 Stratégie: {signal.get('strategy', 'RTM')}")
        print(f"⚡ Étapes raisonnement: {signal.get('reasoning_steps', 0)}")
        
        if 'error' in signal:
            print(f"⚠️  Erreur: {signal['error']}")
            
        client.close()
        print("\n✅ Test terminé avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur de connexion: {e}")

if __name__ == "__main__":
    test_rtm_server()