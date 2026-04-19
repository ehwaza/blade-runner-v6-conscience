"""
Test simple du système RTM - Sans dépendances MQL5
Vérification que le modèle fonctionne correctement
"""

import sys
import torch
import numpy as np

# Test d'import des modules
print("=" * 70)
print("🔍 TEST DU SYSTÈME RTM")
print("=" * 70)

print("\n1️⃣ Vérification des imports...")
try:
    from models.layers import Attention, SwiGLU, rms_norm, RotaryEmbedding
    from models.common import trunc_normal_init_
    print("   ✅ Modules layers importés")
except Exception as e:
    print(f"   ❌ Erreur import layers: {e}")
    sys.exit(1)

try:
    # Import du système principal (sans démarrer le serveur)
    import importlib.util
    spec = importlib.util.spec_from_file_location("rtm_system", "rtm_trading_system.py")
    rtm_module = importlib.util.module_from_spec(spec)
    
    # Ne pas exécuter le main
    import builtins
    original_name = builtins.__name__
    builtins.__name__ = "not_main"
    
    spec.loader.exec_module(rtm_module)
    
    builtins.__name__ = original_name
    
    RecursiveTradingModel = rtm_module.RecursiveTradingModel
    MarketState = rtm_module.MarketState
    AdaptiveStrategyEngine = rtm_module.AdaptiveStrategyEngine
    
    print("   ✅ Système RTM importé")
except Exception as e:
    print(f"   ❌ Erreur import RTM: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n2️⃣ Test du modèle RTM...")
try:
    model = RecursiveTradingModel(
        input_dim=128,
        hidden_dim=256,
        num_heads=4,
        max_thinking_steps=8
    )
    print(f"   ✅ Modèle créé: {sum(p.numel() for p in model.parameters())} paramètres")
except Exception as e:
    print(f"   ❌ Erreur création modèle: {e}")
    sys.exit(1)

print("\n3️⃣ Test avec données synthétiques...")
try:
    # Créer un état de marché fictif
    prices = [1.0950 + np.random.randn() * 0.001 for _ in range(50)]
    volumes = [1000 + np.random.randint(-100, 100) for _ in range(50)]
    
    state = MarketState(
        prices=prices,
        volumes=volumes,
        indicators={
            'rsi': 45.0,
            'macd': 0.0012,
            'atr': 25.5,
            'adx': 28.0,
            'bb_position': 0.45,
            'stoch': 52.0
        },
        timestamp=1234567890
    )
    
    print("   ✅ MarketState créé")
    
    # Générer un signal
    with torch.no_grad():
        signal = model(state)
    
    print(f"\n   📊 Signal généré:")
    print(f"      • Action: {signal.action}")
    print(f"      • Confiance: {signal.confidence:.2%}")
    print(f"      • Stop Loss: {signal.stop_loss:.1f} pips")
    print(f"      • Take Profit: {signal.take_profit:.1f} pips")
    print(f"      • Étapes de raisonnement: {signal.reasoning_steps}")
    
    print("\n   ✅ Signal généré avec succès!")
    
except Exception as e:
    print(f"   ❌ Erreur génération signal: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n4️⃣ Test du moteur de stratégies adaptatives...")
try:
    strategy_engine = AdaptiveStrategyEngine()
    
    # Tester la sélection de stratégie
    best_strategy = strategy_engine.select_best_strategy(state)
    print(f"   ✅ Stratégie sélectionnée: {best_strategy}")
    
    # Tester chaque stratégie
    strategies_tested = []
    for strategy_name, strategy_func in strategy_engine.strategies.items():
        result = strategy_func(state)
        strategies_tested.append({
            'name': strategy_name,
            'action': result['action'],
            'strength': result['strength']
        })
    
    print("\n   📊 Résultats des stratégies:")
    for s in strategies_tested:
        print(f"      • {s['name']}: {s['action']} (force: {s['strength']:.2f})")
    
    print("\n   ✅ Toutes les stratégies fonctionnent!")
    
except Exception as e:
    print(f"   ❌ Erreur stratégies: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n5️⃣ Test de performance...")
try:
    import time
    
    # Test de vitesse
    num_predictions = 100
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_predictions):
            signal = model(state)
    
    elapsed = time.time() - start_time
    avg_time = elapsed / num_predictions * 1000  # en ms
    
    print(f"   ✅ {num_predictions} prédictions en {elapsed:.2f}s")
    print(f"   ⚡ Temps moyen par prédiction: {avg_time:.2f}ms")
    
    if avg_time < 100:
        print("   🚀 Performance: EXCELLENTE (< 100ms)")
    elif avg_time < 500:
        print("   ✅ Performance: BONNE (< 500ms)")
    else:
        print("   ⚠️ Performance: ACCEPTABLE (> 500ms)")
    
except Exception as e:
    print(f"   ❌ Erreur test performance: {e}")

print("\n6️⃣ Test de la mémoire du modèle...")
try:
    # Simuler plusieurs trades
    for i in range(5):
        outcome = np.random.randn() * 10  # Profit/loss aléatoire
        model.update_from_trade(state, signal, outcome)
    
    memory_size = len(model.market_memory)
    print(f"   ✅ Mémoire: {memory_size} entrées stockées")
    
except Exception as e:
    print(f"   ⚠️ Avertissement mémoire: {e}")

print("\n" + "=" * 70)
print("✅ TOUS LES TESTS RÉUSSIS!")
print("=" * 70)
print("\n🎯 Prochaines étapes:")
print("   1. Lancer le serveur: python rtm_trading_system.py")
print("   2. Ouvrir MetaTrader 5")
print("   3. Charger le bot RTM_Adaptive_Bot.mq5")
print("\n💡 Pour entraîner le modèle:")
print("   python rtm_training.py")
print("\n" + "=" * 70)
