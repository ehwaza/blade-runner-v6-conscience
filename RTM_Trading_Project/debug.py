"""
Debug des dimensions du modèle RTM
"""

import torch
import torch.nn as nn
from rtm_trading_system import RecursiveTradingModel, MarketState
import numpy as np

print("🔍 DEBUG DES DIMENSIONS RTM")
print("=" * 50)

# Créer un modèle avec logging des dimensions
model = RecursiveTradingModel(
    input_dim=128,
    hidden_dim=256, 
    num_heads=4,
    max_thinking_steps=8
)

# Données de test
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

print("1. Encodage des features du marché...")
market_features = model.encode_market_state(state)
print(f"   market_features shape: {market_features.shape}")  # Devrait être [1, 128]

print("2. Passage par l'encodeur de marché...")
hidden_state = model.market_encoder(market_features)
print(f"   hidden_state shape: {hidden_state.shape}")  # Devrait être [1, 256]

print("3. Test de l'attention récursive...")
# Simuler quelques étapes de raisonnement
thought_sequence = [hidden_state]
cos_sin = model.rotary_emb()

for step in range(2):  # Juste 2 étapes pour le debug
    print(f"   Étape {step + 1}:")
    thoughts = torch.stack(thought_sequence, dim=1)
    print(f"      thoughts shape: {thoughts.shape}")  # [B, S, D]
    
    # Test de l'attention
    try:
        attended = model.recursive_attention(cos_sin, thoughts)
        print(f"      attended shape: {attended.shape}")
        
        # Nouvelle pensée
        new_thought = attended[:, -1:, :]
        print(f"      new_thought shape: {new_thought.shape}")
        
        thought_sequence.append(new_thought.squeeze(1))
        print(f"      ✅ Étape {step + 1} réussie!")
        
    except Exception as e:
        print(f"      ❌ Erreur à l'étape {step + 1}: {e}")
        import traceback
        traceback.print_exc()
        break

print("4. Vérification des dimensions de l'attention:")
print(f"   hidden_dim: {model.hidden_dim}")
print(f"   num_heads: {4}")  # Fixé à 4 comme dans l'initialisation
print(f"   head_dim: {64}")  # 256 / 4 = 64

print("\n5. Test de génération de signal complet...")
try:
    with torch.no_grad():
        signal = model(state)
    print(f"   ✅ Signal généré avec succès!")
    print(f"   Action: {signal.action}")
    print(f"   Confiance: {signal.confidence:.2%}")
except Exception as e:
    print(f"   ❌ Erreur génération signal: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Debug terminé")