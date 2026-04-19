# RTM Trading Project

## Structure du projet

```
RTM_Trading_Project/
├── models/
│   ├── __init__.py           # Package models
│   ├── common.py             # Fonctions communes (trunc_normal_init)
│   ├── ema.py                # Exponential Moving Average
│   ├── layers.py             # Couches du transformer (Attention, SwiGLU, etc.)
│   ├── losses.py             # Fonctions de loss
│   └── sparse_embedding.py   # Embeddings spécialisés
│
├── rtm_trading_system.py     # Système principal (à copier depuis artifacts)
├── rtm_training.py           # Script d'entraînement (à copier depuis artifacts)
├── rtm_test_simple.py        # Tests unitaires (à copier depuis artifacts)
└── README.md                 # Ce fichier
```

## Installation

1. **Dépendances Python:**
```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib tqdm einops
```

2. **Fichiers manquants:**
Les fichiers suivants doivent être copiés depuis les artifacts Claude:
- `rtm_trading_system.py`
- `rtm_training.py`
- `rtm_test_simple.py`

3. **Bot MQL5:**
Le fichier `RTM_Adaptive_Bot.mq5` doit être placé dans:
```
C:/Users/[User]/AppData/Roaming/MetaQuotes/Terminal/[ID]/MQL5/Experts/
```

## Utilisation rapide

### Test du système
```bash
python rtm_test_simple.py
```

### Lancement du serveur
```bash
python rtm_trading_system.py
```

### Entraînement
```bash
python rtm_training.py
```

## Notes

- Tous les fichiers `models/` sont maintenant créés
- Assurez-vous d'avoir PyTorch installé
- Pour le trading réel, utilisez d'abord un compte démo

## Support

Ce projet utilise un Recursive Tiny Model (RTM) pour le trading adaptatif.
