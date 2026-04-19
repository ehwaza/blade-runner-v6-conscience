"""
RTM Trading System - Adaptive AI Trading with Recursive Reasoning
Système de trading adaptatif utilisant le Recursive Tiny Model
Version corrigée avec logging amélioré
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
from collections import deque
import socket
import threading
import time
import traceback

# Import du RTM (vos fichiers)
from models.layers import Attention, SwiGLU, rms_norm, RotaryEmbedding
from models.losses import ACTLossHead
from models.sparse_embedding import CastedSparseEmbedding


@dataclass
class MarketState:
    """État du marché à un instant T"""
    prices: List[float]  # OHLC récents
    volumes: List[float]
    indicators: Dict[str, float]  # RSI, MACD, etc.
    timestamp: int
    
    
@dataclass
class TradingSignal:
    """Signal de trading généré par le RTM"""
    action: str  # "BUY", "SELL", "HOLD", "CLOSE"
    confidence: float  # 0-1
    stop_loss: float
    take_profit: float
    reasoning_steps: int  # Nombre d'itérations du RTM
    features_importance: Dict[str, float]


class RecursiveTradingModel(nn.Module):
    """
    Modèle de trading récursif adaptatif
    S'adapte dynamiquement selon la complexité du marché
    """
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 4,
        max_thinking_steps: int = 8
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_steps = max_thinking_steps
        
        # Encodeur de features du marché
        self.market_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Transformer récursif pour raisonnement profond
        self.rotary_emb = RotaryEmbedding(
            dim=hidden_dim // num_heads,
            max_position_embeddings=max_thinking_steps,
            base=10000.0
        )
        
        self.recursive_attention = Attention(
            dim=hidden_dim,
            head_dim=hidden_dim // num_heads,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=True
        )
        
        self.reasoning_mlp = SwiGLU(hidden_dim, expansion=2.0)
        
        # Tête de décision: continuer à réfléchir ou agir?
        self.should_continue = nn.Linear(hidden_dim, 1)
        
        # Têtes de prédiction
        self.action_head = nn.Linear(hidden_dim, 4)  # BUY/SELL/HOLD/CLOSE
        self.confidence_head = nn.Linear(hidden_dim, 1)
        self.sl_tp_head = nn.Linear(hidden_dim, 2)  # Stop Loss, Take Profit
        
        # Mémoire des états de marché récents
        self.market_memory = deque(maxlen=100)
        
    def encode_market_state(self, state: MarketState) -> torch.Tensor:
        """Encode l'état du marché en vecteur"""
        # Normalisation des prix
        prices = torch.tensor(state.prices, dtype=torch.float32)
        prices_norm = (prices - prices.mean()) / (prices.std() + 1e-8)
        
        # Normalisation des volumes
        volumes = torch.tensor(state.volumes, dtype=torch.float32)
        volumes_norm = (volumes - volumes.mean()) / (volumes.std() + 1e-8)
        
        # Indicateurs techniques
        indicators = torch.tensor([
            state.indicators.get('rsi', 50) / 100,
            state.indicators.get('macd', 0) / 10,
            state.indicators.get('bb_position', 0.5),
            state.indicators.get('atr', 0) / 100,
            state.indicators.get('adx', 25) / 100,
            state.indicators.get('stoch', 50) / 100,
        ], dtype=torch.float32)
        
        # Patterns de chandeliers (simple)
        price_changes = torch.diff(prices_norm)
        volatility = torch.std(price_changes)
        trend = torch.sign(torch.mean(price_changes))
        
        # Concaténation de toutes les features
        features = torch.cat([
            prices_norm,
            volumes_norm,
            indicators,
            torch.tensor([volatility, trend])
        ])
        
        # Padding à input_dim
        if len(features) < 128:
            features = torch.nn.functional.pad(features, (0, 128 - len(features)))
        else:
            features = features[:128]
            
        return features.unsqueeze(0)  # [1, input_dim]
    
    def recursive_reasoning(self, market_features: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Raisonnement récursif adaptatif
        Continue à réfléchir jusqu'à être confiant ou atteindre max_steps
        """
        batch_size = market_features.size(0)
        
        # État initial
        hidden_state = self.market_encoder(market_features)  # [B, hidden_dim]
        thought_sequence = [hidden_state]
        
        cos_sin = self.rotary_emb()
        
        for step in range(self.max_steps):
            # Stack des pensées jusqu'ici
            thoughts = torch.stack(thought_sequence, dim=1)  # [B, S, D]
            
            # Attention sur l'historique de raisonnement
            attended = self.recursive_attention(cos_sin, thoughts)
            
            # Nouvelle pensée
            new_thought = attended[:, -1:, :]  # [B, 1, D]
            new_thought = rms_norm(new_thought + self.reasoning_mlp(new_thought), 1e-6)
            
            # Décision: continuer ou agir?
            continue_logit = self.should_continue(new_thought.squeeze(1))
            should_continue = torch.sigmoid(continue_logit) > 0.5
            
            thought_sequence.append(new_thought.squeeze(1))
            
            # Si confiant, arrêter le raisonnement
            if not should_continue.any() or step == self.max_steps - 1:
                final_thought = new_thought.squeeze(1)
                return final_thought, step + 1
        
        return thought_sequence[-1], self.max_steps
    
    def forward(self, state: MarketState) -> TradingSignal:
        """Génère un signal de trading adaptatif"""
        with torch.no_grad():
            # Encoder le marché
            market_features = self.encode_market_state(state)
            
            # Raisonnement récursif
            final_thought, steps_used = self.recursive_reasoning(market_features)
            
            # Décisions
            action_logits = self.action_head(final_thought)
            action_probs = torch.softmax(action_logits, dim=-1)
            action_idx = torch.argmax(action_probs, dim=-1).item()
            
            actions = ["BUY", "SELL", "HOLD", "CLOSE"]
            action = actions[action_idx]
            
            confidence = torch.sigmoid(self.confidence_head(final_thought)).item()
            
            # Stop Loss et Take Profit adaptatifs
            sl_tp = self.sl_tp_head(final_thought)
            sl_pips = abs(sl_tp[0, 0].item()) * 100  # 0-100 pips
            tp_pips = abs(sl_tp[0, 1].item()) * 200  # 0-200 pips
            
            # Ajustement selon la volatilité
            atr = state.indicators.get('atr', 50)
            sl_pips = max(sl_pips, atr * 0.5)
            tp_pips = max(tp_pips, atr * 1.5)
            
            # Importance des features (attention scores)
            features_importance = {
                'trend': float(action_probs[0, action_idx]),
                'volatility': state.indicators.get('atr', 0) / 100,
                'momentum': state.indicators.get('rsi', 50) / 100,
                'reasoning_depth': steps_used / self.max_steps
            }
            
            return TradingSignal(
                action=action,
                confidence=confidence,
                stop_loss=sl_pips,
                take_profit=tp_pips,
                reasoning_steps=steps_used,
                features_importance=features_importance
            )
    
    def update_from_trade(self, state: MarketState, signal: TradingSignal, 
                          outcome: float):
        """
        Apprentissage online à partir du résultat du trade
        outcome: profit/loss en pips
        """
        self.market_memory.append({
            'state': state,
            'signal': signal,
            'outcome': outcome
        })
        
        # Apprentissage par renforcement simple
        # Si assez de mémoire, réentraîner sur batch
        if len(self.market_memory) >= 32:
            self._online_learning_step()
    
    def _online_learning_step(self):
        """Étape d'apprentissage online"""
        # TODO: Implémenter PPO ou Q-learning adaptatif
        # Pour l'instant, simple mise à jour basée sur les résultats
        pass


class AdaptiveStrategyEngine:
    """
    Moteur de stratégies adaptatives
    Combine plusieurs approches et s'adapte selon le contexte
    """
    def __init__(self):
        self.strategies = {
            'trend_following': self.trend_following_strategy,
            'mean_reversion': self.mean_reversion_strategy,
            'breakout': self.breakout_strategy,
            'scalping': self.scalping_strategy
        }
        
        # Poids adaptatifs pour chaque stratégie
        self.strategy_weights = {k: 0.25 for k in self.strategies}
        self.strategy_performance = {k: deque(maxlen=50) for k in self.strategies}
        
    def select_best_strategy(self, state: MarketState) -> str:
        """Sélectionne la meilleure stratégie selon le contexte"""
        # Analyse du marché
        volatility = state.indicators.get('atr', 50)
        trend_strength = state.indicators.get('adx', 25)
        rsi = state.indicators.get('rsi', 50)
        
        scores = {}
        
        # Trend following: bon en forte tendance
        if trend_strength > 25:
            scores['trend_following'] = trend_strength / 50
        
        # Mean reversion: bon en range
        if volatility < 30 and 30 < rsi < 70:
            scores['mean_reversion'] = 1 - (volatility / 100)
        
        # Breakout: bon en consolidation avant breakout
        if volatility < 20 and trend_strength < 20:
            scores['breakout'] = 0.8
        
        # Scalping: bon en haute volatilité
        if volatility > 40:
            scores['scalping'] = volatility / 100
        
        # Combine avec performance historique
        for strategy, score in scores.items():
            if self.strategy_performance[strategy]:
                avg_perf = np.mean(list(self.strategy_performance[strategy]))
                scores[strategy] = score * 0.7 + avg_perf * 0.3
        
        return max(scores.items(), key=lambda x: x[1])[0] if scores else 'trend_following'
    
    def trend_following_strategy(self, state: MarketState) -> Dict:
        """Stratégie de suivi de tendance"""
        prices = state.prices
        ma_fast = np.mean(prices[-10:])
        ma_slow = np.mean(prices[-30:])
        rsi = state.indicators.get('rsi', 50)
        
        if ma_fast > ma_slow and rsi < 70:
            return {'action': 'BUY', 'strength': 0.8}
        elif ma_fast < ma_slow and rsi > 30:
            return {'action': 'SELL', 'strength': 0.8}
        else:
            return {'action': 'HOLD', 'strength': 0.3}
    
    def mean_reversion_strategy(self, state: MarketState) -> Dict:
        """Stratégie de retour à la moyenne"""
        rsi = state.indicators.get('rsi', 50)
        bb_pos = state.indicators.get('bb_position', 0.5)
        
        if rsi < 30 and bb_pos < 0.2:
            return {'action': 'BUY', 'strength': 0.85}
        elif rsi > 70 and bb_pos > 0.8:
            return {'action': 'SELL', 'strength': 0.85}
        else:
            return {'action': 'HOLD', 'strength': 0.2}
    
    def breakout_strategy(self, state: MarketState) -> Dict:
        """Stratégie de cassure"""
        prices = state.prices
        high = max(prices[-20:])
        low = min(prices[-20:])
        current = prices[-1]
        volume = state.volumes[-1]
        avg_volume = np.mean(state.volumes[-20:])
        
        if current > high * 1.001 and volume > avg_volume * 1.5:
            return {'action': 'BUY', 'strength': 0.9}
        elif current < low * 0.999 and volume > avg_volume * 1.5:
            return {'action': 'SELL', 'strength': 0.9}
        else:
            return {'action': 'HOLD', 'strength': 0.1}
    
    def scalping_strategy(self, state: MarketState) -> Dict:
        """Stratégie de scalping"""
        prices = state.prices
        short_ma = np.mean(prices[-5:])
        current = prices[-1]
        atr = state.indicators.get('atr', 50)
        
        if current > short_ma * 1.0005 and atr > 30:
            return {'action': 'BUY', 'strength': 0.6}
        elif current < short_ma * 0.9995 and atr > 30:
            return {'action': 'SELL', 'strength': 0.6}
        else:
            return {'action': 'CLOSE', 'strength': 0.8}


class MQL5Bridge:
    """Pont de communication avec MQL5 - Version améliorée avec logging"""
    def __init__(self, host='localhost', port=9091, trading_system=None):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        self.trading_system = trading_system
        self.connection_count = 0
        
    def start(self):
        """Démarre le serveur pour MQL5"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            self.socket.settimeout(1.0)  # Timeout pour permettre la vérification de running
            self.running = True
            
            print(f"🚀 RTM Trading Bridge démarré sur {self.host}:{self.port}")
            print(f"📡 En écoute active - Prêt pour les connexions...")
            
            # Démarrer le thread d'écoute
            listener_thread = threading.Thread(target=self._listen, daemon=True, name="BridgeListener")
            listener_thread.start()
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur démarrage serveur: {e}")
            return False
    
    def _listen(self):
        """Écoute les requêtes MQL5 avec logging amélioré"""
        print(f"🎯 Thread d'écoute démarré sur le port {self.port}")
        
        while self.running:
            try:
                client, addr = self.socket.accept()
                self.connection_count += 1
                client_id = self.connection_count
                
                print(f"\n📡 CONNEXION #{client_id} reçue de {addr[0]}:{addr[1]}")
                print(f"   ⏰ Heure: {time.strftime('%H:%M:%S')}")
                
                # Démarrer le thread de traitement
                client_thread = threading.Thread(
                    target=self._handle_client, 
                    args=(client, addr, client_id),
                    daemon=True,
                    name=f"ClientHandler-{client_id}"
                )
                client_thread.start()
                
            except socket.timeout:
                # Timeout normal pour vérifier si running
                continue
            except Exception as e:
                if self.running:
                    print(f"⚠️  Erreur accept(): {e}")
    
    def _handle_client(self, client, addr, client_id):
        """Traite une requête client avec logging détaillé"""
        try:
            client.settimeout(10.0)  # Timeout pour la lecture
            
            # Réception des données
            print(f"   📥 Réception données du client #{client_id}...")
            data = client.recv(4096).decode('utf-8')
            
            if not data:
                print(f"   ⚠️  Client #{client_id} a fermé la connexion")
                return
                
            request = json.loads(data)
            print(f"   📊 Données reçues: {len(request.get('prices', []))} prix, "
                  f"RSI: {request.get('indicators', {}).get('rsi', 'N/A')}")
            
            # Traitement de la requête
            response = self.process_request(request)
            
            # Log du signal généré
            print(f"   🎯 Signal #{client_id}: {response.get('action', 'N/A')} "
                  f"(confiance: {response.get('confidence', 0):.1%})")
            
            # Envoi de la réponse
            client.send(json.dumps(response).encode('utf-8'))
            print(f"   ✅ Réponse envoyée au client #{client_id}")
            
        except json.JSONDecodeError as e:
            print(f"   ❌ Erreur JSON client #{client_id}: {e}")
            error_response = {
                'action': 'HOLD',
                'confidence': 0.0,
                'error': 'Invalid JSON format'
            }
            client.send(json.dumps(error_response).encode('utf-8'))
        except socket.timeout:
            print(f"   ⏰ Timeout client #{client_id}")
        except Exception as e:
            print(f"   ❌ Erreur client #{client_id}: {e}")
            traceback.print_exc()
        finally:
            client.close()
            print(f"   🔌 Connexion fermée avec client #{client_id}")
    
    def process_request(self, request: Dict) -> Dict:
        """Traite une requête de signal"""
        try:
            # Parse les données de marché
            prices = request.get('prices', [])
            indicators = request.get('indicators', {})
            
            if not prices or not indicators:
                return {'error': 'Invalid request format', 'action': 'HOLD', 'confidence': 0.0}
            
            # Créer un MarketState
            state = MarketState(
                prices=prices[-50:] if len(prices) >= 50 else prices,
                volumes=request.get('volumes', [1000] * min(len(prices), 50)),
                indicators=indicators,
                timestamp=int(time.time())
            )
            
            # Générer le signal avec le système
            if self.trading_system:
                signal = self.trading_system.generate_signal(state)
            else:
                # Fallback: utiliser juste le modèle
                model = RecursiveTradingModel()
                signal = model(state)
            
            # Formater la réponse en JSON
            response = {
                'action': signal.action,
                'confidence': float(signal.confidence),
                'stop_loss': float(signal.stop_loss),
                'take_profit': float(signal.take_profit),
                'strategy': getattr(signal, 'strategy', 'RTM'),
                'reasoning_steps': signal.reasoning_steps,
                'timestamp': int(time.time())
            }
            
            return response
            
        except Exception as e:
            print(f"❌ Erreur traitement requête: {e}")
            traceback.print_exc()
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'stop_loss': 50.0,
                'take_profit': 100.0,
                'error': str(e)
            }
    
    def stop(self):
        """Arrête le serveur"""
        print("🛑 Arrêt du bridge MQL5...")
        self.running = False
        if self.socket:
            self.socket.close()
        print("✅ Bridge MQL5 arrêté")


class RTMTradingSystem:
    """Système de trading complet avec RTM - Version améliorée"""
    def __init__(self):
        self.model = RecursiveTradingModel()
        self.strategy_engine = AdaptiveStrategyEngine()
        self.bridge = MQL5Bridge(trading_system=self)
        
        print("✅ RTM Trading System initialisé")
        print("📊 Stratégies: trend_following, mean_reversion, breakout, scalping")
        print("🧠 Mode: Apprentissage adaptatif online")
        print(f"🔧 Modèle: {sum(p.numel() for p in self.model.parameters()):,} paramètres")
    
    def generate_signal(self, state: MarketState) -> TradingSignal:
        """Génère un signal de trading intelligent"""
        # Signal du RTM
        rtm_signal = self.model(state)
        
        # Sélection de la meilleure stratégie
        best_strategy = self.strategy_engine.select_best_strategy(state)
        strategy_signal = self.strategy_engine.strategies[best_strategy](state)
        
        # Fusion des signaux (ensemble learning)
        if rtm_signal.confidence > 0.7:
            # Haute confiance RTM: suivre le RTM
            final_signal = rtm_signal
            final_signal.features_importance['strategy'] = 'RTM'
        elif strategy_signal['strength'] > 0.8:
            # Haute confiance stratégie: utiliser stratégie
            final_signal = TradingSignal(
                action=strategy_signal['action'],
                confidence=strategy_signal['strength'],
                stop_loss=rtm_signal.stop_loss,
                take_profit=rtm_signal.take_profit,
                reasoning_steps=0,
                features_importance={'strategy': best_strategy}
            )
        else:
            # Combiner les deux
            if rtm_signal.action == strategy_signal['action']:
                final_signal = rtm_signal
                final_signal.confidence = (rtm_signal.confidence + strategy_signal['strength']) / 2
                final_signal.features_importance['strategy'] = f"RTM+{best_strategy}"
            else:
                final_signal = TradingSignal(
                    action='HOLD',
                    confidence=0.5,
                    stop_loss=rtm_signal.stop_loss,
                    take_profit=rtm_signal.take_profit,
                    reasoning_steps=rtm_signal.reasoning_steps,
                    features_importance={'conflict': True, 'rtm_action': rtm_signal.action, 'strategy_action': strategy_signal['action']}
                )
        
        return final_signal
    
    def run(self):
        """Lance le système de trading avec monitoring"""
        if not self.bridge.start():
            print("❌ Impossible de démarrer le bridge MQL5")
            return
        
        print("\n" + "="*60)
        print("🎯 SYSTÈME RTM TRADING OPÉRATIONNEL")
        print("="*60)
        print("📡 En attente de connexions clients...")
        print("💡 Utilisez Ctrl+C pour arrêter le système")
        print("="*60)
        
        try:
            # Boucle principale avec monitoring
            last_status_time = time.time()
            while True:
                # Afficher le statut toutes les 30 secondes
                current_time = time.time()
                if current_time - last_status_time > 30:
                    print(f"📊 Statut: Serveur actif - {self.bridge.connection_count} connexions traitées - {time.strftime('%H:%M:%S')}")
                    last_status_time = current_time
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Arrêt du système demandé...")
        except Exception as e:
            print(f"\n❌ Erreur système: {e}")
            traceback.print_exc()
        finally:
            self.bridge.stop()
            print("✅ Système RTM Trading arrêté proprement")


# Point d'entrée
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                   RTM TRADING SYSTEM v2.0                    ║
    ║              Système de Trading Adaptatif AI                 ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    system = RTMTradingSystem()
    system.run()