"""
RTM Training System - Entraînement du modèle sur données historiques
Apprentissage par renforcement pour le trading adaptatif
Version corrigée pour gérer les objets personnalisés
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from pathlib import Path
import time

# Import du modèle RTM
from rtm_trading_system import RecursiveTradingModel, MarketState, TradingSignal


class TradingDataset(Dataset):
    """Dataset pour l'entraînement du RTM - Version corrigée"""
    def __init__(self, csv_file: str, lookback: int = 100):
        """
        Args:
            csv_file: Fichier CSV avec colonnes: timestamp, open, high, low, close, volume
            lookback: Nombre de bougies à regarder en arrière
        """
        self.data = pd.read_csv(csv_file)
        self.lookback = lookback
        
        # Convertir le timestamp en datetime si nécessaire
        if 'timestamp' in self.data.columns:
            if self.data['timestamp'].dtype == 'object':
                try:
                    self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                    print("✅ Timestamps convertis en datetime")
                except:
                    print("⚠️  Impossible de convertir les timestamps")
        
        # Calcul des indicateurs
        self._compute_indicators()
        
        print(f"📊 Dataset chargé: {len(self.data)} bougies")
        if 'timestamp' in self.data.columns:
            print(f"📅 Période: {self.data['timestamp'].iloc[0]} -> {self.data['timestamp'].iloc[-1]}")
    
    def _compute_indicators(self):
        """Calcule les indicateurs techniques"""
        df = self.data
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['atr'] = ranges.max(axis=1).rolling(14).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ADX (simplifié)
        df['adx'] = df['atr'].rolling(14).mean() / df['close'] * 100
        
        # Stochastic
        df['stoch'] = ((df['close'] - df['low'].rolling(14).min()) / 
                       (df['high'].rolling(14).max() - df['low'].rolling(14).min())) * 100
        
        # Remplir les NaN
        df = df.bfill()
        df = df.ffill()
        
        self.data = df
    
    def __len__(self):
        return len(self.data) - self.lookback - 10
    
    def __getitem__(self, idx):
        """Retourne un état de marché et le résultat futur - Version corrigée"""
        start_idx = idx
        end_idx = idx + self.lookback
        
        # État du marché
        state_data = self.data.iloc[start_idx:end_idx]
        
        # Gestion du timestamp - conversion en int si nécessaire
        if 'timestamp' in state_data.columns:
            last_timestamp = state_data['timestamp'].iloc[-1]
            if isinstance(last_timestamp, (pd.Timestamp, str)):
                # Convertir en timestamp UNIX
                if isinstance(last_timestamp, str):
                    last_timestamp = pd.to_datetime(last_timestamp)
                timestamp_int = int(last_timestamp.timestamp())
            else:
                timestamp_int = int(last_timestamp)
        else:
            timestamp_int = int(time.time())
        
        # Créer un dictionnaire au lieu d'un objet MarketState pour le DataLoader
        state_dict = {
            'prices': state_data['close'].tolist(),
            'volumes': state_data['volume'].tolist(),
            'indicators': {
                'rsi': float(state_data['rsi'].iloc[-1]),
                'macd': float(state_data['macd'].iloc[-1]),
                'atr': float(state_data['atr'].iloc[-1]),
                'adx': float(state_data['adx'].iloc[-1]),
                'bb_position': float(state_data['bb_position'].iloc[-1]),
                'stoch': float(state_data['stoch'].iloc[-1]),
            },
            'timestamp': timestamp_int
        }
        
        # Résultat futur (pour l'apprentissage supervisé)
        future_data = self.data.iloc[end_idx:end_idx+10]
        entry_price = state_data['close'].iloc[-1]
        
        # Meilleur trade possible sur les 10 prochaines bougies
        max_profit_long = (future_data['high'].max() - entry_price) / entry_price
        max_loss_long = (future_data['low'].min() - entry_price) / entry_price
        
        max_profit_short = (entry_price - future_data['low'].min()) / entry_price
        max_loss_short = (entry_price - future_data['high'].max()) / entry_price
        
        # Label optimal
        if max_profit_long > abs(max_loss_long) and max_profit_long > 0.001:
            optimal_action = 0  # BUY
            reward = max_profit_long
        elif max_profit_short > abs(max_loss_short) and max_profit_short > 0.001:
            optimal_action = 1  # SELL
            reward = max_profit_short
        else:
            optimal_action = 2  # HOLD
            reward = 0.0
        
        return state_dict, optimal_action, reward


def custom_collate_fn(batch):
    """Fonction de collation personnalisée pour gérer les dictionnaires d'état"""
    states, actions, rewards = zip(*batch)
    
    # Convertir en listes
    states = list(states)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    
    return states, actions, rewards


class RTMTrainer:
    """Entraîneur pour le RTM"""
    def __init__(self, model: RecursiveTradingModel, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
        # Métriques
        self.train_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Entraîne le modèle pour une époque"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, (states, actions, rewards) in enumerate(pbar):
            self.optimizer.zero_grad()
            
            batch_loss = 0
            batch_correct = 0
            batch_size = len(states)
            
            # Pour chaque état dans le batch
            for i, state_dict in enumerate(states):
                try:
                    # Convertir le dictionnaire en MarketState
                    state = MarketState(
                        prices=state_dict['prices'],
                        volumes=state_dict['volumes'],
                        indicators=state_dict['indicators'],
                        timestamp=state_dict['timestamp']
                    )
                    
                    # Forward pass
                    signal = self.model(state)
                    
                    # Encodage du signal
                    market_features = self.model.encode_market_state(state)
                    final_thought, _ = self.model.recursive_reasoning(market_features)
                    
                    # Prédictions
                    action_logits = self.model.action_head(final_thought)
                    confidence = torch.sigmoid(self.model.confidence_head(final_thought))
                    
                    # Loss: Cross-entropy pour l'action + MSE pour la reward
                    action_loss = nn.CrossEntropyLoss()(
                        action_logits, 
                        torch.tensor([actions[i]], dtype=torch.long, device=self.device)
                    )
                    
                    # Pénalité si mauvaise confiance
                    confidence_target = torch.tensor([1.0 if rewards[i] > 0.002 else 0.5], 
                                                    dtype=torch.float32, device=self.device)
                    confidence_loss = nn.MSELoss()(confidence, confidence_target.unsqueeze(0))
                    
                    loss = action_loss + 0.5 * confidence_loss
                    batch_loss += loss
                    
                    # Accuracy
                    pred_action = torch.argmax(action_logits, dim=-1).item()
                    if pred_action == actions[i].item():
                        batch_correct += 1
                        
                except Exception as e:
                    print(f"⚠️  Erreur sur l'échantillon {i}: {e}")
                    continue
            
            if batch_size == 0:
                continue
                
            # Moyenne du batch
            batch_loss /= batch_size
            batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += batch_loss.item()
            correct += batch_correct
            total += batch_size
            
            pbar.set_postfix({
                'loss': f'{batch_loss.item():.4f}',
                'acc': f'{batch_correct/batch_size:.2%}' if batch_size > 0 else '0%'
            })
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        accuracy = correct / total if total > 0 else 0
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Valide le modèle"""
        self.model.eval()
        correct = 0
        total = 0
        total_reward = 0
        
        with torch.no_grad():
            for states, actions, rewards in tqdm(dataloader, desc="Validation"):
                for i, state_dict in enumerate(states):
                    try:
                        # Convertir le dictionnaire en MarketState
                        state = MarketState(
                            prices=state_dict['prices'],
                            volumes=state_dict['volumes'],
                            indicators=state_dict['indicators'],
                            timestamp=state_dict['timestamp']
                        )
                        
                        signal = self.model(state)
                        
                        # Vérifier si l'action prédite est correcte
                        action_map = {"BUY": 0, "SELL": 1, "HOLD": 2, "CLOSE": 2}
                        pred_action = action_map.get(signal.action, 2)
                        
                        if pred_action == actions[i].item():
                            correct += 1
                            total_reward += rewards[i].item()
                        
                        total += 1
                    except Exception as e:
                        print(f"⚠️  Erreur validation: {e}")
                        continue
        
        accuracy = correct / total if total > 0 else 0
        avg_reward = total_reward / total if total > 0 else 0
        
        self.val_accuracies.append(accuracy)
        
        return {
            'accuracy': accuracy,
            'avg_reward': avg_reward,
            'total_samples': total
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int = 50, save_path: str = "rtm_trading_model.pt"):
        """Boucle d'entraînement complète"""
        print("=" * 70)
        print("🚀 DÉBUT DE L'ENTRAÎNEMENT DU RTM")
        print("=" * 70)
        
        best_val_acc = 0.0
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n📊 Époque {epoch}/{num_epochs}")
            
            # Entraînement
            train_metrics = self.train_epoch(train_loader, epoch)
            print(f"   Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Accuracy: {train_metrics['accuracy']:.2%}")
            
            # Validation
            val_metrics = self.validate(val_loader)
            print(f"   Val   - Accuracy: {val_metrics['accuracy']:.2%}, "
                  f"Avg Reward: {val_metrics['avg_reward']:.4f}")
            
            # Scheduler
            self.scheduler.step()
            
            # Sauvegarde du meilleur modèle
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_accuracy': best_val_acc,
                    'train_losses': self.train_losses,
                    'train_accuracies': self.train_accuracies,
                    'val_accuracies': self.val_accuracies,
                }, save_path)
                print(f"   ✅ Modèle sauvegardé (Val Acc: {best_val_acc:.2%})")
            
            # Plot tous les 10 epochs
            if epoch % 10 == 0:
                self.plot_metrics(save_path.replace('.pt', '_metrics.png'))
        
        print("\n" + "=" * 70)
        print(f"✅ ENTRAÎNEMENT TERMINÉ - Meilleure validation: {best_val_acc:.2%}")
        print("=" * 70)
        
        return best_val_acc
    
    def plot_metrics(self, save_path: str):
        """Affiche les métriques d'entraînement"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        axes[0].plot(self.train_losses, label='Train Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(self.train_accuracies, label='Train Acc', linewidth=2)
        axes[1].plot(self.val_accuracies, label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


class BacktestEngine:
    """Moteur de backtest pour le RTM"""
    def __init__(self, model: RecursiveTradingModel, initial_capital: float = 10000):
        self.model = model
        self.initial_capital = initial_capital
        self.capital = initial_capital
        
        self.trades = []
        self.equity_curve = []
        
    def run_backtest(self, dataset: TradingDataset, 
                     risk_per_trade: float = 0.02) -> Dict[str, float]:
        """Exécute un backtest complet"""
        print("\n" + "=" * 70)
        print("📈 BACKTEST EN COURS...")
        print("=" * 70)
        
        position = None
        
        for i in tqdm(range(len(dataset)), desc="Backtest"):
            try:
                state_dict, _, _ = dataset[i]
                
                # Convertir en MarketState
                state = MarketState(
                    prices=state_dict['prices'],
                    volumes=state_dict['volumes'],
                    indicators=state_dict['indicators'],
                    timestamp=state_dict['timestamp']
                )
                
                # Génération du signal
                signal = self.model(state)
                
                current_price = state.prices[-1]
                
                # Gestion de position
                if position is None and signal.action in ["BUY", "SELL"]:
                    # Ouverture de position
                    position = {
                        'type': signal.action,
                        'entry_price': current_price,
                        'sl': signal.stop_loss,
                        'tp': signal.take_profit,
                        'size': self.capital * risk_per_trade / signal.stop_loss,
                        'entry_time': i
                    }
                
                elif position is not None:
                    # Vérification SL/TP
                    pips_move = abs(current_price - position['entry_price']) * 10000
                    
                    close_trade = False
                    profit = 0
                    
                    if position['type'] == "BUY":
                        if current_price <= position['entry_price'] - position['sl'] / 10000:
                            # SL hit
                            profit = -position['size'] * position['sl']
                            close_trade = True
                        elif current_price >= position['entry_price'] + position['tp'] / 10000:
                            # TP hit
                            profit = position['size'] * position['tp']
                            close_trade = True
                        elif signal.action == "CLOSE":
                            profit = position['size'] * pips_move
                            close_trade = True
                    
                    elif position['type'] == "SELL":
                        if current_price >= position['entry_price'] + position['sl'] / 10000:
                            # SL hit
                            profit = -position['size'] * position['sl']
                            close_trade = True
                        elif current_price <= position['entry_price'] - position['tp'] / 10000:
                            # TP hit
                            profit = position['size'] * position['tp']
                            close_trade = True
                        elif signal.action == "CLOSE":
                            profit = position['size'] * pips_move
                            close_trade = True
                    
                    if close_trade:
                        self.capital += profit
                        self.trades.append({
                            'type': position['type'],
                            'profit': profit,
                            'duration': i - position['entry_time'],
                            'exit_time': i
                        })
                        position = None
                
                self.equity_curve.append(self.capital)
                
            except Exception as e:
                print(f"⚠️  Erreur backtest à l'itération {i}: {e}")
                continue
        
        return self.analyze_results()
    
    def analyze_results(self) -> Dict[str, float]:
        """Analyse les résultats du backtest"""
        if not self.trades:
            return {'error': 'No trades'}
        
        profits = [t['profit'] for t in self.trades]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]
        
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        win_rate = len(wins) / len(profits) if profits else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0
        
        # Sharpe ratio (simplifié)
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        
        # Max drawdown
        equity = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        max_drawdown = abs(np.min(drawdown))
        
        results = {
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'final_capital': self.capital
        }
        
        print("\n📊 RÉSULTATS DU BACKTEST:")
        print(f"   Trades: {results['total_trades']}")
        print(f"   Win Rate: {results['win_rate']:.2%}")
        print(f"   Return: {results['total_return']:.2%}")
        print(f"   Profit Factor: {results['profit_factor']:.2f}")
        print(f"   Sharpe: {results['sharpe_ratio']:.2f}")
        print(f"   Max DD: {results['max_drawdown']:.2%}")
        print(f"   Capital final: ${results['final_capital']:.2f}")
        
        return results


def main():
    """Fonction principale d'entraînement"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                   RTM TRADING SYSTEM                          ║
    ║              Entraînement & Backtest                          ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    DATA_FILE = "forex_data.csv"
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🖥️  Device: {DEVICE}")
    
    # Chargement des données
    if not Path(DATA_FILE).exists():
        print(f"⚠️  Fichier {DATA_FILE} introuvable!")
        print("💡 Utilisez mt5_data_downloader.py pour télécharger des données")
        return
    
    try:
        # Dataset
        full_dataset = TradingDataset(DATA_FILE, lookback=50)
        
        # Split train/val (80/20)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # DataLoaders avec fonction de collation personnalisée
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True,
            collate_fn=custom_collate_fn
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False,
            collate_fn=custom_collate_fn
        )
        
        # Modèle
        model = RecursiveTradingModel(
            input_dim=128,
            hidden_dim=256,
            num_heads=4,
            max_thinking_steps=8
        )
        
        # Entraînement
        trainer = RTMTrainer(model, device=DEVICE)
        best_acc = trainer.train(train_loader, val_loader, num_epochs=NUM_EPOCHS)
        
        # Backtest
        print("\n🔄 Lancement du backtest...")
        backtester = BacktestEngine(model, initial_capital=10000)
        results = backtester.run_backtest(full_dataset)
        
        # Sauvegarde des résultats
        with open('backtest_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n✅ Entraînement et backtest terminés!")
        print("📁 Fichiers créés:")
        print("   - rtm_trading_model.pt (modèle)")
        print("   - rtm_trading_model_metrics.png (courbes)")
        print("   - backtest_results.json (résultats)")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()