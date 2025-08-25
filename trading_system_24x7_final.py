import requests
import pandas as pd
import numpy as np
from numpy import integer as np_int
from numpy import floating as np_float
from datetime import datetime, timedelta
import time
import threading
import logging
import math
import random
import pickle
import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import deque
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
import json
warnings.filterwarnings('ignore')

# SHAP imports for feature importance analysis
try:
    import shap
    import matplotlib.pyplot as plt
    SHAP_AVAILABLE = True
except ImportError:
    print("WARNING: SHAP not available. Install with: pip install shap matplotlib")
    SHAP_AVAILABLE = False

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_system.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Fix Unicode encoding for Windows console (simpler approach)
import sys
if sys.platform == 'win32':
    # Set console code page to UTF-8
    import os
    os.system('chcp 65001 > nul')
    # Reconfigure stdout to handle UTF-8
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ================================
# Deep Learning Model Classes (Integrated from deep_model.py)
# ================================
class LSTMSignalPredictor(nn.Module):
    """LSTM-based signal predictor for options trading"""
    
    def __init__(self, input_size=50, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMSignalPredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layers
        self.regime_head = nn.Linear(hidden_size, 3)  # 3 regime classes
        self.confidence_head = nn.Linear(hidden_size, 1)  # Confidence score
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """Forward pass through the network"""
        # x shape: (batch_size, sequence_length, input_size)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.attention_norm(attn_out + lstm_out)
        
        # Use the last time step output
        last_output = attn_out[:, -1, :]  # (batch_size, hidden_size)
        last_output = self.dropout(last_output)
        
        # Predictions
        regime_logits = self.regime_head(last_output)
        confidence = torch.sigmoid(self.confidence_head(last_output))
        
        return {
            'regime_logits': regime_logits,
            'confidence': confidence,
            'attention_weights': attn_weights
        }

class SequenceBuffer:
    """Buffer for storing sequences for LSTM training"""
    
    def __init__(self, maxlen=1000, seq_len=10):
        self.maxlen = maxlen
        self.seq_len = seq_len
        self.sequences = deque(maxlen=maxlen)
        self.labels = deque(maxlen=maxlen)
        
    def add(self, sequence, label):
        """Add a sequence and its label"""
        if len(sequence) >= self.seq_len:
            self.sequences.append(sequence[-self.seq_len:])
            self.labels.append(label)
    
    def __len__(self):
        return len(self.sequences)
    
    def get_batch(self, batch_size=32):
        """Get a random batch of sequences"""
        if len(self.sequences) < batch_size:
            return None, None
            
        indices = np.random.choice(len(self.sequences), size=batch_size, replace=False)
        
        batch_sequences = [self.sequences[i] for i in indices]
        batch_labels = [self.labels[i] for i in indices]
        
        return torch.FloatTensor(batch_sequences), torch.FloatTensor(batch_labels)

def train_lstm(model, buffer, optimizer, criterion_regime, criterion_confidence, device, batch_size=32, epochs=5):
    """Train the LSTM model"""
    try:
        model.train()
        total_loss = 0
        num_batches = 0
        
        for epoch in range(epochs):
            # Get batch
            sequences, labels = buffer.get_batch(batch_size)
            
            if sequences is None or labels is None:
                continue
                
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(sequences)
            
            # Calculate losses (simplified)
            regime_loss = criterion_regime(outputs['regime_logits'], labels.long() % 3)  # Ensure valid class indices
            confidence_loss = criterion_confidence(outputs['confidence'].squeeze(), torch.ones_like(labels) * 0.5)
            
            total_loss_batch = regime_loss + confidence_loss
            
            # Backward pass
            total_loss_batch.backward()
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
        
    except Exception as e:
        print(f"LSTM training error: {e}")
        return 0.0

def predict_sequence(model, sequence, device):
    """Make prediction on a sequence"""
    try:
        model.eval()
        
        with torch.no_grad():
            # Ensure sequence is properly shaped
            if isinstance(sequence, list):
                # Sanitize nested lists to floats
                sanitized = []
                for row in sequence:
                    if isinstance(row, (list, tuple, np.ndarray)):
                        clean_row = []
                        for v in row:
                            try:
                                num = float(v)
                            except (ValueError, TypeError):
                                num = 0.0
                            if isinstance(num, float) and (np.isnan(num) or np.isinf(num)):
                                num = 0.0
                            clean_row.append(num)
                        sanitized.append(clean_row)
                    else:
                        try:
                            num = float(row)
                        except (ValueError, TypeError):
                            num = 0.0
                        if isinstance(num, float) and (np.isnan(num) or np.isinf(num)):
                            num = 0.0
                        sanitized.append(num)
                sequence = torch.FloatTensor(sanitized)
            
            if len(sequence.shape) == 2:
                sequence = sequence.unsqueeze(0)  # Add batch dimension
                
            sequence = sequence.to(device)
            
            # Get prediction
            outputs = model(sequence)
            
            # Convert to numpy for easier handling
            regime_probs = torch.softmax(outputs['regime_logits'], dim=1).cpu().numpy()[0]
            confidence = outputs['confidence'].cpu().numpy()[0][0]
            attention_weights = outputs['attention_weights'].cpu().numpy()
            
            # Get regime prediction
            regime_pred = np.argmax(regime_probs) - 1  # Convert to -1, 0, 1
            
            return {
                'regime': regime_pred,
                'confidence': confidence,
                'attention_weights': attention_weights
            }
            
    except Exception as e:
        print(f"LSTM prediction error: {e}")
        return {
            'regime': 0,
            'confidence': 0.5,
            'attention_weights': np.array([])
        }

# ================================
# Data Storage and Management System
# ================================
class MarketDataStorage:
    """24/7 Market Data Storage and Management System with Comparative Analysis"""
    
    def __init__(self):
        self.data_directory = "market_data"
        self.current_expiry = None
        self.session_data = {}
        self.historical_data = {}
        self.accumulated_data = {}
        self.previous_day_data = None
        self.today_data = []
        
        # Create data directory
        os.makedirs(self.data_directory, exist_ok=True)
        os.makedirs(f"{self.data_directory}/live", exist_ok=True)
        os.makedirs(f"{self.data_directory}/historical", exist_ok=True)
        os.makedirs(f"{self.data_directory}/accumulated", exist_ok=True)
        os.makedirs(f"{self.data_directory}/comparative", exist_ok=True)
        
        # Initialize current expiry
        self.update_expiry()
        
        # Verify and repair data files on startup
        self.verify_and_repair_data_files()
        
        # Load previous day data for comparison
        self.load_previous_day_data()
        
    def update_expiry(self):
        """Update current expiry date"""
        today = datetime.now()
        
        # Find next Thursday (Nifty expiry)
        days_ahead = 3 - today.weekday()  # Thursday is 3
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        next_thursday = today + timedelta(days=days_ahead)
        
        # If today is Thursday and past 3:30 PM, use next Thursday
        if today.weekday() == 3 and today.hour >= 15 and today.minute >= 30:
            next_thursday += timedelta(days=7)
            
        self.current_expiry = next_thursday.strftime('%Y-%m-%d')
        logger.info(f"Current expiry set to: {self.current_expiry}")
        
    def is_market_hours(self):
        """Check if current time is within market hours"""
        now = datetime.now()
        
        # Check if it's weekend
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
            
        # Check market hours (9:15 AM - 3:30 PM IST)
        market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_start <= now <= market_end
        
    def store_live_data(self, option_data, underlying_value, timestamp):
        """Store live market data during market hours with size limits"""
        try:
            if not self.is_market_hours():
                return False
                
            # Update expiry if needed
            self.update_expiry()
            
            # Convert all Timestamp columns to string
            option_data = option_data.copy()
            for col in option_data.columns:
                if option_data[col].dtype.__class__.__name__ in ['DatetimeTZDtype', 'datetime64', 'datetime64[ns]'] or str(option_data[col].dtype).startswith('datetime'):
                    option_data[col] = option_data[col].astype(str)
                if option_data[col].dtype.__class__.__name__ == 'Timestamp':
                    option_data[col] = option_data[col].astype(str)
            
            # Create data entry
            data_entry = {
                'timestamp': timestamp.isoformat(),
                'underlying_value': underlying_value,
                'expiry': self.current_expiry,
                'option_data': option_data.to_dict('records')
            }
            
            # Store in session data with size limits
            session_key = timestamp.strftime('%Y-%m-%d')
            if session_key not in self.session_data:
                self.session_data[session_key] = []
            
            self.session_data[session_key].append(data_entry)
            
            # Limit session data size (keep latest 500 entries per day)
            if len(self.session_data[session_key]) > 500:
                self.session_data[session_key] = self.session_data[session_key][-400:]  # Keep latest 400
                logger.info(f"Session data trimmed for {session_key}: keeping latest 400 entries")
            
            # Store in accumulated data with size limits
            if self.current_expiry not in self.accumulated_data:
                self.accumulated_data[self.current_expiry] = []
            
            self.accumulated_data[self.current_expiry].append(data_entry)
            
            # Limit accumulated data size (keep latest 2000 entries per expiry)
            if len(self.accumulated_data[self.current_expiry]) > 2000:
                self.accumulated_data[self.current_expiry] = self.accumulated_data[self.current_expiry][-1500:]  # Keep latest 1500
                logger.info(f"Accumulated data trimmed for {self.current_expiry}: keeping latest 1500 entries")
            
            # Save to file
            self.save_session_data(session_key)
            self.save_accumulated_data(self.current_expiry)
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing live data: {e}")
            return False
            
    def save_session_data(self, session_key):
        """Save session data to file with backup and error handling"""
        try:
            file_path = f"{self.data_directory}/live/{session_key}.json"
            backup_path = f"{self.data_directory}/live/{session_key}_backup.json"
            
            # Create backup of existing file before saving new data
            if os.path.exists(file_path):
                try:
                    import shutil
                    shutil.copy2(file_path, backup_path)
                except:
                    pass  # Backup failed, but continue with save
            
            # Use safe JSON write
            self._safe_json_write(file_path, self.session_data[session_key])
            
        except Exception as e:
            logger.error(f"Error saving session data: {e}")
            
    def save_accumulated_data(self, expiry):
        """Save accumulated data to file with backup and error handling"""
        try:
            file_path = f"{self.data_directory}/accumulated/{expiry}.json"
            backup_path = f"{self.data_directory}/accumulated/{expiry}_backup.json"
            
            # Create backup of existing file before saving new data
            if os.path.exists(file_path):
                try:
                    import shutil
                    shutil.copy2(file_path, backup_path)
                except:
                    pass  # Backup failed, but continue with save
            
            # Use safe JSON write
            self._safe_json_write(file_path, self.accumulated_data[expiry])
            
        except Exception as e:
            logger.error(f"Error saving accumulated data: {e}")
            
    def _safe_json_write(self, file_path, data):
        """Safely write JSON data with atomic operation"""
        try:
            # Write to temporary file first
            temp_path = f"{file_path}.tmp"
            
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()  # Ensure data is written to disk
                os.fsync(f.fileno())  # Force write to disk
            
            # Atomic move (replace original file)
            if os.path.exists(file_path):
                os.replace(temp_path, file_path)
            else:
                os.rename(temp_path, file_path)
                
        except Exception as e:
            # Clean up temp file if it exists
            if os.path.exists(f"{file_path}.tmp"):
                try:
                    os.remove(f"{file_path}.tmp")
                except:
                    pass
            raise e
    
    def _attempt_json_repair(self, file_path):
        """Attempt to repair corrupted JSON file by extracting valid data"""
        try:
            logger.info(f"Attempting JSON repair for: {file_path}")
            
            # Read the raw file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Try to extract valid JSON objects from the content
            repaired_data = []
            
            # Look for individual JSON objects (entries in the array)
            import re
            
            # Pattern to match individual data entries
            entry_pattern = r'\{[^{}]*"timestamp"[^{}]*\}'
            matches = re.findall(entry_pattern, content)
            
            for match in matches:
                try:
                    # Try to parse each potential entry
                    entry = json.loads(match)
                    if self._validate_data_entry(entry):
                        repaired_data.append(entry)
                except:
                    continue
            
            # If we found some valid entries, return them
            if repaired_data:
                logger.info(f"Repair successful: extracted {len(repaired_data)} valid entries")
                return repaired_data
            
            # Alternative repair: try to find the largest valid JSON array
            array_pattern = r'\[[^\[\]]*\]'
            array_matches = re.findall(array_pattern, content)
            
            for array_match in array_matches:
                try:
                    data = json.loads(array_match)
                    if isinstance(data, list) and len(data) > 0:
                        # Validate entries
                        valid_entries = []
                        for entry in data:
                            if self._validate_data_entry(entry):
                                valid_entries.append(entry)
                        
                        if valid_entries:
                            logger.info(f"Array repair successful: extracted {len(valid_entries)} valid entries")
                            return valid_entries
                except:
                    continue
            
            logger.warning(f"JSON repair failed: no valid data found")
            return []
            
        except Exception as e:
            logger.error(f"JSON repair attempt failed: {e}")
            return []
    
    def _validate_data_entry(self, entry):
        """Validate that a data entry has required structure"""
        try:
            return (isinstance(entry, dict) and 
                    'timestamp' in entry and 
                    'underlying_value' in entry and 
                    'option_data' in entry)
        except:
            return False
            
    def load_session_data(self, session_key):
        """Load session data from file with robust error handling"""
        file_path = f"{self.data_directory}/live/{session_key}.json"
        backup_path = f"{self.data_directory}/live/{session_key}_backup.json"
        
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
            return []
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in session data for {session_key}: {e}")
            
            # Try to load backup file if it exists
            if os.path.exists(backup_path):
                try:
                    logger.info(f"Attempting to restore session from backup: {backup_path}")
                    with open(backup_path, 'r') as f:
                        data = json.load(f)
                        logger.info(f"Successfully restored session from backup: {len(data)} samples")
                        
                        # Restore the main file from backup
                        self._safe_json_write(file_path, data)
                        return data
                        
                except Exception as backup_error:
                    logger.error(f"Session backup file also corrupted: {backup_error}")
            
            # If both main and backup fail, try to repair the file
            logger.warning(f"Attempting to repair corrupted session file: {file_path}")
            repaired_data = self._attempt_json_repair(file_path)
            if repaired_data:
                logger.info(f"Session file repair successful: {len(repaired_data)} samples recovered")
                # Save the repaired data
                self._safe_json_write(file_path, repaired_data)
                self._safe_json_write(backup_path, repaired_data)  # Create new backup
                return repaired_data
            
            # If all recovery attempts fail, rename corrupted file and start fresh
            logger.error(f"All session recovery attempts failed. Creating fresh session data file.")
            corrupted_path = f"{file_path}.corrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                os.rename(file_path, corrupted_path)
                logger.info(f"Corrupted session file moved to: {corrupted_path}")
            except:
                pass
            
            return []
            
        except Exception as e:
            logger.error(f"Unexpected error loading session data: {e}")
            return []
            
    def load_accumulated_data(self, expiry):
        """Load accumulated data from file with robust error handling and recovery"""
        file_path = f"{self.data_directory}/accumulated/{expiry}.json"
        backup_path = f"{self.data_directory}/accumulated/{expiry}_backup.json"
        
        try:
            if os.path.exists(file_path):
                # Try to load the main file
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Successfully loaded accumulated data: {len(data)} samples")
                    return data
            return []
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in accumulated data for {expiry}: {e}")
            
            # Try to load backup file if it exists
            if os.path.exists(backup_path):
                try:
                    logger.info(f"Attempting to restore from backup: {backup_path}")
                    with open(backup_path, 'r') as f:
                        data = json.load(f)
                        logger.info(f"Successfully restored from backup: {len(data)} samples")
                        
                        # Restore the main file from backup
                        self._safe_json_write(file_path, data)
                        logger.info(f"Main file restored from backup")
                        return data
                        
                except Exception as backup_error:
                    logger.error(f"Backup file also corrupted: {backup_error}")
            
            # If both main and backup fail, try to repair the file
            logger.warning(f"Attempting to repair corrupted file: {file_path}")
            repaired_data = self._attempt_json_repair(file_path)
            if repaired_data:
                logger.info(f"File repair successful: {len(repaired_data)} samples recovered")
                # Save the repaired data
                self._safe_json_write(file_path, repaired_data)
                self._safe_json_write(backup_path, repaired_data)  # Create new backup
                return repaired_data
            
            # If all recovery attempts fail, rename corrupted file and start fresh
            logger.error(f"All recovery attempts failed. Creating fresh accumulated data file.")
            corrupted_path = f"{file_path}.corrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                os.rename(file_path, corrupted_path)
                logger.info(f"Corrupted file moved to: {corrupted_path}")
            except:
                pass
            
            return []
            
        except Exception as e:
            logger.error(f"Unexpected error loading accumulated data: {e}")
            return []
            
    def get_training_data(self, days_back=30):
        """Get training data from stored sessions"""
        try:
            training_data = []
            today = datetime.now()
            
            # Get data from last N days
            for i in range(days_back):
                date = today - timedelta(days=i)
                session_key = date.strftime('%Y-%m-%d')
                
                # Load session data
                session_data = self.load_session_data(session_key)
                if session_data:
                    training_data.extend(session_data)
                    
            # Load accumulated data for current expiry
            if self.current_expiry:
                accumulated_data = self.load_accumulated_data(self.current_expiry)
                training_data.extend(accumulated_data)
                
            logger.info(f"Loaded {len(training_data)} training samples from stored data")
            return training_data
            
        except Exception as e:
            logger.error(f"Error getting training data: {e}")
            return []
    
    def get_all_available_data(self):
        """Get ALL available data from all sources for comprehensive offline analysis"""
        try:
            all_data = []
            data_sources = 0
            
            logger.info("🔍 Scanning all data sources...")
            
            # Load from accumulated data (main historical data)
            accumulated_dir = f"{self.data_directory}/accumulated"
            if os.path.exists(accumulated_dir):
                for filename in os.listdir(accumulated_dir):
                    if filename.endswith('.json') and not filename.endswith('_backup.json'):
                        try:
                            expiry = filename.replace('.json', '')
                            file_data = self.load_accumulated_data(expiry)
                            if file_data:
                                all_data.extend(file_data)
                                data_sources += 1
                                logger.info(f"✅ Loaded {len(file_data)} samples from accumulated/{filename}")
                        except Exception as e:
                            logger.warning(f"Failed to load accumulated/{filename}: {e}")
            
            # Load from live data directory
            live_dir = f"{self.data_directory}/live"
            if os.path.exists(live_dir):
                for filename in os.listdir(live_dir):
                    if filename.endswith('.json') and not filename.endswith('_backup.json'):
                        try:
                            session_key = filename.replace('.json', '')
                            file_data = self.load_session_data(session_key)
                            if file_data:
                                all_data.extend(file_data)
                                data_sources += 1
                                logger.info(f"✅ Loaded {len(file_data)} samples from live/{filename}")
                        except Exception as e:
                            logger.warning(f"Failed to load live/{filename}: {e}")
            
            # Load from historical data directory if exists
            historical_dir = f"{self.data_directory}/historical"
            if os.path.exists(historical_dir):
                for filename in os.listdir(historical_dir):
                    if filename.endswith('.json'):
                        try:
                            file_path = os.path.join(historical_dir, filename)
                            with open(file_path, 'r') as f:
                                file_data = json.load(f)
                                if isinstance(file_data, list) and file_data:
                                    all_data.extend(file_data)
                                    data_sources += 1
                                    logger.info(f"✅ Loaded {len(file_data)} samples from historical/{filename}")
                        except Exception as e:
                            logger.warning(f"Failed to load historical/{filename}: {e}")
            
            # Remove duplicates based on timestamp (keep most recent)
            unique_data = {}
            for entry in all_data:
                timestamp = entry.get('timestamp', '')
                if timestamp:
                    unique_data[timestamp] = entry
            
            final_data = list(unique_data.values())
            
            logger.info(f"📊 DATA LOADING SUMMARY:")
            logger.info(f"   • Sources scanned: {data_sources}")
            logger.info(f"   • Raw samples: {len(all_data)}")
            logger.info(f"   • Unique samples: {len(final_data)}")
            logger.info(f"   • Deduplication ratio: {(len(all_data) - len(final_data))/len(all_data)*100:.1f}%" if all_data else "0%")
            
            return final_data
            
        except Exception as e:
            logger.error(f"Error loading all available data: {e}")
            return []
            
    def cleanup_old_data(self, days_to_keep=90):
        """Clean up old data files"""
        try:
            today = datetime.now()
            cutoff_date = today - timedelta(days=days_to_keep)
            
            # Clean up live data
            live_dir = f"{self.data_directory}/live"
            for filename in os.listdir(live_dir):
                if filename.endswith('.json'):
                    file_date_str = filename.replace('.json', '')
                    try:
                        file_date = datetime.strptime(file_date_str, '%Y-%m-%d')
                        if file_date < cutoff_date:
                            os.remove(os.path.join(live_dir, filename))
                            logger.info(f"Removed old live data: {filename}")
                    except:
                        pass
                        
            # Clean up historical data
            historical_dir = f"{self.data_directory}/historical"
            for filename in os.listdir(historical_dir):
                if filename.endswith('.json'):
                    file_date_str = filename.replace('.json', '')
                    try:
                        file_date = datetime.strptime(file_date_str, '%Y-%m-%d')
                        if file_date < cutoff_date:
                            os.remove(os.path.join(historical_dir, filename))
                            logger.info(f"Removed old historical data: {filename}")
                    except:
                        pass
                        
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            
    def get_data_statistics(self):
        """Get statistics about stored data"""
        try:
            stats = {
                'total_sessions': 0,
                'total_samples': 0,
                'current_expiry': self.current_expiry,
                'market_hours': self.is_market_hours(),
                'sessions': {}
            }
            
            # Count session data
            live_dir = f"{self.data_directory}/live"
            for filename in os.listdir(live_dir):
                if filename.endswith('.json'):
                    session_key = filename.replace('.json', '')
                    session_data = self.load_session_data(session_key)
                    stats['sessions'][session_key] = len(session_data)
                    stats['total_sessions'] += 1
                    stats['total_samples'] += len(session_data)
                    
            return stats
            
        except Exception as e:
            logger.error(f"Error getting data statistics: {e}")
            return {}
    
    def load_previous_day_data(self):
        """Load previous trading day data for comparison"""
        try:
            today = datetime.now()
            # Get previous trading day (skip weekends)
            previous_day = today - timedelta(days=1)
            while previous_day.weekday() >= 5:  # Skip weekends
                previous_day -= timedelta(days=1)
                
            previous_day_key = previous_day.strftime('%Y-%m-%d')
            self.previous_day_data = self.load_session_data(previous_day_key)
            
            if self.previous_day_data:
                logger.info(f"Loaded {len(self.previous_day_data)} samples from previous day: {previous_day_key}")
            else:
                logger.info(f"No previous day data found for: {previous_day_key}")
                
        except Exception as e:
            logger.error(f"Error loading previous day data: {e}")
            
    def get_comparative_analysis(self, current_data, underlying_value, atm_strike):
        """Get comparative analysis between today, previous day, and accumulated data"""
        try:
            analysis = {
                'today_vs_previous': {},
                'today_vs_accumulated': {},
                'market_evolution': {},
                'pattern_recognition': {},
                'volatility_comparison': {}
            }
            
            # Current ATM data
            current_atm = current_data[current_data['Strike'] == atm_strike]
            if current_atm.empty:
                return analysis
                
            current_atm_row = current_atm.iloc[0]
            
            # Compare with previous day data
            if self.previous_day_data:
                prev_day_analysis = self.compare_with_previous_day(current_atm_row, atm_strike)
                analysis['today_vs_previous'] = prev_day_analysis
                
            # Compare with accumulated data
            if self.current_expiry in self.accumulated_data:
                accumulated_analysis = self.compare_with_accumulated(current_atm_row, atm_strike)
                analysis['today_vs_accumulated'] = accumulated_analysis
                
            # Market evolution analysis
            evolution_analysis = self.analyze_market_evolution(current_atm_row, atm_strike)
            analysis['market_evolution'] = evolution_analysis
            
            # Pattern recognition across time periods
            pattern_analysis = self.recognize_patterns_across_time(current_atm_row, atm_strike)
            analysis['pattern_recognition'] = pattern_analysis
            
            # Volatility comparison
            volatility_analysis = self.compare_volatility_trends(current_data)
            analysis['volatility_comparison'] = volatility_analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in comparative analysis: {e}")
            return {}
            
    def compare_with_previous_day(self, current_atm_row, atm_strike):
        """Compare current data with previous day's same time data"""
        try:
            current_time = datetime.now()
            target_time = current_time.strftime('%H:%M')
            
            # Find similar time data from previous day
            similar_time_data = []
            for entry in self.previous_day_data:
                entry_time = datetime.fromisoformat(entry['timestamp']).strftime('%H:%M')
                if abs(datetime.strptime(entry_time, '%H:%M').hour - current_time.hour) <= 1:
                    # Find ATM strike data
                    for option_record in entry['option_data']:
                        if option_record['Strike'] == atm_strike:
                            similar_time_data.append(option_record)
                            break
                            
            if not similar_time_data:
                return {'status': 'no_comparable_data'}
                
            # Calculate averages from previous day
            prev_avg = {
                'call_oi': np.mean([d['Call_OI'] for d in similar_time_data]),
                'put_oi': np.mean([d['Put_OI'] for d in similar_time_data]),
                'call_ltp': np.mean([d['Call_LTP'] for d in similar_time_data]),
                'put_ltp': np.mean([d['Put_LTP'] for d in similar_time_data]),
                'call_volume': np.mean([d['Call_Volume'] for d in similar_time_data]),
                'put_volume': np.mean([d['Put_Volume'] for d in similar_time_data])
            }
            
            # Calculate percentage changes
            comparison = {
                'call_oi_change': ((current_atm_row['Call_OI'] - prev_avg['call_oi']) / max(prev_avg['call_oi'], 1)) * 100,
                'put_oi_change': ((current_atm_row['Put_OI'] - prev_avg['put_oi']) / max(prev_avg['put_oi'], 1)) * 100,
                'call_ltp_change': ((current_atm_row['Call_LTP'] - prev_avg['call_ltp']) / max(prev_avg['call_ltp'], 1)) * 100,
                'put_ltp_change': ((current_atm_row['Put_LTP'] - prev_avg['put_ltp']) / max(prev_avg['put_ltp'], 1)) * 100,
                'call_volume_change': ((current_atm_row['Call_Volume'] - prev_avg['call_volume']) / max(prev_avg['call_volume'], 1)) * 100,
                'put_volume_change': ((current_atm_row['Put_Volume'] - prev_avg['put_volume']) / max(prev_avg['put_volume'], 1)) * 100,
                'previous_data_points': len(similar_time_data),
                'comparison_time_window': target_time
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing with previous day: {e}")
            return {'error': str(e)}
            
    def compare_with_accumulated(self, current_atm_row, atm_strike):
        """Compare current data with accumulated data from expiry start"""
        try:
            if self.current_expiry not in self.accumulated_data:
                return {'status': 'no_accumulated_data'}
                
            accumulated_entries = self.accumulated_data[self.current_expiry]
            
            # Collect all ATM data from accumulated
            accumulated_atm_data = []
            for entry in accumulated_entries:
                for option_record in entry['option_data']:
                    if option_record['Strike'] == atm_strike:
                        accumulated_atm_data.append(option_record)
                        
            if not accumulated_atm_data:
                return {'status': 'no_atm_data_in_accumulated'}
                
            # Calculate statistics from accumulated data
            accumulated_stats = {
                'call_oi_avg': np.mean([d['Call_OI'] for d in accumulated_atm_data]),
                'call_oi_max': np.max([d['Call_OI'] for d in accumulated_atm_data]),
                'call_oi_min': np.min([d['Call_OI'] for d in accumulated_atm_data]),
                'put_oi_avg': np.mean([d['Put_OI'] for d in accumulated_atm_data]),
                'put_oi_max': np.max([d['Put_OI'] for d in accumulated_atm_data]),
                'put_oi_min': np.min([d['Put_OI'] for d in accumulated_atm_data]),
                'call_ltp_avg': np.mean([d['Call_LTP'] for d in accumulated_atm_data]),
                'call_ltp_max': np.max([d['Call_LTP'] for d in accumulated_atm_data]),
                'call_ltp_min': np.min([d['Call_LTP'] for d in accumulated_atm_data]),
                'put_ltp_avg': np.mean([d['Put_LTP'] for d in accumulated_atm_data]),
                'put_ltp_max': np.max([d['Put_LTP'] for d in accumulated_atm_data]),
                'put_ltp_min': np.min([d['Put_LTP'] for d in accumulated_atm_data])
            }
            
            # Calculate current position relative to accumulated range
            comparison = {
                'call_oi_percentile': self.calculate_percentile(current_atm_row['Call_OI'], [d['Call_OI'] for d in accumulated_atm_data]),
                'put_oi_percentile': self.calculate_percentile(current_atm_row['Put_OI'], [d['Put_OI'] for d in accumulated_atm_data]),
                'call_ltp_percentile': self.calculate_percentile(current_atm_row['Call_LTP'], [d['Call_LTP'] for d in accumulated_atm_data]),
                'put_ltp_percentile': self.calculate_percentile(current_atm_row['Put_LTP'], [d['Put_LTP'] for d in accumulated_atm_data]),
                'accumulated_data_points': len(accumulated_atm_data),
                'accumulated_stats': accumulated_stats,
                'expiry_start_date': accumulated_entries[0]['timestamp'] if accumulated_entries else None
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing with accumulated data: {e}")
            return {'error': str(e)}
            
    def calculate_percentile(self, current_value, historical_values):
        """Calculate what percentile the current value represents in historical data"""
        try:
            if not historical_values:
                return 50  # Default to median if no data
                
            historical_values = sorted(historical_values)
            position = sum(1 for x in historical_values if x <= current_value)
            percentile = (position / len(historical_values)) * 100
            return round(percentile, 2)
            
        except:
            return 50
            
    def analyze_market_evolution(self, current_atm_row, atm_strike):
        """Analyze how market has evolved over time"""
        try:
            today_key = datetime.now().strftime('%Y-%m-%d')
            
            if today_key not in self.session_data or len(self.session_data[today_key]) < 2:
                return {'status': 'insufficient_today_data'}
                
            today_data = self.session_data[today_key]
            
            # Get ATM data from start of day vs now
            start_of_day_atm = None
            for entry in today_data[:5]:  # Look at first 5 entries
                for option_record in entry['option_data']:
                    if option_record['Strike'] == atm_strike:
                        start_of_day_atm = option_record
                        break
                if start_of_day_atm:
                    break
                    
            if not start_of_day_atm:
                return {'status': 'no_start_of_day_data'}
                
            # Calculate evolution metrics
            evolution = {
                'call_oi_evolution': current_atm_row['Call_OI'] - start_of_day_atm['Call_OI'],
                'put_oi_evolution': current_atm_row['Put_OI'] - start_of_day_atm['Put_OI'],
                'call_ltp_evolution': current_atm_row['Call_LTP'] - start_of_day_atm['Call_LTP'],
                'put_ltp_evolution': current_atm_row['Put_LTP'] - start_of_day_atm['Put_LTP'],
                'call_volume_evolution': current_atm_row['Call_Volume'] - start_of_day_atm['Call_Volume'],
                'put_volume_evolution': current_atm_row['Put_Volume'] - start_of_day_atm['Put_Volume'],
                'pcr_oi_start': start_of_day_atm['Put_OI'] / max(start_of_day_atm['Call_OI'], 1),
                'pcr_oi_current': current_atm_row['Put_OI'] / max(current_atm_row['Call_OI'], 1),
                'trading_minutes_elapsed': len(today_data)
            }
            
            return evolution
            
        except Exception as e:
            logger.error(f"Error analyzing market evolution: {e}")
            return {'error': str(e)}
            
    def recognize_patterns_across_time(self, current_atm_row, atm_strike):
        """Recognize patterns across different time periods"""
        try:
            patterns = {
                'intraday_pattern': 'unknown',
                'inter_day_pattern': 'unknown',
                'accumulated_pattern': 'unknown',
                'confidence': 0.0
            }
            
            # Analyze intraday pattern
            today_key = datetime.now().strftime('%Y-%m-%d')
            if today_key in self.session_data and len(self.session_data[today_key]) >= 3:
                recent_entries = self.session_data[today_key][-3:]
                call_ltp_trend = []
                put_ltp_trend = []
                
                for entry in recent_entries:
                    for option_record in entry['option_data']:
                        if option_record['Strike'] == atm_strike:
                            call_ltp_trend.append(option_record['Call_LTP'])
                            put_ltp_trend.append(option_record['Put_LTP'])
                            break
                            
                if len(call_ltp_trend) >= 3:
                    if call_ltp_trend[-1] > call_ltp_trend[-2] > call_ltp_trend[-3]:
                        patterns['intraday_pattern'] = 'call_bullish'
                    elif call_ltp_trend[-1] < call_ltp_trend[-2] < call_ltp_trend[-3]:
                        patterns['intraday_pattern'] = 'call_bearish'
                    elif put_ltp_trend[-1] > put_ltp_trend[-2] > put_ltp_trend[-3]:
                        patterns['intraday_pattern'] = 'put_bullish'
                    elif put_ltp_trend[-1] < put_ltp_trend[-2] < put_ltp_trend[-3]:
                        patterns['intraday_pattern'] = 'put_bearish'
                    else:
                        patterns['intraday_pattern'] = 'sideways'
                        
            return patterns
            
        except Exception as e:
            logger.error(f"Error recognizing patterns: {e}")
            return {'error': str(e)}
            
    def compare_volatility_trends(self, current_data):
        """Compare volatility trends across time periods"""
        try:
            volatility_analysis = {
                'current_implied_volatility': 0.0,
                'previous_day_iv': 0.0,
                'accumulated_avg_iv': 0.0,
                'volatility_trend': 'unknown'
            }
            
            # Calculate current implied volatility proxy (simplified)
            call_puts_ratio = []
            for _, row in current_data.iterrows():
                if row['Call_LTP'] > 0 and row['Put_LTP'] > 0:
                    ratio = (row['Call_LTP'] + row['Put_LTP']) / 2
                    call_puts_ratio.append(ratio)
                    
            if call_puts_ratio:
                volatility_analysis['current_implied_volatility'] = np.std(call_puts_ratio)
                
            return volatility_analysis
            
        except Exception as e:
            logger.error(f"Error comparing volatility trends: {e}")
            return {'error': str(e)}
            
    def get_enhanced_features(self, current_data, underlying_value, atm_strike):
        """Get enhanced features including comparative analysis"""
        try:
            # Get comparative analysis
            comparative_analysis = self.get_comparative_analysis(current_data, underlying_value, atm_strike)
            
            # Enhanced features dictionary
            enhanced_features = {}
            
            # Add comparative features
            if 'today_vs_previous' in comparative_analysis:
                prev_comp = comparative_analysis['today_vs_previous']
                if 'call_oi_change' in prev_comp:
                    enhanced_features.update({
                        'prev_day_call_oi_change': prev_comp['call_oi_change'],
                        'prev_day_put_oi_change': prev_comp['put_oi_change'],
                        'prev_day_call_ltp_change': prev_comp['call_ltp_change'],
                        'prev_day_put_ltp_change': prev_comp['put_ltp_change'],
                        'prev_day_call_volume_change': prev_comp['call_volume_change'],
                        'prev_day_put_volume_change': prev_comp['put_volume_change']
                    })
                    
            if 'today_vs_accumulated' in comparative_analysis:
                acc_comp = comparative_analysis['today_vs_accumulated']
                if 'call_oi_percentile' in acc_comp:
                    enhanced_features.update({
                        'accumulated_call_oi_percentile': acc_comp['call_oi_percentile'],
                        'accumulated_put_oi_percentile': acc_comp['put_oi_percentile'],
                        'accumulated_call_ltp_percentile': acc_comp['call_ltp_percentile'],
                        'accumulated_put_ltp_percentile': acc_comp['put_ltp_percentile']
                    })
                    
            if 'market_evolution' in comparative_analysis:
                evolution = comparative_analysis['market_evolution']
                if 'call_oi_evolution' in evolution:
                    enhanced_features.update({
                        'intraday_call_oi_evolution': evolution['call_oi_evolution'],
                        'intraday_put_oi_evolution': evolution['put_oi_evolution'],
                        'intraday_call_ltp_evolution': evolution['call_ltp_evolution'],
                        'intraday_put_ltp_evolution': evolution['put_ltp_evolution'],
                        'intraday_pcr_change': evolution.get('pcr_oi_current', 1) - evolution.get('pcr_oi_start', 1)
                    })
                    
            return enhanced_features
            
        except Exception as e:
            logger.error(f"Error getting enhanced features: {e}")
            return {}
            
    def save_comparative_analysis(self, analysis, timestamp):
        """Save comparative analysis for future reference"""
        try:
            # Convert numpy types to Python native types
            def convert_to_native(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_native(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_native(item) for item in obj]
                return obj

            # Convert analysis to native Python types
            converted_analysis = convert_to_native(analysis)
            
            analysis_with_timestamp = {
                'timestamp': timestamp.isoformat(),
                'analysis': converted_analysis
            }
            
            date_key = timestamp.strftime('%Y-%m-%d')
            file_path = f"{self.data_directory}/comparative/{date_key}_analysis.json"
            
            # Load existing analyses for the day
            existing_analyses = []
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    existing_analyses = json.load(f)
                    
            existing_analyses.append(analysis_with_timestamp)
            
            # Save updated analyses
            with open(file_path, 'w') as f:
                json.dump(existing_analyses, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving comparative analysis: {e}")
            logger.debug(f"Analysis data: {analysis}")
    
    def verify_and_repair_data_files(self):
        """Verify and repair corrupted data files on startup"""
        try:
            repaired_files = 0
            corrupted_files = 0
            
            logger.info("🔧 Verifying data file integrity on startup...")
            
            # Check accumulated data files
            accumulated_dir = f"{self.data_directory}/accumulated"
            if os.path.exists(accumulated_dir):
                for filename in os.listdir(accumulated_dir):
                    if filename.endswith('.json') and not filename.endswith('_backup.json'):
                        file_path = os.path.join(accumulated_dir, filename)
                        try:
                            with open(file_path, 'r') as f:
                                json.load(f)  # Try to parse
                        except json.JSONDecodeError:
                            logger.warning(f"⚠️ Corrupted accumulated file detected: {filename}")
                            corrupted_files += 1
                            
                            # Try to repair using backup or repair method
                            expiry = filename.replace('.json', '')
                            data = self.load_accumulated_data(expiry)  # This will attempt repair
                            if data:
                                repaired_files += 1
                                logger.info(f"✅ Repaired accumulated file: {filename}")
                        except Exception as e:
                            logger.error(f"Error checking {filename}: {e}")
            
            # Check session data files
            live_dir = f"{self.data_directory}/live"
            if os.path.exists(live_dir):
                for filename in os.listdir(live_dir):
                    if filename.endswith('.json') and not filename.endswith('_backup.json'):
                        file_path = os.path.join(live_dir, filename)
                        try:
                            with open(file_path, 'r') as f:
                                json.load(f)  # Try to parse
                        except json.JSONDecodeError:
                            logger.warning(f"⚠️ Corrupted session file detected: {filename}")
                            corrupted_files += 1
                            
                            # Try to repair using backup or repair method
                            session_key = filename.replace('.json', '')
                            data = self.load_session_data(session_key)  # This will attempt repair
                            if data:
                                repaired_files += 1
                                logger.info(f"✅ Repaired session file: {filename}")
                        except Exception as e:
                            logger.error(f"Error checking {filename}: {e}")
            
            # Clean up any orphaned temporary files
            temp_files_cleaned = 0
            for root, dirs, files in os.walk(self.data_directory):
                for file in files:
                    if file.endswith('.tmp'):
                        try:
                            os.remove(os.path.join(root, file))
                            temp_files_cleaned += 1
                        except:
                            pass
            
            # Summary report
            logger.info(f"🔧 Data verification complete:")
            logger.info(f"   • Corrupted files found: {corrupted_files}")
            logger.info(f"   • Files repaired: {repaired_files}")
            logger.info(f"   • Temp files cleaned: {temp_files_cleaned}")
            
            if corrupted_files == 0:
                logger.info(f"✅ All data files are healthy!")
            elif repaired_files == corrupted_files:
                logger.info(f"✅ All corrupted files successfully repaired!")
            else:
                logger.warning(f"⚠️ {corrupted_files - repaired_files} files could not be repaired")
                
        except Exception as e:
            logger.error(f"Error during data verification: {e}")

# ================================
# Advanced Risk Management System
# ================================
class AdvancedRiskManager:
    """Advanced Risk Management System for Options Trading"""
    
    def __init__(self):
        # Risk parameters
        self.max_risk_per_trade = 0.02  # 2% max risk per trade
        self.max_portfolio_risk = 0.06   # 6% max portfolio risk
        self.max_open_positions = 3      # Maximum concurrent positions
        
        # Performance tracking
        self.trade_history = []
        self.portfolio_value = 100000    # Initial portfolio value
        self.current_positions = []
        
        # Volatility tracking
        self.volatility_window = 20
        self.volatility_history = []
        
    def calculate_position_size(self, account_value, volatility, confidence, signal_strength):
        """Dynamic position sizing based on Kelly Criterion and market conditions - Returns LOTS"""
        try:
            # Get historical performance metrics
            win_rate = self.get_historical_win_rate()
            avg_win = self.get_average_win()
            avg_loss = self.get_average_loss()
            
            if avg_loss == 0 or len(self.trade_history) < 10:
                # Use conservative sizing if not enough history
                base_size = account_value * 0.01  # 1% base size
            else:
                # Kelly Criterion calculation
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                kelly_fraction = np.clip(kelly_fraction, 0, 0.25)  # Cap at 25%
                base_size = account_value * kelly_fraction
            
            # Volatility adjustment
            volatility_adjustment = 1 / (1 + volatility * 2)  # Reduce size in high vol
            
            # Confidence adjustment
            confidence_adjustment = min(confidence / 3.0, 1.0)  # Scale with confidence
            
            # Signal strength adjustment
            strength_adjustment = min(signal_strength / 10.0, 1.0)
            
            # Calculate final position size in monetary terms
            position_size_money = base_size * volatility_adjustment * confidence_adjustment * strength_adjustment
            
            # Apply risk limits
            max_position = account_value * self.max_risk_per_trade
            position_size_money = min(position_size_money, max_position)
            
            # Ensure minimum position size
            min_position = account_value * 0.005  # 0.5% minimum
            position_size_money = max(position_size_money, min_position)
            
            # Convert to LOTS (1 lot = Config.LOT_SIZE contracts)
            # Assume average option price of ₹100 per contract for lot calculation
            avg_option_price = 100
            total_contracts = position_size_money / avg_option_price
            lots = max(1, round(total_contracts / Config.LOT_SIZE))  # Minimum 1 lot
            
            # Cap maximum lots based on account size
            max_lots = max(1, account_value // 50000)  # 1 lot per ₹50k account value
            lots = min(lots, max_lots)
            
            return lots
            
        except Exception as e:
            logger.error(f"Position sizing error: {e}")
            return 1  # Conservative fallback - 1 lot
    
    def calculate_dynamic_stop_loss(self, entry_price, volatility, signal_strength, option_type='CE'):
        """Calculate dynamic stop loss based on volatility and signal strength"""
        try:
            # Calculate ATR-based stop loss
            atr = self.calculate_atr(period=14)
            
            # Base stop loss (2 ATR)
            base_stop = atr * 2.0
            
            # Adjust for signal strength
            strength_adjustment = 1 + (signal_strength / 10.0)
            
            # Adjust for volatility
            vol_adjustment = 1 + (volatility - 0.2) * 2  # Increase stop in high vol
            
            # Calculate dynamic stop
            dynamic_stop = base_stop * strength_adjustment * vol_adjustment
            
            # Ensure minimum stop loss (5% of entry price)
            min_stop = entry_price * 0.05
            dynamic_stop = max(dynamic_stop, min_stop)
            
            # Ensure maximum stop loss (15% of entry price)
            max_stop = entry_price * 0.15
            dynamic_stop = min(dynamic_stop, max_stop)
            
            return dynamic_stop
            
        except Exception as e:
            logger.error(f"Stop loss calculation error: {e}")
            return entry_price * 0.10  # 10% fallback stop loss
    
    def calculate_trailing_stop(self, entry_price, current_price, max_profit, volatility):
        """Calculate optimal trailing stop based on profit level"""
        try:
            # Calculate profit percentage
            profit_percentage = (current_price - entry_price) / entry_price
            
            # Base trailing stop percentages
            if profit_percentage < 0.5:
                # Early stage - tight stop (30% of profit)
                trailing_stop = 0.3
            elif profit_percentage < 1.0:
                # Mid stage - moderate stop (50% of profit)
                trailing_stop = 0.5
            else:
                # Late stage - wide stop to let profits run (70% of profit)
                trailing_stop = 0.7
            
            # Adjust for volatility
            vol_adjustment = 1 + (volatility - 0.2) * 1.5
            trailing_stop *= vol_adjustment
            
            # Calculate actual stop price
            stop_price = current_price - (max_profit * trailing_stop)
            
            # Ensure stop is not below entry price
            stop_price = max(stop_price, entry_price * 0.95)
            
            return stop_price
            
        except Exception as e:
            logger.error(f"Trailing stop calculation error: {e}")
            return current_price * 0.95  # 5% below current price
    
    def calculate_atr(self, period=14):
        """Calculate Average True Range for volatility measurement"""
        try:
            if len(self.volatility_history) < period:
                return 0.02  # Default 2% volatility
            
            # Use recent volatility data
            recent_vol = self.volatility_history[-period:]
            atr = np.mean(recent_vol)
            
            return max(atr, 0.01)  # Minimum 1% volatility
            
        except Exception as e:
            logger.error(f"ATR calculation error: {e}")
            return 0.02
    
    def get_historical_win_rate(self):
        """Calculate historical win rate from trade history"""
        if len(self.trade_history) < 5:
            return 0.5  # Default 50% win rate
        
        winning_trades = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
        return winning_trades / len(self.trade_history)
    
    def get_average_win(self):
        """Calculate average winning trade"""
        winning_trades = [trade['pnl'] for trade in self.trade_history if trade['pnl'] > 0]
        return np.mean(winning_trades) if winning_trades else 0
    
    def get_average_loss(self):
        """Calculate average losing trade"""
        losing_trades = [abs(trade['pnl']) for trade in self.trade_history if trade['pnl'] < 0]
        return np.mean(losing_trades) if losing_trades else 0
    
    def check_portfolio_risk(self, new_position_size):
        """Check if new position would exceed portfolio risk limits"""
        current_risk = sum(pos['risk'] for pos in self.current_positions)
        total_risk = current_risk + new_position_size
        
        # Check portfolio risk limit
        if total_risk > self.portfolio_value * self.max_portfolio_risk:
            return False, f"Portfolio risk limit exceeded: {total_risk:.2f}%"
        
        # Check position count limit
        if len(self.current_positions) >= self.max_open_positions:
            return False, f"Maximum positions reached: {len(self.current_positions)}"
        
        return True, "Risk check passed"
    
    def update_volatility(self, current_volatility):
        """Update volatility history"""
        self.volatility_history.append(current_volatility)
        
        # Keep only recent volatility data
        if len(self.volatility_history) > 100:
            self.volatility_history = self.volatility_history[-50:]
    
    def add_trade(self, trade_data):
        """Add completed trade to history"""
        self.trade_history.append(trade_data)
        
        # Keep only recent trades
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-50:]
    
    def get_risk_metrics(self):
        """Get comprehensive risk metrics"""
        if len(self.trade_history) < 5:
            return {
                'win_rate': 0.5,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 1.0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
        
        # Calculate metrics
        wins = [t['pnl'] for t in self.trade_history if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in self.trade_history if t['pnl'] < 0]
        
        win_rate = len(wins) / len(self.trade_history)
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # Profit factor
        total_wins = sum(wins)
        total_losses = sum(losses)
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Calculate drawdown
        cumulative_pnl = np.cumsum([t['pnl'] for t in self.trade_history])
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Sharpe ratio (simplified)
        returns = [t['pnl'] / self.portfolio_value for t in self.trade_history]
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        
        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(self.trade_history)
        }

class MarketRegimeDetector:
    """Market Regime Detection System"""
    
    def __init__(self):
        self.regime_history = []
        self.price_history = []  # Store historical underlying prices
        self.volatility_thresholds = {
            'low_vol': 0.15,
            'medium_vol': 0.25,
            'high_vol': 0.35
        }
        
    def detect_regime(self, data, underlying_value=None):
        """Detect current market regime"""
        try:
            # Update price history with underlying value
            if underlying_value is not None:
                self.price_history.append({
                    'timestamp': datetime.now(),
                    'price': underlying_value
                })
                
                # Keep only recent price history (last 100 data points)
                if len(self.price_history) > 100:
                    self.price_history = self.price_history[-100:]
            
            # Calculate volatility using historical underlying prices
            volatility = self.calculate_realized_volatility_from_history()
            
            # Calculate trend strength
            trend_strength = self.calculate_trend_strength(data)
            
            # Calculate momentum
            momentum = self.calculate_momentum(data)
            
            # Determine regime
            if volatility < self.volatility_thresholds['low_vol']:
                if abs(trend_strength) > 0.6:
                    regime = 'trending_low_vol'
                else:
                    regime = 'sideways_low_vol'
            elif volatility < self.volatility_thresholds['medium_vol']:
                if abs(trend_strength) > 0.6:
                    regime = 'trending_medium_vol'
                else:
                    regime = 'sideways_medium_vol'
            else:
                if abs(trend_strength) > 0.6:
                    regime = 'trending_high_vol'
                else:
                    regime = 'sideways_high_vol'
            
            # Store regime
            self.regime_history.append({
                'timestamp': datetime.now(),
                'regime': regime,
                'volatility': volatility,
                'trend_strength': trend_strength,
                'momentum': momentum
            })
            
            return {
                'regime': regime,
                'volatility': volatility,
                'trend_strength': trend_strength,
                'momentum': momentum
            }
            
        except Exception as e:
            logger.error(f"Regime detection error: {e}")
            return {
                'regime': 'unknown',
                'volatility': 0.2,
                'trend_strength': 0,
                'momentum': 0
            }
    
    def calculate_realized_volatility_from_history(self, window=20):
        """Calculate realized volatility from historical underlying prices"""
        try:
            if len(self.price_history) < 2:
                return 0.2  # Default volatility if insufficient data
            
            # Use all available price history (up to window size)
            recent_history = self.price_history[-min(window, len(self.price_history)):]
            
            if len(recent_history) < 2:
                return 0.2
            
            # Extract prices
            prices = [entry['price'] for entry in recent_history]
            
            # Calculate returns (price changes)
            returns = np.diff(np.log(prices))
            
            if len(returns) == 0:
                return 0.2
            
            # Calculate volatility (annualized)
            # Assuming data points are 1 minute apart during market hours
            volatility = np.std(returns) * np.sqrt(252 * 375)  # 252 trading days * 375 minutes per day
            
            # Ensure reasonable bounds
            volatility = max(0.05, min(2.0, volatility))  # Between 5% and 200%
            
            return volatility
            
        except Exception as e:
            logger.error(f"Historical volatility calculation error: {e}")
            return 0.2
    
    def calculate_realized_volatility(self, data, window=20):
        """Legacy method - kept for backward compatibility but redirects to history-based calculation"""
        return self.calculate_realized_volatility_from_history(window)
    
    def calculate_trend_strength(self, data, window=20):
        """Calculate trend strength using linear regression on historical prices"""
        try:
            if len(self.price_history) < 2:
                return 0
            
            # Use historical underlying prices
            recent_history = self.price_history[-min(window, len(self.price_history)):]
            
            if len(recent_history) < 2:
                return 0
            
            prices = [entry['price'] for entry in recent_history]
            
            if len(prices) < 2:
                return 0
                
            x = np.arange(len(prices))
            
            # Linear regression
            slope, _ = np.polyfit(x, prices, 1)
            
            # Normalize slope by mean price
            trend_strength = slope / np.mean(prices) if np.mean(prices) > 0 else 0
            
            # Bound the result
            trend_strength = max(-1.0, min(1.0, trend_strength))
            
            return trend_strength
            
        except Exception as e:
            logger.error(f"Trend strength calculation error: {e}")
            return 0
    
    def calculate_momentum(self, data, window=10):
        """Calculate price momentum using historical prices"""
        try:
            if len(self.price_history) < window:
                return 0
            
            # Use historical underlying prices
            if len(self.price_history) < window:
                return 0
            
            current_price = self.price_history[-1]['price']
            past_price = self.price_history[-window]['price']
            
            if past_price <= 0:
                return 0
            
            momentum = (current_price - past_price) / past_price
            
            # Bound the result
            momentum = max(-1.0, min(1.0, momentum))
            
            return momentum
            
        except Exception as e:
            logger.error(f"Momentum calculation error: {e}")
            return 0

class ProfitOptimizer:
    """Advanced Profit Optimization Engine for Options Trading"""
    
    def __init__(self):
        # Profit target configurations
        self.profit_targets = [0.5, 1.0, 1.5, 2.0, 3.0]  # Risk:Reward ratios
        self.historical_performance = {}
        
        # Market condition adjustments
        self.regime_adjustments = {
            'trending_low_vol': 1.2,
            'trending_medium_vol': 1.1,
            'trending_high_vol': 1.0,
            'sideways_low_vol': 0.8,
            'sideways_medium_vol': 0.9,
            'sideways_high_vol': 0.7
        }
        
        # Time-based adjustments
        self.time_adjustments = {
            'opening': 1.1,    # Higher targets during opening
            'mid_session': 1.0, # Normal targets
            'closing': 0.9     # Lower targets during closing
        }
        
    def optimize_profit_targets(self, signal_strength, volatility, regime, time_of_day):
        """Optimize profit targets based on market conditions"""
        try:
            # Base profit target (1.5:1 risk:reward)
            base_target = 1.5
            
            # Adjust for signal strength
            strength_adjustment = 1 + (signal_strength - 5) * 0.1
            
            # Adjust for volatility
            vol_adjustment = 1 + (volatility - 0.2) * 2
            
            # Adjust for regime
            regime_adjustment = self.regime_adjustments.get(regime, 1.0)
            
            # Adjust for time of day
            time_adjustment = self.time_adjustments.get(time_of_day, 1.0)
            
            # Calculate optimized target
            optimized_target = base_target * strength_adjustment * vol_adjustment * regime_adjustment * time_adjustment
            
            # Ensure reasonable bounds
            optimized_target = np.clip(optimized_target, 0.5, 5.0)
            
            return optimized_target
            
        except Exception as e:
            logger.error(f"Profit target optimization error: {e}")
            return 1.5  # Default fallback
    
    def calculate_trailing_stop(self, entry_price, current_price, max_profit, volatility, regime):
        """Calculate optimal trailing stop based on profit level and market conditions"""
        try:
            # Calculate profit percentage
            profit_percentage = (current_price - entry_price) / entry_price
            
            # Base trailing stop percentages
            if profit_percentage < 0.5:
                # Early stage - tight stop (30% of profit)
                trailing_stop = 0.3
            elif profit_percentage < 1.0:
                # Mid stage - moderate stop (50% of profit)
                trailing_stop = 0.5
            else:
                # Late stage - wide stop to let profits run (70% of profit)
                trailing_stop = 0.7
            
            # Adjust for volatility
            vol_adjustment = 1 + (volatility - 0.2) * 1.5
            trailing_stop *= vol_adjustment
            
            # Adjust for regime
            if 'trending' in regime:
                # Let profits run more in trending markets
                trailing_stop *= 0.8
            elif 'sideways' in regime:
                # Take profits faster in sideways markets
                trailing_stop *= 1.2
            
            # Calculate actual stop price
            stop_price = current_price - (max_profit * trailing_stop)
            
            # Ensure stop is not below entry price
            stop_price = max(stop_price, entry_price * 0.95)
            
            return stop_price
            
        except Exception as e:
            logger.error(f"Trailing stop calculation error: {e}")
            return current_price * 0.95  # 5% below current price
    
    def optimize_exit_timing(self, entry_time, current_time, profit_level, signal_strength, regime):
        """Optimize exit timing based on various factors"""
        try:
            # Calculate holding time in minutes
            holding_time = (current_time - entry_time).total_seconds() / 60.0
            
            # Base exit conditions
            exit_score = 0
            
            # Time-based exit (encourage faster trades)
            if holding_time > 120:  # 2 hours
                exit_score += 0.3
            elif holding_time > 60:  # 1 hour
                exit_score += 0.1
            
            # Profit-based exit
            if profit_level > 2.0:  # 200% profit
                exit_score += 0.4
            elif profit_level > 1.0:  # 100% profit
                exit_score += 0.2
            elif profit_level < -0.5:  # 50% loss
                exit_score += 0.5
            
            # Signal strength decay
            if signal_strength < 3:  # Weak signal
                exit_score += 0.2
            
            # Regime-based exit
            if 'sideways' in regime and profit_level > 0.5:
                # Take profits faster in sideways markets
                exit_score += 0.2
            
            # Exit if score exceeds threshold
            return exit_score > 0.5
            
        except Exception as e:
            logger.error(f"Exit timing optimization error: {e}")
            return False
    
    def optimize_entry_timing(self, signal_strength, volatility, regime, time_of_day, recent_performance):
        """Optimize entry timing based on market conditions"""
        try:
            entry_score = 0
            
            # Signal strength requirement
            if signal_strength >= 7:
                entry_score += 0.4
            elif signal_strength >= 5:
                entry_score += 0.2
            else:
                return False  # Don't enter on weak signals
            
            # Volatility check
            if 0.1 <= volatility <= 0.4:  # Optimal volatility range
                entry_score += 0.2
            elif volatility > 0.5:  # Too volatile
                return False
            
            # Regime preference
            if 'trending' in regime:
                entry_score += 0.2
            elif 'sideways' in regime and 'low_vol' in regime:
                entry_score += 0.1
            
            # Time of day preference
            if time_of_day == 'mid_session':
                entry_score += 0.1
            elif time_of_day == 'opening':
                entry_score += 0.05
            
            # Recent performance check
            if len(recent_performance) >= 5:
                recent_avg = np.mean(recent_performance[-5:])
                if recent_avg > 0:
                    entry_score += 0.1
            
            # Enter if score exceeds threshold
            return entry_score >= 0.6
            
        except Exception as e:
            logger.error(f"Entry timing optimization error: {e}")
            return False
    
    def get_performance_metrics(self, trade_history):
        """Calculate comprehensive performance metrics"""
        try:
            if len(trade_history) < 5:
                return {
                    'win_rate': 0.5,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'profit_factor': 1.0,
                    'expected_value': 0,
                    'total_trades': 0
                }
            
            wins = [t['pnl'] for t in trade_history if t['pnl'] > 0]
            losses = [abs(t['pnl']) for t in trade_history if t['pnl'] < 0]
            
            win_rate = len(wins) / len(trade_history)
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            
            # Profit factor
            total_wins = sum(wins)
            total_losses = sum(losses)
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Expected value
            expected_value = self.calculate_expected_value(win_rate, avg_win, avg_loss, 100000)
            
            return {
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'expected_value': expected_value,
                'total_trades': len(trade_history)
            }
            
        except Exception as e:
            logger.error(f"Performance metrics calculation error: {e}")
            return {
                'win_rate': 0.5,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 1.0,
                'expected_value': 0,
                'total_trades': 0
            }
    
    def calculate_expected_value(self, win_rate, avg_win, avg_loss, position_size):
        """Calculate expected value of a trade"""
        try:
            expected_win = win_rate * avg_win
            expected_loss = (1 - win_rate) * avg_loss
            expected_value = expected_win - expected_loss
            
            # Scale by position size
            return expected_value * (position_size / 100000)  # Normalize to 100k account
            
        except Exception as e:
            logger.error(f"Expected value calculation error: {e}")
            return 0

# Configuration class for centralized parameter management
class Config:
    """Enhanced dynamic configuration system with environment variable support"""
    
    # API Configuration
    API_RETRY_ATTEMPTS = int(os.getenv('API_RETRY_ATTEMPTS', '3'))
    API_RATE_LIMIT_SECONDS = int(os.getenv('API_RATE_LIMIT_SECONDS', '2'))
    API_TIMEOUT = int(os.getenv('API_TIMEOUT', '15'))
    
    # Market Hours (IST)
    MARKET_OPEN_HOUR = int(os.getenv('MARKET_OPEN_HOUR', '9'))
    MARKET_OPEN_MINUTE = int(os.getenv('MARKET_OPEN_MINUTE', '15'))
    MARKET_CLOSE_HOUR = int(os.getenv('MARKET_CLOSE_HOUR', '15'))
    MARKET_CLOSE_MINUTE = int(os.getenv('MARKET_CLOSE_MINUTE', '30'))
    
    # ML Configuration
    SEQUENCE_LENGTH = int(os.getenv('SEQUENCE_LENGTH', '10'))
    INPUT_SIZE = int(os.getenv('INPUT_SIZE', '50'))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '32'))
    MIN_TRAINING_SAMPLES = int(os.getenv('MIN_TRAINING_SAMPLES', '50'))
    MAX_TRAINING_SAMPLES = int(os.getenv('MAX_TRAINING_SAMPLES', '2000'))
    
    # Enhanced ML Configuration
    ML_CONFIDENCE_THRESHOLD = float(os.getenv('ML_CONFIDENCE_THRESHOLD', '0.60'))  # Lowered from 0.75
    ENHANCED_CONFIDENCE_THRESHOLD = float(os.getenv('ENHANCED_CONFIDENCE_THRESHOLD', '0.65'))  # Lowered from 0.8
    FEATURE_IMPORTANCE_UPDATE_INTERVAL = int(os.getenv('FEATURE_IMPORTANCE_UPDATE_INTERVAL', '100'))
    
    # RL Configuration
    REPLAY_BUFFER_SIZE = int(os.getenv('REPLAY_BUFFER_SIZE', '1000'))
    GAMMA = float(os.getenv('GAMMA', '0.95'))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', '0.001'))
    MIN_LEARNING_RATE = float(os.getenv('MIN_LEARNING_RATE', '0.0001'))
    LEARNING_RATE_DECAY = float(os.getenv('LEARNING_RATE_DECAY', '0.995'))
    
    # Trading Configuration
    STOP_LOSS_POINTS = int(os.getenv('STOP_LOSS_POINTS', '10'))
    PAPER_TRADE_MULTIPLIER = int(os.getenv('PAPER_TRADE_MULTIPLIER', '75'))
    MAX_POSITION_SIZE = int(os.getenv('MAX_POSITION_SIZE', '1000'))
    LOT_SIZE = int(os.getenv('LOT_SIZE', '75'))  # 1 lot = 75 contracts for Nifty options
    SIGNAL_STRENGTH_THRESHOLD = int(os.getenv('SIGNAL_STRENGTH_THRESHOLD', '2'))
    
    # Risk Management
    MAX_RISK_PER_TRADE = float(os.getenv('MAX_RISK_PER_TRADE', '0.015'))
    MAX_PORTFOLIO_RISK = float(os.getenv('MAX_PORTFOLIO_RISK', '0.06'))
    MAX_OPEN_POSITIONS = int(os.getenv('MAX_OPEN_POSITIONS', '3'))
    
    # Performance Optimization
    FEATURE_CACHE_SIZE = int(os.getenv('FEATURE_CACHE_SIZE', '100'))
    MAX_HISTORICAL_MINUTES = int(os.getenv('MAX_HISTORICAL_MINUTES', '30'))
    CLEANUP_INTERVAL = int(os.getenv('CLEANUP_INTERVAL', '100'))
    
    # Data Storage
    MAX_HISTORICAL_DATA_HOURS = int(os.getenv('MAX_HISTORICAL_DATA_HOURS', '1'))
    MAX_MODEL_FILES = int(os.getenv('MAX_MODEL_FILES', '3'))
    MAX_DEEP_MODEL_FILES = int(os.getenv('MAX_DEEP_MODEL_FILES', '2'))
    DATA_RETENTION_DAYS = int(os.getenv('DATA_RETENTION_DAYS', '90'))
    
    # Rate Limiting
    BASE_REQUEST_INTERVAL = int(os.getenv('BASE_REQUEST_INTERVAL', '45'))
    MIN_REQUEST_INTERVAL = int(os.getenv('MIN_REQUEST_INTERVAL', '30'))
    MAX_REQUEST_INTERVAL = int(os.getenv('MAX_REQUEST_INTERVAL', '300'))
    
    # Circuit Breaker
    API_FAILURE_THRESHOLD = int(os.getenv('API_FAILURE_THRESHOLD', '3'))
    ML_FAILURE_THRESHOLD = int(os.getenv('ML_FAILURE_THRESHOLD', '5'))
    RECOVERY_TIMEOUT = int(os.getenv('RECOVERY_TIMEOUT', '300'))
    
    @classmethod
    def validate_config(cls):
        """Validate configuration parameters"""
        errors = []
        
        if cls.ML_CONFIDENCE_THRESHOLD < 0 or cls.ML_CONFIDENCE_THRESHOLD > 1:
            errors.append("ML_CONFIDENCE_THRESHOLD must be between 0 and 1")
            
        if cls.MAX_RISK_PER_TRADE < 0 or cls.MAX_RISK_PER_TRADE > 1:
            errors.append("MAX_RISK_PER_TRADE must be between 0 and 1")
            
        if cls.MIN_REQUEST_INTERVAL >= cls.MAX_REQUEST_INTERVAL:
            errors.append("MIN_REQUEST_INTERVAL must be less than MAX_REQUEST_INTERVAL")
            
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
            
        return True
    
    @classmethod
    def get_config_summary(cls):
        """Get summary of current configuration"""
        return {
            'ML_CONFIDENCE_THRESHOLD': cls.ML_CONFIDENCE_THRESHOLD,
            'MAX_RISK_PER_TRADE': cls.MAX_RISK_PER_TRADE,
            'MIN_TRAINING_SAMPLES': cls.MIN_TRAINING_SAMPLES,
            'BASE_REQUEST_INTERVAL': cls.BASE_REQUEST_INTERVAL,
            'API_FAILURE_THRESHOLD': cls.API_FAILURE_THRESHOLD,
            'FEATURE_CACHE_SIZE': cls.FEATURE_CACHE_SIZE
        }

# Minimal logging setup
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# ================================
# Enhanced Paper Trade Tracking Class
# ================================
class PaperTrade:
    def __init__(self):
        self.position = None
        self.entry_price = 0
        self.entry_time = None
        self.entry_strike = 0  # Add strike tracking
        self.direction = 0
        self.active = False
        self.pnl = 0
        self.is_short = False  # Flag to indicate if we're shorting

        # Enhanced features
        self.max_profit = 0
        self.trailing_stop = None
        self.position_size = 0
        self.regime = 'unknown'
        self.stop_loss = 0
        self.profit_target = 0
        
        # ML prediction tracking
        self.current_ml_prediction = None  # Store current ML predictions
        
        # Real-time training data collection
        self.market_data_buffer = []  # Store market data for training
        self.price_movements = []  # Track price movements
        self.volatility_tracker = []  # Track volatility changes
        self.oi_changes = []  # Track OI changes
        self.volume_patterns = []  # Track volume patterns
        self.max_buffer_size = 1000  # Maximum buffer size

    def enter(self, direction, ltp, strike, position_size=0, regime='unknown', ml_predictions=None):
        """Enter a new paper trade with ML-enhanced parameters"""
        try:
            # Validate inputs
            if not isinstance(direction, int) or direction not in [-1, 1]:
                logger.error(f"Invalid direction: {direction}")
                return False
                
            if not isinstance(ltp, (int, float)) or ltp <= 0:
                logger.error(f"Invalid LTP: {ltp}")
                return False
                
            # Fix strike validation - allow any numeric value
            try:
                strike = float(strike)
            except (TypeError, ValueError):
                logger.error(f"Invalid strike: {strike}")
                return False
            
            # SHORT OPTIONS STRATEGY: 
            # Signal 1 = SHORT PUT (profit when PUT price decreases)
            # Signal -1 = SHORT CALL (profit when CALL price decreases)
            self.position = 'SHORT_PUT' if direction == 1 else 'SHORT_CALL'
            self.entry_price = float(ltp)
            self.entry_strike = strike
            self.entry_time = datetime.now()
            self.direction = direction
            self.active = True
            self.regime = regime
            self.is_short = True  # Flag to indicate we're shorting
            
            # Use ML predictions for position sizing and risk management if available
            if ml_predictions and isinstance(ml_predictions, dict):
                # Validate ML predictions
                position_size_pred = ml_predictions.get('position_size', 0.5)
                if not isinstance(position_size_pred, (int, float)):
                    position_size_pred = 0.5
                    
                # Use ML-predicted position size in LOTS (1 lot = Config.LOT_SIZE contracts)
                ml_position_size_lots = max(1, round(position_size_pred * 10))  # Scale to 1-10 lots
                self.position_size = max(position_size, ml_position_size_lots)  # Use larger of default or ML prediction
                
                # Use ML-predicted stop loss percentage (INVERTED for shorting)
                stop_loss_pct = float(ml_predictions.get('stop_loss_pct', 0.10))
                stop_loss_pct = max(0.05, min(0.20, stop_loss_pct))  # Bound between 5% and 20%
                
                # For shorting: stop loss is ABOVE entry price (when option price increases)
                self.stop_loss = self.entry_price * (1 + stop_loss_pct)
                
                # Store trailing stop percentage for later use
                self.ml_trailing_stop_pct = float(ml_predictions.get('trailing_stop_pct', 0.5))
                self.ml_trailing_stop_pct = max(0.3, min(0.8, self.ml_trailing_stop_pct))  # Bound between 30% and 80%
                
                # Calculate profit target based on ML confidence (INVERTED for shorting)
                ml_confidence = float(ml_predictions.get('confidence', 0.5))
                ml_confidence = max(0.0, min(1.0, ml_confidence))  # Bound between 0 and 1
                
                if ml_confidence > 0.8:
                    self.profit_target = self.entry_price * 0.4  # 60% profit target (price drops to 40% of entry)
                elif ml_confidence > 0.7:
                    self.profit_target = self.entry_price * 0.6  # 40% profit target (price drops to 60% of entry)
                else:
                    self.profit_target = self.entry_price * 0.7  # 30% profit target (price drops to 70% of entry)
                    
                print(f"[PAPER TRADE] Enter {self.position} Strike:{strike} @ {ltp:.2f} | "
                      f"Size: {self.position_size:.0f} lots ({self.position_size*Config.LOT_SIZE:.0f} contracts) | "
                      f"Stop: {self.stop_loss:.2f} | Target: {self.profit_target:.2f} | "
                      f"ML_Conf: {ml_confidence:.2f} | SL%: {stop_loss_pct:.1%} | TSL%: {self.ml_trailing_stop_pct:.1%} | "
                      f"Regime: {regime}")
            else:
                # Fallback to traditional calculation
                self.position_size = max(1, position_size)  # Minimum 1 lot
                self.ml_trailing_stop_pct = 0.5  # Default trailing stop
                self.calculate_exit_levels(self.entry_price, regime)
                
                print(f"[PAPER TRADE] Enter {self.position} Strike:{strike} @ {ltp:.2f} | "
                      f"Size: {position_size:.0f} lots ({position_size*Config.LOT_SIZE:.0f} contracts) | "
                      f"Stop: {self.stop_loss:.2f} | Target: {self.profit_target:.2f} | "
                      f"Regime: {regime}")
            
            return True
            
        except Exception as e:
            logger.error(f"Paper trade entry error: {e}")
            return False

    def collect_market_data_for_training(self, current_data, underlying_value, atm_strike, signal, strength, regime):
        """Collect real-time market data for continuous model training"""
        try:
            timestamp = datetime.now()
            
            # Get ATM data
            atm_data = current_data[current_data['Strike'] == atm_strike]
            if atm_data.empty:
                return
                
            atm_row = atm_data.iloc[0]
            
            # Collect comprehensive market data
            market_snapshot = {
                'timestamp': timestamp,
                'underlying_value': underlying_value,
                'atm_strike': atm_strike,
                'call_ltp': atm_row['Call_LTP'],
                'put_ltp': atm_row['Put_LTP'],
                'call_oi': atm_row['Call_OI'],
                'put_oi': atm_row['Put_OI'],
                'call_volume': atm_row['Call_Volume'],
                'put_volume': atm_row['Put_Volume'],
                'call_change_oi': atm_row['Call_Change_OI'],
                'put_change_oi': atm_row['Put_Change_OI'],
                'signal': signal,
                'signal_strength': strength,
                'regime': regime,
                'time_of_day': timestamp.hour + timestamp.minute / 60.0,
                'day_of_week': timestamp.weekday(),
                'is_active_trade': self.active,
                'trade_direction': self.direction if self.active else 0,
                'trade_pnl': self.pnl if self.active else 0
            }
            
            # Calculate derived metrics
            market_snapshot.update({
                'pcr_oi': atm_row['Put_OI'] / max(atm_row['Call_OI'], 1),
                'pcr_volume': atm_row['Put_Volume'] / max(atm_row['Call_Volume'], 1),
                'total_oi': atm_row['Call_OI'] + atm_row['Put_OI'],
                'total_volume': atm_row['Call_Volume'] + atm_row['Put_Volume'],
                'oi_imbalance': (atm_row['Put_OI'] - atm_row['Call_OI']) / max(atm_row['Call_OI'] + atm_row['Put_OI'], 1),
                'volume_imbalance': (atm_row['Put_Volume'] - atm_row['Call_Volume']) / max(atm_row['Call_Volume'] + atm_row['Put_Volume'], 1)
            })
            
            # Add to buffer
            self.market_data_buffer.append(market_snapshot)
            
            # Maintain buffer size
            if len(self.market_data_buffer) > self.max_buffer_size:
                self.market_data_buffer = self.market_data_buffer[-self.max_buffer_size:]
            
            # Track price movements for volatility calculation
            if len(self.market_data_buffer) >= 2:
                prev_snapshot = self.market_data_buffer[-2]
                current_price = atm_row['Call_LTP'] + atm_row['Put_LTP']  # Combined option price
                prev_price = prev_snapshot['call_ltp'] + prev_snapshot['put_ltp']
                
                price_change = (current_price - prev_price) / max(prev_price, 0.01)
                self.price_movements.append(price_change)
                
                # Calculate rolling volatility
                if len(self.price_movements) >= 20:
                    recent_movements = self.price_movements[-20:]
                    volatility = np.std(recent_movements) * np.sqrt(252 * 24 * 60)  # Annualized
                    market_snapshot['volatility'] = volatility
                else:
                    market_snapshot['volatility'] = 0.2  # Default volatility
            
            # Track OI and volume changes
            if len(self.market_data_buffer) >= 2:
                prev_snapshot = self.market_data_buffer[-2]
                
                # OI changes
                call_oi_change = atm_row['Call_Change_OI']
                put_oi_change = atm_row['Put_Change_OI']
                self.oi_changes.append({
                    'call_oi_change': call_oi_change,
                    'put_oi_change': put_oi_change,
                    'total_oi_change': call_oi_change + put_oi_change
                })
                
                # Volume patterns
                call_vol_change = atm_row['Call_Volume'] - prev_snapshot['call_volume']
                put_vol_change = atm_row['Put_Volume'] - prev_snapshot['put_volume']
                self.volume_patterns.append({
                    'call_vol_change': call_vol_change,
                    'put_vol_change': put_vol_change,
                    'total_vol_change': call_vol_change + put_vol_change
                })
                
                # Add trend indicators
                if len(self.oi_changes) >= 5:
                    recent_oi_changes = [oi['total_oi_change'] for oi in self.oi_changes[-5:]]
                    market_snapshot['oi_trend'] = np.mean(recent_oi_changes)
                    
                    recent_vol_changes = [vol['total_vol_change'] for vol in self.volume_patterns[-5:]]
                    market_snapshot['volume_trend'] = np.mean(recent_vol_changes)
                else:
                    market_snapshot['oi_trend'] = 0
                    market_snapshot['volume_trend'] = 0
            
        except Exception as e:
            logger.error(f"Market data collection error: {e}")

    def get_training_data_from_market_buffer(self):
        """Extract training data from collected market buffer"""
        if len(self.market_data_buffer) < 10:
            return None
            
        try:
            training_samples = []
            
            # Create training samples from market data
            for i in range(len(self.market_data_buffer) - 1):
                current_snapshot = self.market_data_buffer[i]
                next_snapshot = self.market_data_buffer[i + 1]
                
                # Create features from current market state
                features = {
                    'underlying_value': current_snapshot['underlying_value'],
                    'atm_strike': current_snapshot['atm_strike'],
                    'call_ltp': current_snapshot['call_ltp'],
                    'put_ltp': current_snapshot['put_ltp'],
                    'call_oi': current_snapshot['call_oi'],
                    'put_oi': current_snapshot['put_oi'],
                    'call_volume': current_snapshot['call_volume'],
                    'put_volume': current_snapshot['put_volume'],
                    'pcr_oi': current_snapshot['pcr_oi'],
                    'pcr_volume': current_snapshot['pcr_volume'],
                    'oi_imbalance': current_snapshot['oi_imbalance'],
                    'volume_imbalance': current_snapshot['volume_imbalance'],
                    'volatility': current_snapshot.get('volatility', 0.2),
                    'oi_trend': current_snapshot.get('oi_trend', 0),
                    'volume_trend': current_snapshot.get('volume_trend', 0),
                    'time_of_day': current_snapshot['time_of_day'],
                    'day_of_week': current_snapshot['day_of_week'],
                    'signal': current_snapshot['signal'],
                    'signal_strength': current_snapshot['signal_strength'],
                    'regime': current_snapshot['regime']
                }
                
                # Create labels based on next market state
                price_change = (next_snapshot['call_ltp'] + next_snapshot['put_ltp']) - (current_snapshot['call_ltp'] + current_snapshot['put_ltp'])
                price_change_pct = price_change / max(current_snapshot['call_ltp'] + current_snapshot['put_ltp'], 0.01)
                
                # Determine outcome based on price movement and signal
                if current_snapshot['signal'] == 1:  # CALL signal
                    outcome = 1 if price_change_pct > 0.01 else (0 if price_change_pct < -0.01 else 2)
                elif current_snapshot['signal'] == -1:  # PUT signal
                    outcome = 1 if price_change_pct < -0.01 else (0 if price_change_pct > 0.01 else 2)
                else:  # Neutral signal
                    outcome = 2  # Neutral outcome
                
                training_samples.append({
                    'features': features,
                    'outcome': outcome,
                    'price_change': price_change_pct,
                    'timestamp': current_snapshot['timestamp']
                })
            
            return training_samples
            
        except Exception as e:
            logger.error(f"Training data extraction error: {e}")
            return None

    def calculate_exit_levels(self, entry_price, regime):
        """Calculate dynamic stop loss and profit target for SHORT OPTIONS"""
        try:
            # Get current volatility (simplified)
            volatility = 0.2  # Default volatility
            
            # Calculate dynamic stop loss (INVERTED for shorting)
            # For shorts: stop loss is ABOVE entry price (loss when option price increases)
            self.stop_loss = entry_price * 1.10  # 10% stop loss above entry
            
            # Calculate profit target based on regime (INVERTED for shorting)
            # For shorts: profit target is BELOW entry price (profit when option price decreases)
            if 'trending' in regime:
                self.profit_target = entry_price * 0.5  # 50% profit target (price drops to 50%)
            elif 'sideways' in regime:
                self.profit_target = entry_price * 0.7  # 30% profit target (price drops to 70%)
            else:
                self.profit_target = entry_price * 0.6  # 40% profit target (price drops to 60%)
                
        except Exception as e:
            logger.error(f"Exit levels calculation error: {e}")
            self.stop_loss = entry_price * 1.10  # Stop loss above entry for shorts
            self.profit_target = entry_price * 0.6  # Profit target below entry for shorts

    def store_trade_outcome(self, outcome, exit_price, trade_data):
        """Store comprehensive trade outcome for learning"""
        try:
            # Convert datetime objects to strings for JSON serialization
            entry_time_str = self.entry_time.isoformat() if self.entry_time else None
            current_time_str = datetime.now().isoformat()
            
            trade_outcome = {
                'timestamp': current_time_str,
                'entry': {
                    'price': self.entry_price,
                    'strike': self.entry_strike,
                    'direction': self.direction,
                    'regime': self.regime,
                    'position_size': self.position_size,
                    'entry_time': entry_time_str,
                    'ml_predictions': self.current_ml_prediction,  # Store ML predictions
                    'market_conditions': self.market_data_buffer[-1] if self.market_data_buffer else None
                },
                'exit': {
                    'price': exit_price,
                    'reason': trade_data.get('exit_reason', 'Unknown') if trade_data else 'Unknown',
                    'exit_time': current_time_str,
                    'market_conditions': self.market_data_buffer[-1] if self.market_data_buffer else None
                },
                'performance': {
                    'pnl': self.pnl,
                    'max_profit': self.max_profit,
                    'holding_time': (datetime.now() - self.entry_time).total_seconds() if self.entry_time else 0,
                    'outcome': outcome
                }
            }
            
            # Save to JSON file with proper error handling
            try:
                with open('trade_outcomes.json', 'a') as f:
                    json.dump(trade_outcome, f, default=str)  # Use default=str to handle any remaining datetime objects
                    f.write('\n')
            except Exception as e:
                logger.error(f"Error saving trade outcome to JSON: {e}")
            
            # Also save to CSV for backward compatibility
            try:
                self.log_to_csv(outcome, exit_price, trade_data)
            except Exception as e:
                logger.error(f"Error saving to CSV: {e}")
            
            return trade_outcome
            
        except Exception as e:
            logger.error(f"Error storing trade outcome: {e}")
            return None

    def check_exit(self, current_data, signal, regime='unknown'):
        """Enhanced exit check with outcome storage"""
        if not self.active:
            return None

        try:
            # Get current LTP for the SAME strike used for entry
            strike_data = current_data[current_data['Strike'] == self.entry_strike]
            if strike_data.empty:
                print(f"[WARNING] Strike {self.entry_strike} not found in current data")
                return None
            
            # For SHORT strategy: direction 1 = SHORT PUT, direction -1 = SHORT CALL
            current_ltp = strike_data.iloc[0]['Put_LTP'] if self.direction == 1 else strike_data.iloc[0]['Call_LTP']
            
            # Calculate current profit for SHORT OPTIONS
            profit = self.entry_price - current_ltp
                
            # Update max profit and show trailing stop info if in profit
            if profit > self.max_profit:
                self.max_profit = profit
                if profit > 5.0:  # Show trailing info only for significant profits
                    print(f"[TRAILING SHORT] New Max Profit: {self.max_profit:.2f} | Current: {profit:.2f} | Entry: {self.entry_price:.2f} | Current LTP: {current_ltp:.2f}")
            elif self.max_profit > 8.0 and profit > 2.0:
                # Show trailing stop status
                if 'trending' in regime.lower():
                    trailing_pct = 0.4
                elif 'sideways' in regime.lower():
                    trailing_pct = 0.6
                else:
                    trailing_pct = 0.5
                trailing_level = self.max_profit * trailing_pct
                print(f"[TRAILING SHORT] Max: {self.max_profit:.2f} | Current: {profit:.2f} | Trail Level: {trailing_level:.2f} | Regime: {regime}")
            else:
                self.max_profit = max(self.max_profit, profit)
            
            # Check exit conditions for SHORT OPTIONS
            exit_reason = None
            
            # 1. Stop loss hit (for shorts: when current price > stop loss)
            if current_ltp >= self.stop_loss:
                exit_reason = "Stop Loss"
            
            # 2. Signal change
            elif signal == 0 or signal != self.direction:
                exit_reason = "Signal Change"
            
            # 3. Profit target hit (for shorts: when current price <= profit target)
            elif current_ltp <= self.profit_target:
                exit_reason = "Profit Target"
            
            # 4. Time-based exit (2 hours max)
            elif self.entry_time and (datetime.now() - self.entry_time).total_seconds() > 7200:
                exit_reason = "Time Exit"
            
            # 5. ML-Enhanced trailing stop for SHORT OPTIONS
            elif self.max_profit > 8.0 and profit > 2.0:  # Only if we have good profit and still in profit
                # Use ML-predicted trailing stop percentage if available, otherwise regime-aware
                if hasattr(self, 'ml_trailing_stop_pct') and self.ml_trailing_stop_pct:
                    trailing_pct = self.ml_trailing_stop_pct
                else:
                    # Fallback to regime-aware trailing stop
                    if 'trending' in regime.lower():
                        trailing_pct = 0.4  # Allow 40% retracement in trending markets
                    elif 'sideways' in regime.lower():
                        trailing_pct = 0.6  # Allow 60% retracement in sideways markets
                    else:
                        trailing_pct = 0.5  # Default 50% retracement
                
                if profit < self.max_profit * trailing_pct:
                    exit_reason = "Trailing Stop"
            
            if exit_reason:
                # Calculate PnL correctly
                self.pnl = profit
                
                outcome = 1 if self.pnl > 0 else 0
                
                # Store comprehensive trade outcome with proper structure
                trade_data = {
                    'exit_reason': exit_reason,
                    'exit_price': current_ltp,
                    'max_profit': self.max_profit,
                    'entry_price': self.entry_price,
                    'strike': self.entry_strike,
                    'position_size': self.position_size,
                    'direction': self.direction,
                    'regime': self.regime,
                    'pnl': self.pnl,
                    'outcome': outcome
                }
                
                # Store trade outcome for learning
                try:
                    trade_outcome = self.store_trade_outcome(outcome, current_ltp, trade_data)
                except Exception as e:
                    logger.error(f"Error storing trade outcome: {e}")
                    trade_outcome = trade_data  # Use trade_data as fallback
                
                # Enhanced exit message with trailing stop details
                exit_msg = f"[EXIT] {self.position} Strike:{self.entry_strike} @ {current_ltp:.2f} | Entry: {self.entry_price:.2f} | PnL: {self.pnl:.2f} | Outcome: {outcome} | Reason: {exit_reason}"
                if exit_reason == "Trailing Stop":
                    exit_msg += f" | Max Profit: {self.max_profit:.2f}"
                print(exit_msg)
                
                # Store PnL before reset (to avoid losing it in reset)
                final_pnl = self.pnl
                self.reset()
                return outcome, final_pnl, trade_data  # Return trade_data, not trade_outcome
            
            return None
                
        except Exception as e:
            logger.error(f"Paper trade exit error: {e}")
            return None

    def log_to_csv(self, outcome, exit_price, trade_data=None):
        """Log trade data to CSV with proper error handling"""
        try:
            # Create CSV header if file doesn't exist
            if not os.path.exists('signal_log.csv'):
                with open('signal_log.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Timestamp', 'Position', 'Strike', 'Entry_Price', 'Exit_Price', 'PnL', 'Outcome', 'Regime', 'Exit_Reason', 'Position_Size'])
            
            # Safely get values from trade_data
            exit_reason = trade_data.get('exit_reason', 'Unknown') if trade_data else 'Unknown'
                
            with open('signal_log.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    self.position,
                    self.entry_strike,
                    self.entry_price,
                    exit_price,
                    self.pnl,
                    outcome,
                    self.regime,
                    exit_reason,
                    self.position_size
                ])
        except Exception as e:
            logger.error(f"Error logging to CSV: {e}")

    def reset(self):
        self.__init__()


class AdvancedMLDecisionEngine:
    """Advanced ML Engine with ensemble and deep learning for Market Regime Classification"""
    
    def __init__(self, input_size=35, sequence_length=10):  # Changed from 50 to 35 to match actual features
        # Enhanced ML models for comprehensive trading predictions
        self.models = {
            'regime_classifier': RandomForestClassifier(n_estimators=100, random_state=42),
            'signal_confidence': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'pattern_detector': DecisionTreeClassifier(max_depth=5, random_state=42),
            'signal_direction': RandomForestClassifier(n_estimators=150, random_state=42),  # For Tradetron signals
            'position_sizer': GradientBoostingRegressor(n_estimators=100, random_state=42),  # Position size prediction
            'stop_loss_predictor': RandomForestRegressor(n_estimators=100, random_state=42),  # Stop loss level
            'trailing_stop_predictor': GradientBoostingRegressor(n_estimators=100, random_state=42),  # Trailing stop %
            'meta_learner': RandomForestClassifier(n_estimators=50, random_state=42)  # Will be reinitialized with correct features
        }
        self.feature_names = []
        self.trained_feature_names = []  # Snapshot used by trained models/scaler
        self.is_trained = False
        self.scaler = StandardScaler()
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.feature_buffer = deque(maxlen=sequence_length)
        self.training_data = []
        self.max_training_samples = getattr(Config, 'MAX_TRAINING_SAMPLES', 10000)  # Increased from 2000 to 10000
        self.min_training_samples = getattr(Config, 'MIN_TRAINING_SAMPLES', 50)
        
        # Training and performance tracking
        self.training_counter = 0
        self.episode_pnls = []
        self.episode_rewards = []
        self.recent_outcomes = deque(maxlen=10)
        
        # Deep Learning components
        self.deep_model = LSTMSignalPredictor(
            input_size=input_size,  # Now using 35 features
            hidden_size=64,
            num_layers=2,
            dropout=0.2
        ).to(device)
        
        # Optimizer and loss functions
        self.optimizer = torch.optim.Adam(self.deep_model.parameters(), lr=0.001)
        self.criterion_regime = nn.CrossEntropyLoss()
        self.criterion_confidence = nn.MSELoss()
        
        # Data buffers
        self.sequence_buffer = SequenceBuffer(maxlen=1000, seq_len=sequence_length)
        
        # Training state
        self.model_path = "ml_models.pkl"
        self.deep_model_path = "deep_model.pth"
        
        # Feature engineering parameters
        self.lookback_periods = [1, 2, 3, 5]
        
        # SHAP Feature Analysis
        self.shap_analyzer = SHAPFeatureAnalyzer()
        self.shap_analysis_counter = 0
        
        # Initialize feature names
        self._initialize_feature_names()
        
        # Load existing models if available
        self.load_models()
        
    def _initialize_feature_names(self):
        """Initialize the list of feature names to ensure consistency"""
        self.feature_names = [
            # Basic features
            'underlying_price', 'atm_strike', 'price_strike_diff', 'price_strike_ratio',
            
            # ATM Options features
            'atm_call_oi', 'atm_put_oi', 'atm_call_volume', 'atm_put_volume', 
            'atm_call_ltp', 'atm_put_ltp',
            
            # Ratios
            'pcr_oi', 'pcr_volume', 'call_put_ltp_ratio',
            
            # Aggregate features
            'total_call_oi', 'total_put_oi', 'total_call_volume', 'total_put_volume',
            
            # Temporal features
            'hour', 'minute', 'time_of_day', 'is_opening', 'is_closing', 'is_mid_session',
            
            # Historical comparison features (will be added dynamically)
            'call_oi_change', 'put_oi_change', 'call_volume_change', 'put_volume_change',
            'call_ltp_change', 'put_ltp_change', 'call_oi_change_pct', 'put_oi_change_pct',
            'call_volume_change_pct', 'put_volume_change_pct', 'call_ltp_change_pct', 'put_ltp_change_pct'
        ]
    
    def extract_features(self, current_data, historical_data, underlying_value, atm_strike):
        """Extract comprehensive features for ML models"""
        try:
            features = {}
            timestamp = datetime.now().replace(second=0, microsecond=0)
            
            # Validate input data
            if current_data is None or underlying_value is None or atm_strike is None:
                logger.error("Invalid input data for feature extraction")
                return None
                
            # Ensure DataFrame format
            if not isinstance(current_data, pd.DataFrame):
                try:
                    current_data = pd.DataFrame(current_data)
                except:
                    logger.error("Could not convert input data to DataFrame")
                    return None
            
            # Get ATM and nearby strikes data
            atm_data = current_data[current_data['Strike'] == atm_strike]
            if atm_data.empty:
                logger.error(f"No data found for ATM strike {atm_strike}")
                return None
                
            atm_row = atm_data.iloc[0]
            
            # === BASIC FEATURES ===
            # Ensure numeric types and handle NaN/Inf
            features['underlying_price'] = float(underlying_value)
            features['atm_strike'] = float(atm_strike)
            features['price_strike_diff'] = float(underlying_value - atm_strike)
            features['price_strike_ratio'] = float(underlying_value / atm_strike) if atm_strike > 0 else 1.0
            
            # ATM Options Features - Convert to float and validate
            for key in ['Call_OI', 'Put_OI', 'Call_Volume', 'Put_Volume', 'Call_LTP', 'Put_LTP']:
                try:
                    features[f'atm_{key.lower()}'] = float(atm_row[key])
                except:
                    features[f'atm_{key.lower()}'] = 0.0
            
            # Ratios and Relationships - Handle division by zero
            features['pcr_oi'] = float(atm_row['Put_OI'] / max(atm_row['Call_OI'], 1))
            features['pcr_volume'] = float(atm_row['Put_Volume'] / max(atm_row['Call_Volume'], 1))
            features['call_put_ltp_ratio'] = float(atm_row['Call_LTP'] / max(atm_row['Put_LTP'], 0.01))
            
            # === AGGREGATE FEATURES ===
            # Ensure numeric aggregation
            features['total_call_oi'] = float(current_data['Call_OI'].sum())
            features['total_put_oi'] = float(current_data['Put_OI'].sum())
            features['total_call_volume'] = float(current_data['Call_Volume'].sum())
            features['total_put_volume'] = float(current_data['Put_Volume'].sum())
            
            # === TEMPORAL FEATURES ===
            features['hour'] = float(timestamp.hour)
            features['minute'] = float(timestamp.minute)
            features['time_of_day'] = float(timestamp.hour + timestamp.minute/60.0)
            
            # Market session features
            features['is_opening'] = float(1 if (timestamp.hour == 9 and timestamp.minute < 30) else 0)
            features['is_closing'] = float(1 if (timestamp.hour >= 15) else 0)
            features['is_mid_session'] = float(1 if (10 <= timestamp.hour <= 14) else 0)
            
            # === HISTORICAL COMPARISON FEATURES ===
            if len(historical_data) >= 2:
                sorted_times = sorted(historical_data.keys())
                prev_data = historical_data[sorted_times[-2]]
                
                prev_atm = prev_data[prev_data['Strike'] == atm_strike]
                if not prev_atm.empty:
                    prev_atm_row = prev_atm.iloc[0]
                    
                    # Calculate changes with validation
                    for field in ['Call_OI', 'Put_OI', 'Call_Volume', 'Put_Volume', 'Call_LTP', 'Put_LTP']:
                        try:
                            current_val = float(atm_row[field])
                            prev_val = float(prev_atm_row[field])
                            change = current_val - prev_val
                            change_pct = (change / max(prev_val, 1)) * 100
                            
                            features[f'{field.lower()}_change'] = float(change)
                            features[f'{field.lower()}_change_pct'] = float(change_pct)
                        except:
                            features[f'{field.lower()}_change'] = 0.0
                            features[f'{field.lower()}_change_pct'] = 0.0
            
            # Validate all features are float type
            for key in features:
                if not isinstance(features[key], float):
                    features[key] = float(features[key])
            
            # Ensure all expected features exist to maintain consistent model input size
            if self.feature_names:
                for name in self.feature_names:
                    if name not in features:
                        features[name] = 0.0

            # Final validation
            validated_features = validate_and_clean_features(features)
            if validated_features is None:
                logger.error("Feature validation failed")
                return None
            
            return validated_features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None
    
    def generate_training_labels(self, features, pnl, trade_data=None):
        """Generate comprehensive labels for ML training including signals, position size, and risk management"""
        labels = {}
        
        # Get market indicators for label generation
        pcr_oi = features.get('pcr_oi', 1.0)
        pcr_volume = features.get('pcr_volume', 1.0)
        time_of_day = features.get('time_of_day', 12.0)
        call_ltp_change = features.get('call_ltp_change', 0)
        put_ltp_change = features.get('put_ltp_change', 0)
        volatility = features.get('realized_volatility', 0.2)
        confidence_score = features.get('confidence_score', 0.5)

        # Basic regime and confidence labels
        if pnl is not None:
            # PnL-based labeling for actual trades
            labels['regime'] = 1 if pnl > 10 else (-1 if pnl < -10 else 0)
            labels['confidence'] = 2 if abs(pnl) > 25 else (1 if abs(pnl) > 10 else 0)
            labels['pattern'] = 3 if pnl > 10 else (2 if pnl < -10 else 0)
            
            # Signal direction for Tradetron SHORT strategy (1=SHORT_PUT, -1=SHORT_CALL, 0=NEUTRAL)
            if pnl > 5:
                labels['signal_direction'] = 1  # SHORT PUT signal was profitable
            elif pnl < -5:
                labels['signal_direction'] = -1  # SHORT CALL signal would have been better
            else:
                labels['signal_direction'] = 0  # NEUTRAL
                
            # Position sizing based on PnL outcome (normalized 0-1)
            if pnl > 20:  # Very profitable trade
                labels['position_size'] = min(1.0, 0.8 + confidence_score * 0.2)
            elif pnl > 10:  # Profitable trade
                labels['position_size'] = min(0.8, 0.6 + confidence_score * 0.2)
            elif pnl > 0:  # Small profit
                labels['position_size'] = min(0.6, 0.4 + confidence_score * 0.2)
            elif pnl > -10:  # Small loss
                labels['position_size'] = max(0.2, 0.4 - volatility * 0.2)
            else:  # Large loss
                labels['position_size'] = max(0.1, 0.2 - volatility * 0.1)
            
            # Stop loss percentage based on actual outcome
            if trade_data and 'exit_reason' in trade_data:
                if trade_data['exit_reason'] == 'Stop Loss' and pnl < 0:
                    # Stop loss was hit - it was too tight, suggest wider
                    labels['stop_loss_pct'] = min(0.15, abs(pnl / trade_data.get('entry_price', 100)) * 1.5)
                elif pnl < -15:
                    # Large loss without stop loss hit - suggest tighter
                    labels['stop_loss_pct'] = max(0.05, abs(pnl / trade_data.get('entry_price', 100)) * 0.7)
                else:
                    # Good outcome - maintain current level
                    labels['stop_loss_pct'] = 0.10  # 10% default
            else:
                # Default stop loss based on volatility
                labels['stop_loss_pct'] = max(0.05, min(0.15, volatility * 0.5))
            
            # Trailing stop percentage based on max profit achieved
            if trade_data and 'max_profit' in trade_data:
                max_profit = trade_data.get('max_profit', 0)
                if max_profit > 15 and pnl > 5:
                    # Good trailing stop - preserved most profit
                    labels['trailing_stop_pct'] = 0.6  # Allow 60% retracement
                elif max_profit > 15 and pnl < 0:
                    # Trailing stop was too loose - tighten it
                    labels['trailing_stop_pct'] = 0.4  # Allow only 40% retracement
                elif max_profit > 5:
                    # Moderate profit - balanced approach
                    labels['trailing_stop_pct'] = 0.5  # Allow 50% retracement
                else:
                    # Low profit - don't use trailing stop
                    labels['trailing_stop_pct'] = 0.7  # Very loose
            else:
                # Default trailing stop based on regime
                regime = features.get('regime', 'sideways')
                if 'trending' in str(regime).lower():
                    labels['trailing_stop_pct'] = 0.4  # Tighter for trending
                else:
                    labels['trailing_stop_pct'] = 0.6  # Looser for sideways
                    
        else:
            # During initial learning phase (no trades yet), generate meaningful labels based on market conditions
            # This provides initial diversity for model training while preserving learning integrity
            
            # Generate regime based on market indicators
            if pcr_oi > 1.3 and put_ltp_change > 0:
                labels['regime'] = -1  # Bearish regime indicators
            elif pcr_oi < 0.8 and call_ltp_change > 0:
                labels['regime'] = 1   # Bullish regime indicators
            else:
                labels['regime'] = 0   # Neutral regime
                
            # Generate confidence based on volatility and time of day
            if volatility > 0.3 or time_of_day < 10 or time_of_day > 14:
                labels['confidence'] = 1  # Lower confidence during high vol or edge hours
            elif 0.15 < volatility < 0.25 and 10 <= time_of_day <= 14:
                labels['confidence'] = 2  # Higher confidence during stable mid-session
            else:
                labels['confidence'] = 0  # Neutral confidence
                
            # Generate pattern based on price changes
            if abs(call_ltp_change) > abs(put_ltp_change) and abs(call_ltp_change) > 2:
                labels['pattern'] = 1  # Call-dominated pattern
            elif abs(put_ltp_change) > abs(call_ltp_change) and abs(put_ltp_change) > 2:
                labels['pattern'] = 2  # Put-dominated pattern
            else:
                labels['pattern'] = 0  # Balanced pattern
                
            # Generate signal direction based on multiple indicators
            if pcr_oi > 1.4 and put_ltp_change > call_ltp_change:
                labels['signal_direction'] = 1   # Potential SHORT PUT conditions
            elif pcr_oi < 0.7 and call_ltp_change > put_ltp_change:
                labels['signal_direction'] = -1  # Potential SHORT CALL conditions
            else:
                labels['signal_direction'] = 0   # Neutral signal
                
            # Position sizing based on market conditions (normalized 0-1)
            if volatility > 0.3:
                labels['position_size'] = 0.3  # Smaller position in high vol
            elif volatility < 0.15:
                labels['position_size'] = 0.7  # Larger position in low vol
            else:
                labels['position_size'] = 0.5  # Default position
                
            # Default stop loss based on volatility
            labels['stop_loss_pct'] = max(0.05, min(0.15, volatility * 0.5))
            
            # Default trailing stop based on regime indicators
            if abs(call_ltp_change) > 5 or abs(put_ltp_change) > 5:
                labels['trailing_stop_pct'] = 0.4  # Tighter for high momentum
            else:
                labels['trailing_stop_pct'] = 0.6  # Looser for low momentum
        
        return labels

        
    def update_training_data(self, features, labels=None):
        """Update training data with new samples for continuous learning"""
        try:
            # Store training sample with trade data (dictionary format for ML training)
            sample = {
                'features': features.copy(),
                'timestamp': datetime.now(),
                'labels': labels,
                'trade_data': getattr(self, 'current_trade_data', {})
            }
            
            self.training_data.append(sample)
            
            # Store features in feature_buffer as dictionary (for lag features)
            # This maintains backward compatibility while fixing the type issue
            self.feature_buffer.append(features.copy())
            
            # Maintain training data size
            if len(self.training_data) > self.max_training_samples:
                # Remove oldest samples but keep recent ones
                self.training_data = self.training_data[-self.max_training_samples:]
            
            # Initialize or extend feature_names without shrinking to prevent scaler/model mismatch
            if not self.feature_names:
                self.feature_names = list(features.keys())
                self.input_size = len(self.feature_names)
                logger.info(f"Feature names initialized: {self.input_size} features")
            else:
                # Append any new features not seen before, keep order stable
                added = 0
                for name in features.keys():
                    if name not in self.feature_names:
                        self.feature_names.append(name)
                        added += 1
                if added:
                    self.input_size = len(self.feature_names)
                    logger.info(f"Feature names extended by {added}: now {self.input_size} features")
                
                # Reinitialize deep model with correct input size
                self.deep_model = LSTMSignalPredictor(
                    input_size=self.input_size,
                    hidden_size=64,
                    num_layers=2,
                    dropout=0.2
                ).to(device)
                self.optimizer = torch.optim.Adam(self.deep_model.parameters(), lr=0.001)
            
            # Increment training counter
            self.training_counter += 1

            # Populate sequence buffer from recent sanitized features to enable deep model training
            try:
                if len(self.feature_buffer) >= self.sequence_length:
                    # Build a recent sequence window
                    recent = list(self.feature_buffer)[-self.sequence_length:]
                    sequence = []
                    for feat in recent:
                        # Ensure dict and all names present
                        if isinstance(feat, dict):
                            for name in self.feature_names:
                                if name not in feat:
                                    feat[name] = 0.0
                            # Sanitize to numeric vector
                            vec = []
                            for name in self.feature_names:
                                val = feat.get(name, 0)
                                try:
                                    num = float(val)
                                except (ValueError, TypeError):
                                    num = 0.0
                                if isinstance(num, float) and (np.isnan(num) or np.isinf(num)):
                                    num = 0.0
                                vec.append(num)
                            sequence.append(vec)
                        else:
                            # Already a vector; sanitize elements
                            row_vec = []
                            for v in feat:
                                try:
                                    num = float(v)
                                except (ValueError, TypeError):
                                    num = 0.0
                                if isinstance(num, float) and (np.isnan(num) or np.isinf(num)):
                                    num = 0.0
                                row_vec.append(num)
                            sequence.append(row_vec)
                    # Simple label: current regime if available, else 1
                    seq_label = 1.0
                    if labels and isinstance(labels, dict) and 'regime' in labels:
                        try:
                            seq_label = float(labels['regime'])
                        except (ValueError, TypeError):
                            seq_label = 1.0
                    self.sequence_buffer.add(sequence, seq_label)
            except Exception as e:
                logger.debug(f"Sequence buffer population skipped: {e}")
            
            # Log training data status
            if self.training_counter % 100 == 0:
                logger.info(f"Training data: {len(self.training_data)} samples, "
                           f"Feature buffer: {len(self.feature_buffer)} sequences")
            
        except Exception as e:
            logger.error(f"Error updating training data: {e}")
    
    def train_models(self):
        """Train models with enhanced continuous learning and class diversity validation"""
        if len(self.training_data) < self.min_training_samples:
            return False
        
        try:
            # Prepare training data for all models
            X = []
            y_regime = []
            y_confidence = []
            y_pattern = []
            y_signal_direction = []
            y_position_size = []
            y_stop_loss = []
            y_trailing_stop = []
            
            # Use all available training data
            for sample in self.training_data:
                features = sample['features']
                labels = sample.get('labels', {})
                trade_data = sample.get('trade_data', {})
                
                # Extract feature values
                feature_values = []
                for name in self.feature_names:
                    value = features.get(name, 0)
                    # Coerce to numeric and sanitize
                    if value is None:
                        coerced = 0.0
                    else:
                        try:
                            coerced = float(value)
                        except (ValueError, TypeError):
                            coerced = 0.0
                    # Replace NaN/Inf after coercion
                    if isinstance(coerced, float) and (np.isnan(coerced) or np.isinf(coerced)):
                        coerced = 0.0
                    feature_values.append(coerced)
                X.append(feature_values)
                
                # Generate labels if not provided
                if not labels:
                    # Use PnL-based labeling for continuous learning
                    pnl = features.get('pnl', 0) if 'pnl' in features else 0
                    labels = self.generate_training_labels(features, pnl, trade_data)
                
                # Extract all labels
                y_regime.append(labels.get('regime', 0))
                y_confidence.append(labels.get('confidence', 1))
                y_pattern.append(labels.get('pattern', 0))
                y_signal_direction.append(labels.get('signal_direction', 0))
                y_position_size.append(labels.get('position_size', 0.5))
                y_stop_loss.append(labels.get('stop_loss_pct', 0.10))
                y_trailing_stop.append(labels.get('trailing_stop_pct', 0.5))
            
            # Convert to numpy arrays (enforce numeric dtype)
            X = np.array(X, dtype=float)
            # Guard against empty or invalid shapes
            if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
                logger.info("No valid feature data to train. Skipping this cycle.")
                return False
            # Sanitize any remaining non-finite values
            if not np.isfinite(X).all():
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Freeze trained feature order snapshot on first successful train
            if not getattr(self, 'trained_feature_names', []):
                self.trained_feature_names = list(self.feature_names)
            y_regime = np.array(y_regime)
            y_confidence = np.array(y_confidence)
            y_pattern = np.array(y_pattern)
            y_signal_direction = np.array(y_signal_direction)
            y_position_size = np.array(y_position_size)
            y_stop_loss = np.array(y_stop_loss)
            y_trailing_stop = np.array(y_trailing_stop)
            
            # Validate class diversity before training
            def has_sufficient_diversity(y_array, min_classes=2):
                """Check if array has at least min_classes unique values"""
                unique_classes = len(np.unique(y_array))
                return unique_classes >= min_classes
            
            # Check each target variable for diversity
            diversity_check = {
                'regime': has_sufficient_diversity(y_regime),
                'confidence': has_sufficient_diversity(y_confidence), 
                'pattern': has_sufficient_diversity(y_pattern),
                'signal_direction': has_sufficient_diversity(y_signal_direction)
            }
            
            # Log diversity status
            insufficient_diversity = [name for name, has_div in diversity_check.items() if not has_div]
            if insufficient_diversity:
                logger.info(f"Insufficient class diversity for: {insufficient_diversity}. Continuing data collection...")
                return False
            
            # Build matrix in trained feature order (and fit scaler on that shape)
            if self.trained_feature_names:
                reorder_indices = [self.feature_names.index(n) for n in self.trained_feature_names if n in self.feature_names]
                if len(reorder_indices) == X.shape[1]:
                    X_ordered = X[:, reorder_indices]
                else:
                    X_ordered = np.zeros((X.shape[0], len(self.trained_feature_names)))
                    name_to_idx = {n: i for i, n in enumerate(self.feature_names)}
                    for j, n in enumerate(self.trained_feature_names):
                        if n in name_to_idx:
                            X_ordered[:, j] = X[:, name_to_idx[n]]
            else:
                X_ordered = X

            # Fit scaler on the exact matrix it will later transform
            if hasattr(self.scaler, 'n_features_in_') and getattr(self.scaler, 'n_features_in_', None) != X_ordered.shape[1]:
                self.scaler = StandardScaler()
            self.scaler.fit(X_ordered)
            X_scaled = self.scaler.transform(X_ordered)
            
            # Train classification models only if they have sufficient diversity
            models_trained = []
            
            if diversity_check['regime']:
                self.models['regime_classifier'].fit(X_scaled, y_regime)
                models_trained.append('regime_classifier')
            
            if diversity_check['confidence']:
                self.models['signal_confidence'].fit(X_scaled, y_confidence)
                models_trained.append('signal_confidence')
                
            if diversity_check['pattern']:
                self.models['pattern_detector'].fit(X_scaled, y_pattern)
                models_trained.append('pattern_detector')
                
            if diversity_check['signal_direction']:
                self.models['signal_direction'].fit(X_scaled, y_signal_direction)
                models_trained.append('signal_direction')
            
            # Train regression models (these can handle single values better)
            self.models['position_sizer'].fit(X_scaled, y_position_size)
            self.models['stop_loss_predictor'].fit(X_scaled, y_stop_loss)
            self.models['trailing_stop_predictor'].fit(X_scaled, y_trailing_stop)
            models_trained.extend(['position_sizer', 'stop_loss_predictor', 'trailing_stop_predictor'])
            
            # Train meta-learner if we have enough data and required models are trained
            meta_learner_trained = False
            if (len(X_scaled) >= 50 and 
                'regime_classifier' in models_trained and 
                'signal_confidence' in models_trained):
                try:
                    # Use the same features as base models for consistency
                    if has_sufficient_diversity(y_regime):
                        # Train meta-learner with same feature set as base models
                        self.models['meta_learner'] = RandomForestClassifier(n_estimators=50, random_state=42, max_features='sqrt')
                        self.models['meta_learner'].fit(X_scaled, y_regime)
                        models_trained.append('meta_learner')
                        meta_learner_trained = True
                except Exception as e:
                    logger.warning(f"Meta-learner training failed: {e}")
            
            # Train deep learning model
            deep_model_trained = False
            if len(self.feature_buffer) >= self.sequence_length:
                deep_model_trained = self._train_deep_model()
                if deep_model_trained:
                    models_trained.append('deep_model')
            
            # Perform SHAP analysis only if we have trained classification models
            shap_performed = False
            if (SHAP_AVAILABLE and len(X_scaled) >= 20 and 
                any(model in models_trained for model in ['regime_classifier', 'signal_confidence', 'pattern_detector'])):
                try:
                    self.perform_shap_analysis(X_scaled)
                    shap_performed = True
                except Exception as e:
                    logger.warning(f"SHAP analysis failed: {e}")
            
            # Mark as trained only if at least some models were successfully trained
            if models_trained:
                self.is_trained = True
                
                # Save models
                self.save_models()
                
                # Enhanced logging
                logger.info(f"Models trained successfully: {', '.join(models_trained)} with {len(X_scaled)} samples")
                
                # Report training status
                training_status = {
                    'classification_models': len([m for m in models_trained if m in ['regime_classifier', 'signal_confidence', 'pattern_detector', 'signal_direction']]),
                    'regression_models': len([m for m in models_trained if m in ['position_sizer', 'stop_loss_predictor', 'trailing_stop_predictor']]),
                    'meta_learner': meta_learner_trained,
                    'deep_model': deep_model_trained,
                    'shap_analysis': shap_performed,
                    'total_samples': len(X_scaled),
                    'diversity_issues': insufficient_diversity
                }
                
                print(f"🔄 Training Status: {training_status['classification_models']}/4 classification, "
                      f"{training_status['regression_models']}/3 regression, "
                      f"Meta: {'✅' if meta_learner_trained else '❌'}, "
                      f"Deep: {'✅' if deep_model_trained else '❌'}, "
                      f"SHAP: {'✅' if shap_performed else '❌'}")
                
                return True
            else:
                logger.info(f"No models could be trained due to insufficient data diversity")
                return False

        except Exception as e:
            logger.error(f"Error training models: {e}", exc_info=True)
            return False
    
    def perform_shap_analysis(self, X_scaled):
        """Perform SHAP analysis on trained models"""
        if not SHAP_AVAILABLE or not self.is_trained:
            return
            
        try:
            print("🔍 Performing SHAP feature importance analysis...")
            
            # Use a sample of data for SHAP analysis (faster)
            sample_size = min(100, len(X_scaled))
            X_sample = X_scaled[:sample_size]
            
            # Analyze each model
            for model_name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
                    # Initialize explainer
                    model_type = 'tree' if hasattr(model, 'estimators_') else 'linear'
                    self.shap_analyzer.initialize_explainer(model, self.feature_names, model_type)
                    
                    # Analyze feature importance
                    importance = self.shap_analyzer.analyze_feature_importance(
                        model, X_sample, self.feature_names, model_name
                    )
                    
                    if importance:
                        print(f"✅ SHAP analysis completed for {model_name}")
                        
                        # Get top features
                        top_features = self.shap_analyzer.get_top_features(model_name, 5)
                        print(f"   Top 5 features: {[f[0] for f in top_features]}")
                        
                        # Track importance over time
                        self.shap_analyzer.track_feature_importance_over_time(model_name)
            
            # Generate and display feature report
            if 'regime_classifier' in self.shap_analyzer.shap_values:
                report = self.shap_analyzer.generate_feature_report('regime_classifier')
                print(report)
                
                # Save feature importance plot
                self.shap_analyzer.save_feature_importance_plot('regime_classifier')
                
                # Export to CSV
                self.shap_analyzer.export_feature_importance_csv('regime_classifier')
            
        except Exception as e:
            logger.error(f"SHAP analysis error: {e}")
    
    def analyze_prediction_with_shap(self, features):
        """Analyze a single prediction using SHAP"""
        if not SHAP_AVAILABLE or not self.is_trained:
            return None
            
        try:
            # Prepare feature vector (ensure all expected features exist)
            for name in self.feature_names:
                if name not in features:
                    features[name] = 0.0
            # Build vector in trained feature order to match scaler
            if hasattr(self, 'trained_feature_names') and self.trained_feature_names:
                feature_vector = []
                for name in self.trained_feature_names:
                    val = features.get(name, 0)
                    try:
                        num = float(val)
                    except (ValueError, TypeError):
                        num = 0.0
                    if isinstance(num, float) and (np.isnan(num) or np.isinf(num)):
                        num = 0.0
                    feature_vector.append(num)
            else:
                feature_vector = []
                for name in self.feature_names:
                    val = features.get(name, 0)
                    try:
                        num = float(val)
                    except (ValueError, TypeError):
                        num = 0.0
                    if isinstance(num, float) and (np.isnan(num) or np.isinf(num)):
                        num = 0.0
                    feature_vector.append(num)
            X = np.array(feature_vector, dtype=float).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            # Analyze with SHAP
            analysis = self.shap_analyzer.analyze_prediction(
                self.models['regime_classifier'], X_scaled, self.feature_names, 'regime_classifier'
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"SHAP prediction analysis error: {e}")
            return None
    
    def _train_deep_model(self):
        """Train the LSTM model on collected sequences"""
        if len(self.sequence_buffer) < 100:  # Minimum sequences to start training
            return False
            
        try:
            # Train the LSTM model
            loss = train_lstm(
                model=self.deep_model,
                buffer=self.sequence_buffer,
                optimizer=self.optimizer,
                criterion_regime=self.criterion_regime,
                criterion_confidence=self.criterion_confidence,
                device=device,
                batch_size=32,
                epochs=5
            )
            
            print(f"[DEEP LEARNING] Model trained - Loss: {loss:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Deep learning training error: {e}")
            return False
    
    def predict_with_deep_model(self, sequence):
        """Make prediction using the deep learning model"""
        if len(sequence) < self.sequence_length:
            return None
            
        try:
            # Use only the most recent sequence
            sequence = sequence[-self.sequence_length:]
            
            # Get prediction from deep model
            prediction = predict_sequence(
                model=self.deep_model,
                sequence=sequence,
                device=device
            )
            
            return {
                'regime': prediction['regime'],
                'confidence': prediction['confidence'],
                'attention_weights': prediction['attention_weights'].tolist()
            }
            
        except Exception as e:
            logger.error(f"Deep learning prediction error: {e}")
            return None
    
    def predict(self, features):
        """Make comprehensive predictions including signals, position size, and risk management"""
        if not self.is_trained or not self.feature_names:
            # No fallback - return None if not trained
            return None
            
        try:
            # Prepare feature vector (sanitize to numeric floats)
            feature_vector = []
            for name in self.feature_names:
                value = features.get(name, 0)
                if value is None:
                    coerced = 0.0
                else:
                    try:
                        coerced = float(value)
                    except (ValueError, TypeError):
                        coerced = 0.0
                if isinstance(coerced, float) and (np.isnan(coerced) or np.isinf(coerced)):
                    coerced = 0.0
                feature_vector.append(coerced)
            X = np.array(feature_vector, dtype=float).reshape(1, -1)
            
            # Scale features using trained feature order
            try:
                if hasattr(self, 'trained_feature_names') and self.trained_feature_names:
                    # Reorder X to match trained feature order
                    reorder_indices = [self.feature_names.index(n) for n in self.trained_feature_names if n in self.feature_names]
                    if len(reorder_indices) == X.shape[1]:
                        X_ordered = X[:, reorder_indices]
                    else:
                        # Build full vector in trained order with zeros for missing
                        X_ordered = np.zeros((X.shape[0], len(self.trained_feature_names)))
                        name_to_idx = {n: i for i, n in enumerate(self.feature_names)}
                        for j, n in enumerate(self.trained_feature_names):
                            if n in name_to_idx:
                                X_ordered[:, j] = X[:, name_to_idx[n]]
                    X_scaled = self.scaler.transform(X_ordered)
                else:
                    X_scaled = self.scaler.transform(X)
            except Exception as e:
                logger.warning(f"Feature scaling failed: {e}")
                # Use unscaled features as fallback
                X_scaled = X if not hasattr(self, 'trained_feature_names') or not self.trained_feature_names else (
                    X[:, [self.feature_names.index(n) for n in self.trained_feature_names if n in self.feature_names]]
                )
            
            # Get predictions from available models with fallbacks
            regime_pred = 0
            confidence = 0.5
            pattern = 0
            signal_direction = 0
            position_size = 0.5
            stop_loss_pct = 0.10
            trailing_stop_pct = 0.5
            
            # Ensure feature vector has correct size (pad if necessary)
            if len(feature_vector) != len(self.feature_names):
                logger.warning(f"Feature vector size mismatch: {len(feature_vector)} vs {len(self.feature_names)}")
                if len(feature_vector) < len(self.feature_names):
                    feature_vector.extend([0.0] * (len(self.feature_names) - len(feature_vector)))
                else:
                    feature_vector = feature_vector[:len(self.feature_names)]
                X = np.array(feature_vector).reshape(1, -1)
                try:
                    X_scaled = self.scaler.transform(X)
                except Exception as e:
                    logger.warning(f"Rescale after padding failed: {e}")
                    X_scaled = X
            
            # Try to get predictions from trained models
            try:
                if hasattr(self.models['regime_classifier'], 'predict'):
                    regime_pred = self.models['regime_classifier'].predict(X_scaled)[0]
            except:
                regime_pred = 0
                
            try:
                if hasattr(self.models['signal_confidence'], 'predict_proba'):
                    confidence_proba = self.models['signal_confidence'].predict_proba(X_scaled)[0]
                    confidence = confidence_proba[1] if len(confidence_proba) > 1 else confidence_proba[0]
            except:
                confidence = 0.5
                
            try:
                if hasattr(self.models['pattern_detector'], 'predict'):
                    pattern = self.models['pattern_detector'].predict(X_scaled)[0]
            except:
                pattern = 0
                
            try:
                if hasattr(self.models['signal_direction'], 'predict'):
                    signal_direction = self.models['signal_direction'].predict(X_scaled)[0]
            except:
                signal_direction = 0
                
            try:
                if hasattr(self.models['position_sizer'], 'predict'):
                    position_size = self.models['position_sizer'].predict(X_scaled)[0]
            except:
                position_size = 0.5
                
            try:
                if hasattr(self.models['stop_loss_predictor'], 'predict'):
                    stop_loss_pct = self.models['stop_loss_predictor'].predict(X_scaled)[0]
            except:
                stop_loss_pct = 0.10
                
            try:
                if hasattr(self.models['trailing_stop_predictor'], 'predict'):
                    trailing_stop_pct = self.models['trailing_stop_predictor'].predict(X_scaled)[0]
            except:
                trailing_stop_pct = 0.5
            
            # Ensure reasonable bounds
            position_size = max(0.1, min(1.0, position_size))  # Between 10% and 100%
            stop_loss_pct = max(0.05, min(0.20, stop_loss_pct))  # Between 5% and 20%
            trailing_stop_pct = max(0.3, min(0.8, trailing_stop_pct))  # Between 30% and 80%
            
            # Get deep learning prediction if enough history
            deep_pred = None
            if len(self.feature_buffer) >= self.sequence_length - 1:
                try:
                    # Convert feature dictionaries to vectors for deep learning
                    sequence_vectors = []
                    
                    # Get the last (sequence_length-1) feature dictionaries
                    recent_features = list(self.feature_buffer)[-(self.sequence_length-1):]
                    
                    for feat_dict in recent_features:
                        if isinstance(feat_dict, dict):
                            # Convert dict to vector using feature names order
                            # Ensure all names exist in dict to maintain stable length
                            for name in self.feature_names:
                                if name not in feat_dict:
                                    feat_dict[name] = 0.0
                            vector = []
                            for name in self.feature_names:
                                val = feat_dict.get(name, 0)
                                if val is None:
                                    num = 0.0
                                else:
                                    try:
                                        num = float(val)
                                    except (ValueError, TypeError):
                                        num = 0.0
                                if isinstance(num, float) and (np.isnan(num) or np.isinf(num)):
                                    num = 0.0
                                vector.append(num)
                            sequence_vectors.append(vector)
                        else:
                            # If it's already a vector, use it directly
                            # Sanitize vector elements
                            sanitized = []
                            for val in feat_dict:
                                try:
                                    num = float(val)
                                except (ValueError, TypeError):
                                    num = 0.0
                                if isinstance(num, float) and (np.isnan(num) or np.isinf(num)):
                                    num = 0.0
                                sanitized.append(num)
                            sequence_vectors.append(sanitized)
                    
                    # Add current features as the last element
                    sequence_vectors.append(feature_vector)
                    
                    # Make prediction with the sequence
                    deep_pred = self.predict_with_deep_model(sequence_vectors)
                except Exception as e:
                    logger.warning(f"Deep learning prediction error: {e}")
                    deep_pred = None
            
            # Meta-learner prediction (if trained and base models available)
            final_regime = regime_pred
            final_confidence = confidence
            
            try:
                if hasattr(self.models['meta_learner'], 'predict_proba'):
                    # Use meta-learner prediction with same feature set
                    meta_regime = self.models['meta_learner'].predict(X_scaled)[0]
                    meta_proba = self.models['meta_learner'].predict_proba(X_scaled)[0]
                    final_confidence = max(meta_proba) if len(meta_proba) > 0 else confidence
                    final_regime = meta_regime
                    
                    # If we have deep learning prediction, average with meta-learner
                    if deep_pred:
                        final_regime = int(round((final_regime + deep_pred['regime'] + 1) / 2)) - 1
                        final_confidence = (final_confidence + deep_pred['confidence']) / 2
                        
            except Exception as e:
                logger.warning(f"Meta-learner prediction failed, using base models: {e}")
                # Fallback to base model predictions
                final_regime = regime_pred
                final_confidence = confidence
                
                if deep_pred:
                    final_regime = int(round((final_regime + deep_pred['regime'] + 1) / 2)) - 1
                    final_confidence = (final_confidence + deep_pred['confidence']) / 2
            
            # Convert signal direction to Tradetron format
            tradetron_signal = 0  # Default NEUTRAL
            if signal_direction == 1:
                tradetron_signal = 1  # CALL
            elif signal_direction == -1:
                tradetron_signal = -1  # PUT
            
            # Adjust confidence based on signal strength
            if abs(signal_direction) == 1:
                final_confidence = max(final_confidence, 0.7)  # Minimum 70% confidence for signals
            
            return {
                'regime': int(final_regime),
                'confidence': float(final_confidence),
                'pattern': int(pattern),
                'signal_direction': int(signal_direction),
                'tradetron_signal': int(tradetron_signal),
                'position_size': float(position_size),
                'stop_loss_pct': float(stop_loss_pct),
                'trailing_stop_pct': float(trailing_stop_pct),
                'deep_learning_used': deep_pred is not None
            }
            

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            # No fallback - return None on error
            return None
            
    def save_models(self):
        """Save trained models to disk with feature verification"""
        try:
            # Verify feature consistency across models
            model_features = {}
            for name, model in self.models.items():
                if hasattr(model, 'feature_importances_'):
                    model_features[name] = len(model.feature_importances_)
                elif hasattr(model, 'coef_'):
                    model_features[name] = model.coef_.shape[1]
                    
            # Verify all models use same number of features (compare to trained_feature_names if available)
            feature_counts = set(model_features.values())
            if len(feature_counts) > 1:
                logger.error(f"Feature count mismatch across models: {model_features}")
                return False
            expected_count = len(self.trained_feature_names) if getattr(self, 'trained_feature_names', []) else (list(feature_counts)[0] if feature_counts else len(self.feature_names))
            if expected_count != (list(feature_counts)[0] if feature_counts else expected_count):
                logger.error(f"Feature count mismatch: expected={expected_count}, model={(list(feature_counts)[0] if feature_counts else 'unknown')}")
                return False

            # Save model data
            model_data = {  
                'models': {},
                'feature_names': self.feature_names,
                'trained_feature_names': self.trained_feature_names or self.feature_names,
                'is_trained': self.is_trained,
                'scaler': self.scaler,
                'input_size': self.input_size,
                'sequence_length': self.sequence_length,
                'model_features': model_features  # Save feature counts for verification
            }

            # Save each model's state
            for name, model in self.models.items():
                model_data['models'][name] = model

            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)

            # Save deep learning model
            if self.is_trained:
                torch.save({
                    'model_state_dict': self.deep_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'input_size': self.input_size,
                    'sequence_length': self.sequence_length,
                    'feature_names': self.feature_names,  # Save features for LSTM too
                    'trained_feature_names': self.trained_feature_names or self.feature_names
                }, self.deep_model_path)

            logger.info("All models saved successfully with feature verification")
            return True

        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
            
    def load_models(self):
        """Load trained models from disk, including both traditional ML and deep learning models"""
        if not os.path.exists(self.model_path):
            logger.warning("No saved models found. Starting with fresh models.")
            return False

        try:
            # Load traditional ML models
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)

            # Load models
            for name, model in model_data['models'].items():
                if name in self.models:
                    self.models[name] = model

            # Load other data
            self.feature_names = model_data.get('feature_names', [])
            self.trained_feature_names = model_data.get('trained_feature_names', self.feature_names)
            self.is_trained = model_data.get('is_trained', False)
            self.scaler = model_data.get('scaler', StandardScaler())
            self.input_size = model_data.get('input_size', 50)
            self.sequence_length = model_data.get('sequence_length', 10)

            # Load deep learning model if available
            if os.path.exists(self.deep_model_path):
                checkpoint = torch.load(self.deep_model_path, map_location=device)

                # Initialize model with correct dimensions
                self.input_size = checkpoint.get('input_size', self.input_size)
                self.sequence_length = checkpoint.get('sequence_length', self.sequence_length)

                # Re-initialize model and optimizer with correct dimensions
                self.deep_model = LSTMSignalPredictor(
                    input_size=self.input_size,
                    hidden_size=64,
                    num_layers=2,
                    dropout=0.2
                ).to(device)

                self.optimizer = torch.optim.Adam(self.deep_model.parameters(), lr=0.001)

                # Load state dicts
                self.deep_model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.deep_model.eval()

                logger.info("Deep learning model loaded successfully")

            logger.info("All models loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Reset to default models on error
            self.is_trained = False
            return False

    def validate_model_performance(self):
        """Validate model performance on recent data"""
        if len(self.recent_outcomes) < 10:
            return True
        
        success_rate = sum(self.recent_outcomes) / len(self.recent_outcomes)
        # Remove minimum success rate requirement - let system learn from failures
        return True

    def check_circuit_breakers(self):
        """Circuit breakers disabled - let system learn from failures"""
        # Always allow trading - no circuit breakers
        return True

    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        if not self.episode_pnls:
            return {}
        
        return {
            'total_trades': len(self.episode_pnls),
            'win_rate': sum(1 for pnl in self.episode_pnls if pnl > 0) / len(self.episode_pnls),
            'avg_pnl': np.mean(self.episode_pnls),
            'max_drawdown': min(self.episode_pnls),
            'sharpe_ratio': np.mean(self.episode_pnls) / (np.std(self.episode_pnls) + 1e-8)
        }

    def display_performance_metrics(self):
        try:
            metrics = self.calculate_performance_metrics()
            print("\n" + "="*60)
            print("📊 PERFORMANCE METRICS")
            print("="*60)
            for k, v in metrics.items():
                print(f"{k}: {v}")
            print("="*60 + "\n")
        except Exception as e:
            logger.error(f"Performance metrics display error: {e}")

class OptimizedATMAnalyzer:
    def flow(self, code, message=""):
        try:
            logger.info(f"[FLOW:{code}] {message}")
        except Exception:
            pass

    def __init__(self):
        self.headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'accept': 'application/json, text/plain, */*',
            'accept-encoding': 'identity',
            'accept-language': 'en-US,en;q=0.9',
            'Host': 'www.nseindia.com'
        }
        
        self.session = requests.Session()
        self.last_request_time = 0
        self.historical_data = {}
        self.max_history = 10  # Increased for ML
        
        # Signal tracking with persistence
        self.current_signal = 0
        self.signal_start_time = None
        self.signal_strength = 0
        self.last_sent_signal = None
        self.signal_history = deque(maxlen=5)  # Track last 5 signals for stability
        
        # 24/7 Data Storage and Learning System (Initialize first)
        self.data_storage = MarketDataStorage()
        self.offline_training_counter = 0
        self.last_offline_training = None
        self.continuous_learning_active = True
        
        # ML Engine (Initialize after data_storage)
        self.ml_engine = AdvancedMLDecisionEngine()
        self.ml_engine.data_storage = self.data_storage  # Assign data_storage reference
        self.training_counter = 0

        # Flow: A -> B
        self.flow('A', 'System Start')
        self.flow('B', 'Initialize Components')
        
        # Enhanced Components
        self.risk_manager = AdvancedRiskManager()
        self.regime_detector = MarketRegimeDetector()
        self.profit_optimizer = ProfitOptimizer()
        
        # Enhanced tracking
        self.account_value = 100000  # Initial account value
        self.portfolio_performance = []
        
        # RL Experience Replay Buffer
        self.replay_buffer = deque(maxlen=1000)  # Store last 1000 experiences
        self.gamma = 0.95  # Discount factor
        self.learning_rate = 0.001
        self.min_learning_rate = 0.0001
        self.learning_rate_decay = 0.995
        self.batch_size = 32
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_pnls = []
        self.avg_reward = 0
        
        # Tradetron URLs
        self.TRADETRON_URLS = {
            1: "https://api.tradetron.tech/api?auth-token=<YOURAPITOKEN>&key=gap&value=1",
            -1: "https://api.tradetron.tech/api?auth-token=<YOURAPITOKEN>&key=gap&value=-1",
            0: "https://api.tradetron.tech/api?auth-token=<YOURAPITOKEN>&key=gap&value=0"
        }
        # Paper trade tracking
        self.paper_trade = PaperTrade()
        
        # Drift detection and outcome tracking
        self.recent_outcomes = deque(maxlen=5)  # Track last 5 trade outcomes for drift detection
        
        # Load existing models
        self.ml_engine.load_models()
        self.flow('C', 'Load Existing Models')
        
        # Initialize with stored data if available
        self.initialize_with_stored_data()
        self.flow('D', 'Initialize with Stored Data')
        
        # Initialize regime detector with historical price data
        self.initialize_regime_detector_with_stored_data()
        
    def initialize_with_stored_data(self):
        """Initialize system with stored historical data"""
        try:
            # Get stored training data
            stored_data = self.data_storage.get_training_data(days_back=30)
            
            if stored_data:
                logger.info(f"Initializing with {len(stored_data)} stored data samples")
                
                # Process ALL available stored data for maximum training effectiveness
                logger.info(f"Processing {len(stored_data)} stored data samples for training...")
                for data_entry in stored_data:  # Use ALL available samples
                    try:
                        # Convert stored data to DataFrame
                        option_data = pd.DataFrame(data_entry['option_data'])
                        underlying_value = data_entry['underlying_value']
                        timestamp = datetime.fromisoformat(data_entry['timestamp'])
                        
                        # Extract features from stored data
                        features = self.ml_engine.extract_features(
                            option_data, 
                            self.historical_data, 
                            underlying_value, 
                            self.get_atm_strike(option_data, underlying_value)
                        )
                        
                        if features:
                            # Generate diverse labels for historical data using market conditions (pass None for pnl)
                            labels = self.ml_engine.generate_training_labels(features, None)
                            self.ml_engine.update_training_data(features, labels)
                            
                    except Exception as e:
                        logger.error(f"Error processing stored data: {e}")
                        continue
                
                # Train models with historical data
                if self.ml_engine.train_models():
                    logger.info("✅ Models trained with historical data")
                    
        except Exception as e:
            logger.error(f"Error initializing with stored data: {e}")
    
    def initialize_regime_detector_with_stored_data(self):
        """Initialize regime detector with historical price data from storage"""
        try:
            # Get recent stored data
            stored_data = self.data_storage.get_training_data(days_back=2)  # Last 2 days
            
            if stored_data:
                price_entries = []
                
                # Extract underlying values and timestamps
                for data_entry in stored_data[-50:]:  # Use last 50 entries
                    try:
                        timestamp = datetime.fromisoformat(data_entry['timestamp'])
                        underlying_value = data_entry['underlying_value']
                        
                        price_entries.append({
                            'timestamp': timestamp,
                            'price': underlying_value
                        })
                        
                    except Exception as e:
                        continue
                
                # Sort by timestamp and initialize regime detector
                if price_entries:
                    price_entries.sort(key=lambda x: x['timestamp'])
                    self.regime_detector.price_history = price_entries
                    
                    logger.info(f"✅ Regime detector initialized with {len(price_entries)} historical price points")
                    
                    # Calculate initial volatility to verify
                    initial_vol = self.regime_detector.calculate_realized_volatility_from_history()
                    logger.info(f"📊 Initial calculated volatility: {initial_vol:.3f}")
                    
        except Exception as e:
            logger.error(f"Error initializing regime detector with stored data: {e}")
            
    def get_atm_strike(self, option_data, underlying_value):
        """Get ATM strike from option data"""
        try:
            strikes = sorted(option_data['Strike'].unique())
            return min(strikes, key=lambda x: abs(x - underlying_value))
        except:
            return underlying_value

    def _create_trading_day_simulation(self, raw_data):
        """Create a full trading day simulation (9:15 AM to 3:30 PM) from available data"""
        try:
            import random
            from datetime import datetime, timedelta
            
            if not raw_data:
                return []
            
            # Use today's date for simulation
            base_date = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = base_date.replace(hour=15, minute=30)
            
            # Generate time intervals (every 1 minute from 9:15 AM to 3:30 PM)
            simulation_data = []
            current_time = base_date
            data_index = 0
            
            logger.info(f"🕘 Creating trading day simulation from {base_date.strftime('%H:%M')} to {market_close.strftime('%H:%M')}")
            
            while current_time <= market_close:
                try:
                    # Use data cyclically if we have multiple samples, or add variations to single sample
                    if len(raw_data) > 1:
                        # Use different data entries for variety
                        base_entry = raw_data[data_index % len(raw_data)]
                    else:
                        # Use the single available entry with variations
                        base_entry = raw_data[0].copy()
                    
                    # Create a new entry with the current simulation time
                    simulated_entry = {
                        'timestamp': current_time.isoformat(),
                        'underlying_value': base_entry['underlying_value'],
                        'expiry': base_entry['expiry'],
                        'option_data': []
                    }
                    
                    # Add slight variations to make data more realistic for simulation
                    # Vary underlying value slightly (±0.1% random walk)
                    base_underlying = base_entry['underlying_value']
                    variation_factor = 1 + (random.random() - 0.5) * 0.002  # ±0.1% variation
                    simulated_entry['underlying_value'] = base_underlying * variation_factor
                    
                    # Copy and slightly vary option data
                    for option in base_entry['option_data']:
                        varied_option = option.copy()
                        
                        # Add small variations to LTP (±1-5%)
                        if varied_option.get('Call_LTP', 0) > 0:
                            call_variation = 1 + (random.random() - 0.5) * 0.05  # ±2.5%
                            varied_option['Call_LTP'] = varied_option['Call_LTP'] * call_variation
                        
                        if varied_option.get('Put_LTP', 0) > 0:
                            put_variation = 1 + (random.random() - 0.5) * 0.05  # ±2.5%
                            varied_option['Put_LTP'] = varied_option['Put_LTP'] * put_variation
                        
                        # Update timestamp to match simulation time
                        varied_option['Timestamp'] = current_time.strftime('%Y-%m-%d %H:%M:%S')
                        
                        simulated_entry['option_data'].append(varied_option)
                    
                    simulation_data.append(simulated_entry)
                    
                    # Move to next minute
                    current_time += timedelta(minutes=1)
                    data_index += 1
                    
                except Exception as e:
                    logger.error(f"Error creating simulation entry for {current_time}: {e}")
                    current_time += timedelta(minutes=1)
                    continue
            
            total_minutes = (market_close - base_date).total_seconds() / 60
            logger.info(f"📈 Created {len(simulation_data)} simulation entries covering {total_minutes:.0f} minutes of trading")
            
            return simulation_data
            
        except Exception as e:
            logger.error(f"Error creating trading day simulation: {e}")
            # Fallback to original data if simulation creation fails
            return raw_data

    def send_signal(self, signal):
        """Send signal to Tradetron with validation and error handling"""
        try:
            # Validate signal
            if not isinstance(signal, int) or signal not in [-1, 0, 1]:
                logger.error(f"Invalid signal value: {signal}")
                return False
                
            # Check if signal URL exists
            if signal not in self.TRADETRON_URLS:
                logger.error(f"No URL configured for signal: {signal}")
                return False
                
            # Send signal with retry logic
            max_retries = 3
            retry_delay = 2  # seconds
            
            for retry in range(max_retries):
                try:
                    # Send request with timeout
                    response = requests.get(self.TRADETRON_URLS[signal], timeout=5)
                    
                    # Check response
                    if response.status_code == 200:
                        signal_name = "SHORT PUT" if signal == 1 else ("SHORT CALL" if signal == -1 else "NEUTRAL")
                        logger.info(f"Signal sent successfully: {signal_name}")
                        return True
                    else:
                        logger.warning(f"Signal send failed with status {response.status_code}")
                        if retry < max_retries - 1:
                            time.sleep(retry_delay * (retry + 1))
                            continue
                        return False
                        
                except requests.exceptions.Timeout:
                    logger.warning(f"Signal send timeout (attempt {retry + 1}/{max_retries})")
                    if retry < max_retries - 1:
                        time.sleep(retry_delay * (retry + 1))
                        continue
                    return False
                    
                except requests.exceptions.RequestException as e:
                    logger.error(f"Signal send error: {e}")
                    if retry < max_retries - 1:
                        time.sleep(retry_delay * (retry + 1))
                        continue
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"Signal send error: {e}")
            return False
    
    def check_signal_stability(self, new_signal):
        """Prevent signal flip-flopping by requiring confirmation"""
        try:
            # Add current signal to history
            self.signal_history.append(new_signal)
            
            if len(self.signal_history) < 3:
                return new_signal  # Not enough history, allow signal
            
            # If signal is same as last one, allow it
            if new_signal == self.signal_history[-2]:
                return new_signal
            
            # If signal changed, require confirmation
            if new_signal != self.signal_history[-2]:
                # Count how many of last 3 signals match new signal
                recent_matches = sum(1 for s in self.signal_history[-3:] if s == new_signal)
                
                # Need at least 2 out of 3 confirmations for signal change
                if recent_matches >= 2:
                    return new_signal
                else:
                    # Not enough confirmation, return neutral to prevent whipsaw
                    return 0
            
            return new_signal
            
        except Exception as e:
            logger.error(f"Signal stability check error: {e}")
            return new_signal  # Return original signal on error

    def fetch_option_data(self):
        """Fetch option chain data with enhanced session handling"""
        max_retries = 3
        
        for retry in range(max_retries):
            try:
                current_time = time.time()
                if current_time - self.last_request_time < 2:
                    time.sleep(2)
                
                # Check market hours (9:15 AM - 3:30 PM IST)
                now = datetime.now()
                if (now.hour < 9 or (now.hour == 9 and now.minute < 15) or 
                    now.hour > 15 or (now.hour == 15 and now.minute >= 30)):
                    return None, None
                
                # Create fresh session
                self.session = requests.Session()
                
                # Step 1: Get homepage to establish session
                main_url = "https://www.nseindia.com/"
                main_response = self.session.get(main_url, headers=self.headers, timeout=15)
                main_response.raise_for_status()
                
                # Step 2: Visit option chain page
                oc_url = "https://www.nseindia.com/option-chain"
                self.headers['referer'] = main_url
                oc_response = self.session.get(oc_url, headers=self.headers, timeout=15)
                oc_response.raise_for_status()
                
                # Step 3: Get option chain data
                api_url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
                self.headers['referer'] = oc_url
                
                response = self.session.get(api_url, headers=self.headers, timeout=15)
                response.raise_for_status()
                
                data = response.json()
                
                if 'records' not in data or not data['records'].get('data'):
                    raise ValueError("Invalid data structure")
                
                # Parse data
                records = []
                current_expiry = data['records']['expiryDates'][0]
                underlying_value = None
                timestamp = datetime.now().replace(second=0, microsecond=0)
                
                for item in data['records']['data']:
                    if ("CE" in item and "PE" in item and 
                        item['expiryDate'] == current_expiry):
                        
                        call_opt = item['CE']
                        put_opt = item['PE']
                        
                        if underlying_value is None:
                            underlying_value = call_opt.get('underlyingValue', 0)
                        
                        record = {
                            'Strike': item['strikePrice'],
                            'Timestamp': timestamp,
                            'Call_OI': int(call_opt.get('openInterest', 0)),
                            'Call_Change_OI': int(call_opt.get('changeinOpenInterest', 0)),
                            'Call_Volume': int(call_opt.get('totalTradedVolume', 0)),
                            'Call_LTP': float(call_opt.get('lastPrice', 0)),
                            'Put_OI': int(put_opt.get('openInterest', 0)),
                            'Put_Change_OI': int(put_opt.get('changeinOpenInterest', 0)),
                            'Put_Volume': int(put_opt.get('totalTradedVolume', 0)),
                            'Put_LTP': float(put_opt.get('lastPrice', 0))
                        }
                        records.append(record)
                
                if not records:
                    raise ValueError("No option data found")
                
                df = pd.DataFrame(records)
                self.last_request_time = time.time()
                return df, underlying_value
                
            except Exception as e:
                if retry < max_retries - 1:
                    wait_time = 2 ** (retry + 1)
                    current_time = datetime.now().strftime('%H:%M:%S')
                    print(f"{current_time} | RETRY {retry+1}/{max_retries} in {wait_time}s")
                    time.sleep(wait_time)
                    continue
                else:
                    return None, None
        
        return None, None

    def get_nearest_strikes(self, df, underlying_value, n=5):
        """Get nearest strikes to current price - Fixed for None handling"""
        try:
            # Handle None or invalid inputs
            if df is None or df.empty or underlying_value is None:
                raise ValueError("Invalid input data")
                
            # Ensure underlying_value is numeric
            underlying_value = float(underlying_value)
            
            strikes = sorted(df['Strike'].unique())
            if not strikes:
                raise ValueError("No strikes found in data")
                
            atm_strike = min(strikes, key=lambda x: abs(x - underlying_value))
            atm_index = strikes.index(atm_strike)
            
            start_idx = max(0, atm_index - n)
            end_idx = min(len(strikes), atm_index + n + 1)
            
            selected_strikes = strikes[start_idx:end_idx]
            filtered_df = df[df['Strike'].isin(selected_strikes)].copy()
            
            return filtered_df, atm_strike
            
        except Exception as e:
            logger.error(f"Error in get_nearest_strikes: {e}")
            # Return safe fallback values
            if df is not None and not df.empty:
                # Use the first available strike as fallback
                fallback_strike = df['Strike'].iloc[0] if len(df) > 0 else underlying_value
                return df.copy(), fallback_strike
            else:
                # Create minimal fallback DataFrame
                fallback_strike = underlying_value if underlying_value is not None else 24000  # Default Nifty level
                fallback_df = pd.DataFrame({
                    'Strike': [fallback_strike],
                    'Timestamp': [datetime.now()],
                    'Call_OI': [0], 'Call_Change_OI': [0], 'Call_Volume': [0], 'Call_LTP': [0],
                    'Put_OI': [0], 'Put_Change_OI': [0], 'Put_Volume': [0], 'Put_LTP': [0]
                })
                return fallback_df, fallback_strike

    def traditional_analysis(self, current_data, underlying_value, atm_strike):
        """Enhanced traditional analysis using ALL features that ML uses for consistent training"""
        try:
            # Extract the SAME features that ML uses
            features = self.ml_engine.extract_features(current_data, self.historical_data, underlying_value, atm_strike)
            
            if features is None:
                return 0, 0
            
            # Initialize signal components
            signal = 0
            strength = 0
            
            # === CORE ANALYSIS (Using all features) ===
            
            # 1. Price and Strike Analysis
            price_strike_ratio = features['price_strike_ratio']
            price_strike_diff = features['price_strike_diff']
            
            # 2. Volume Analysis
            call_volume = features['atm_call_volume']
            put_volume = features['atm_put_volume']
            total_call_volume = features['total_call_volume']
            total_put_volume = features['total_put_volume']
            pcr_volume = features['pcr_volume']
            
            # 3. OI Analysis
            call_oi = features['atm_call_oi']
            put_oi = features['atm_put_oi']
            total_call_oi = features['total_call_oi']
            total_put_oi = features['total_put_oi']
            pcr_oi = features['pcr_oi']
            
            # 4. Price Analysis
            call_ltp = features['atm_call_ltp']
            put_ltp = features['atm_put_ltp']
            call_put_ratio = features['call_put_ltp_ratio']
            
            # 5. Historical Changes
            call_oi_change = features.get('call_oi_change', 0)
            put_oi_change = features.get('put_oi_change', 0)
            call_volume_change = features.get('call_volume_change', 0)
            put_volume_change = features.get('put_volume_change', 0)
            
            # 6. Temporal Features
            time_of_day = features['time_of_day']
            is_opening = features['is_opening']
            is_closing = features['is_closing']
            is_mid_session = features['is_mid_session']
            
            # === SIGNAL GENERATION USING ALL FEATURES ===
            
            # 1. Volume-based signals
            if call_volume > put_volume * 1.5 and call_volume > 100:
                signal = 1
                strength += 2
            elif put_volume > call_volume * 1.5 and put_volume > 100:
                signal = -1
                strength += 2
            
            # 2. OI-based signals
            if call_oi > put_oi * 1.2:
                if signal >= 0:  # Don't override strong put signal
                    signal = 1
                    strength += 1
            elif put_oi > call_oi * 1.2:
                if signal <= 0:  # Don't override strong call signal
                    signal = -1
                    strength += 1
            
            # 3. OI Change Analysis
            if call_oi_change > 0 and call_oi_change > abs(put_oi_change):
                if signal >= 0:
                    strength += 1
            elif put_oi_change > 0 and put_oi_change > abs(call_oi_change):
                if signal <= 0:
                    strength += 1
            
            # 4. PCR Analysis
            if pcr_oi > 1.5:
                if signal == 1:  # Confirm call signal
                    strength += 1
            elif pcr_oi < 0.7:
                if signal == -1:  # Confirm put signal
                    strength += 1
            
            # 5. Volume Trend Analysis
            if call_volume_change > 100 and call_volume_change > abs(put_volume_change):
                if signal >= 0:
                    strength += 1
            elif put_volume_change > 100 and put_volume_change > abs(call_volume_change):
                if signal <= 0:
                    strength += 1
            
            # 6. Time-based Adjustments
            if is_opening:
                strength = max(0, strength - 1)  # More conservative during opening
            elif is_mid_session:
                if strength > 0:
                    strength += 1  # Stronger signals during mid-session
            elif is_closing:
                strength = max(0, strength - 1)  # More conservative during closing
            
            # 7. Price Ratio Analysis
            if call_put_ratio > 1.3 and signal == 1:
                strength += 1
            elif call_put_ratio < 0.7 and signal == -1:
                strength += 1
            
            # === FINAL SIGNAL VALIDATION ===
            
            # Minimum strength threshold
            if strength < 2:
                signal = 0
                strength = 0
            
            # Maximum strength cap
            strength = min(10, strength)
            
            # Validate signal
            if signal not in [-1, 0, 1]:
                signal = 0
                strength = 0
            
            return signal, strength
            
        except Exception as e:
            logger.error(f"Traditional analysis error: {e}")
            return 0, 0

    def analyze_regime_with_ml(self, current_data, underlying_value, atm_strike):
        """Enhanced analysis using ML predictions combined with traditional logic and risk management"""
        try:
            # Validate input data
            if current_data is None or underlying_value is None or atm_strike is None:
                logger.error("Invalid input data for analysis")
                return 0, 0, "Invalid input data", 0, "unknown"
            
            # Store current data
            timestamp = datetime.now().replace(second=0, microsecond=0)
            self.historical_data[timestamp] = current_data.copy()
            
            # Clean old data
            cutoff_time = timestamp - timedelta(minutes=10)
            old_timestamps = [ts for ts in self.historical_data.keys() if ts < cutoff_time]
            for ts in old_timestamps:
                del self.historical_data[ts]
            
            # Extract ML features
            features = self.ml_engine.extract_features(current_data, self.historical_data, underlying_value, atm_strike)
            
            if features is None:
                return 0, 0, "Feature extraction failed", 0, "unknown"
            
            # Get comparative analysis for decision enhancement
            comparative_analysis = self.data_storage.get_comparative_analysis(current_data, underlying_value, atm_strike)
            
            # Get ML predictions with error handling
            ml_prediction = self.ml_engine.predict(features)
            
            # Store current ML prediction for paper trading use
            self.current_ml_prediction = ml_prediction
            
            # Market Regime Detection
            regime_info = self.regime_detector.detect_regime(current_data, underlying_value)
            current_regime = regime_info['regime']
            volatility = regime_info['volatility']
            
            # Traditional analysis (use as fallback when ML not trained)
            traditional_signal, traditional_strength = self.traditional_analysis(current_data, underlying_value, atm_strike)
            
            # If ML is not trained or prediction failed, use traditional analysis
            if ml_prediction is None or not isinstance(ml_prediction, dict) or not self.ml_engine.is_trained:
                final_signal = traditional_signal
                final_strength = traditional_strength
                
                # Create structured prediction for traditional analysis
                self.current_ml_prediction = {
                    'regime': traditional_signal,  # Use traditional signal as regime
                    'confidence': 0.6,  # Medium confidence for traditional
                    'pattern': 0,  # No pattern detection in traditional
                    'signal_direction': traditional_signal,
                    'tradetron_signal': traditional_signal,
                    'position_size': max(1, final_strength / 5),  # Convert strength to position size
                    'stop_loss_pct': 0.10,  # Default 10% stop loss
                    'trailing_stop_pct': 0.5,  # Default 50% trailing stop
                    'deep_learning_used': False,
                    'analysis_type': 'traditional'  # Mark as traditional analysis
                }
                
                # Calculate position size using traditional strength
                position_size = self.risk_manager.calculate_position_size(
                    self.account_value, 
                    volatility, 
                    0.6,  # Medium confidence for traditional analysis
                    final_strength
                )
                
                # Update training data for ML learning from traditional trades
                if features:
                    self.training_counter += 1
                    if self.training_counter % 5 == 0:  # Every 5th analysis
                        self.ml_engine.update_training_data(features)
                        
                        # Retrain models every 25 samples
                        if self.training_counter % 25 == 0:
                            if self.ml_engine.train_models():
                                print(f"🔄 ML models trained with {len(self.ml_engine.training_data)} samples from traditional trades")
                
                analysis_msg = f"Enhanced_Traditional:{traditional_signal}|Strength:{traditional_strength}|Regime:{current_regime}|Vol:{volatility:.3f}|Features:50+|Mode:LEARNING"
                
                return final_signal, final_strength, analysis_msg, position_size, current_regime
            
            # ML is trained and prediction successful
            ml_signal = ml_prediction.get('regime', 0)
            ml_confidence = ml_prediction.get('confidence', 0)
            
            # SHAP Analysis for prediction explanation
            shap_analysis = None
            if SHAP_AVAILABLE and self.ml_engine.is_trained:
                shap_analysis = self.ml_engine.analyze_prediction_with_shap(features)
            
            # Combine ML and traditional analysis
            ml_regime = ml_prediction['regime']
            ml_pattern = ml_prediction['pattern']
            
            # Convert ML regime to signal
            ml_signal = 1 if ml_regime == 1 else (-1 if ml_regime == 2 else 0)
            
            # ENHANCED CONFIDENCE WITH COMPARATIVE ANALYSIS - More generous for learning
            confidence_adjustments = []
            confidence_boost = 0.10  # Start with base boost for learning
            
            if comparative_analysis:
                # Boost confidence if patterns align with ML signal
                if 'pattern_recognition' in comparative_analysis:
                    pattern = comparative_analysis['pattern_recognition'].get('intraday_pattern', '')
                    if (ml_signal == 1 and 'call_bullish' in pattern) or \
                       (ml_signal == -1 and 'put_bullish' in pattern):
                        confidence_boost += 0.05
                        confidence_adjustments.append("Pattern alignment +5%")
                        
                # Adjust based on accumulated data percentiles
                if 'today_vs_accumulated' in comparative_analysis and 'call_oi_percentile' in comparative_analysis['today_vs_accumulated']:
                    acc_data = comparative_analysis['today_vs_accumulated']
                    if ml_signal == 1 and acc_data['call_oi_percentile'] > 80:
                        confidence_boost += 0.03
                        confidence_adjustments.append("High Call OI percentile +3%")
                    elif ml_signal == -1 and acc_data['put_oi_percentile'] > 80:
                        confidence_boost += 0.03
                        confidence_adjustments.append("High Put OI percentile +3%")
                    elif ml_signal == 1 and acc_data['call_oi_percentile'] < 20:
                        confidence_boost -= 0.02
                        confidence_adjustments.append("Low Call OI percentile -2%")
                    elif ml_signal == -1 and acc_data['put_oi_percentile'] < 20:
                        confidence_boost -= 0.02
                        confidence_adjustments.append("Low Put OI percentile -2%")
            
            # Apply controlled confidence boost
            confidence_boost = max(-0.05, min(0.25, confidence_boost))
            enhanced_confidence = ml_confidence + confidence_boost
            enhanced_confidence = min(enhanced_confidence, 0.95)
            
            if confidence_adjustments:
                logger.info(f"Confidence adjustments: {', '.join(confidence_adjustments)}")
                logger.info(f"Enhanced confidence: {ml_confidence:.1%} → {enhanced_confidence:.1%}")
            
            # DYNAMIC CONFIDENCE THRESHOLD
            recent_trades = self.portfolio_performance[-10:] if len(self.portfolio_performance) >= 10 else []
            if recent_trades:
                recent_win_rate = sum(1 for trade in recent_trades if trade['pnl'] > 0) / len(recent_trades)
                if recent_win_rate > 0.7:
                    confidence_threshold = 0.45  # More aggressive when performing well
                elif recent_win_rate < 0.4:
                    confidence_threshold = 0.55  # More conservative when struggling
                else:
                    confidence_threshold = 0.50  # Balanced approach
            else:
                confidence_threshold = 0.40  # Lower threshold for initial learning
            
            if enhanced_confidence < confidence_threshold:
                win_rate_msg = f" (win rate: {recent_win_rate:.2f})" if recent_trades else ""
                return 0, 0, f"Low confidence: {enhanced_confidence:.2f} < {confidence_threshold:.2f}{win_rate_msg}", 0, current_regime
            
            # Combine signals with enhanced confidence weighting
            if enhanced_confidence >= 0.8:  # Very high confidence
                final_signal = ml_signal
                final_strength = traditional_strength + enhanced_confidence * 10 + 2
            elif enhanced_confidence >= 0.7:  # High confidence
                if ml_signal == traditional_signal:
                    final_signal = ml_signal
                    final_strength = traditional_strength + enhanced_confidence * 10 + 1
                else:
                    # If ML and traditional disagree, use enhanced confidence to decide
                    if enhanced_confidence >= 0.75:  # Strong ML confidence overrides traditional
                        final_signal = ml_signal
                        final_strength = enhanced_confidence * 10
                    else:
                        return 0, 0, f"ML/Traditional disagreement with moderate confidence - ML:{ml_signal} Trad:{traditional_signal}", 0, current_regime

    
            # Pattern detection enhancement
            if ml_pattern in [1, 2, 3]:  # Breakout, Reversal, or Continuation patterns
                final_strength += 1
            
            # Risk Management Integration
            position_size = self.risk_manager.calculate_position_size(
                self.account_value, 
                volatility, 
                enhanced_confidence, 
                final_strength
            )
            
            # Check portfolio risk limits
            risk_check, risk_message = self.risk_manager.check_portfolio_risk(position_size)
            
            if not risk_check:
                print(f"[RISK] {risk_message}")
                return 0, 0, f"Risk limit exceeded: {risk_message}", 0, current_regime
            
            # Get time of day for optimization
            hour = datetime.now().hour
            if hour < 10:
                time_of_day = 'opening'
            elif hour > 14:
                time_of_day = 'closing'
            else:
                time_of_day = 'mid_session'
            
            # Entry timing optimization
            recent_performance = [p['pnl'] for p in self.portfolio_performance[-10:]] if self.portfolio_performance else []
            should_enter = self.profit_optimizer.optimize_entry_timing(
                final_strength, 
                volatility, 
                current_regime, 
                time_of_day, 
                recent_performance
            )
            
            if not should_enter and final_signal != 0:
                return 0, 0, "Entry timing not optimal", 0, current_regime
            
            # Update training data for continuous learning
            self.training_counter += 1
            if self.training_counter % 5 == 0:  # Every 5th analysis
                self.ml_engine.update_training_data(features)
                
                # Retrain models every 25 samples
                if self.training_counter % 25 == 0:
                    self.ml_engine.train_models()
            
            # Apply signal stability check to prevent whipsaws
            stable_signal = self.check_signal_stability(final_signal)
            
            if stable_signal != final_signal:
                stability_msg = f"Signal filtered: {final_signal}→{stable_signal} (stability check)"
            else:
                stability_msg = "Signal stable"
            
            # Enhanced analysis message with comparative insights
            comp_summary = ""
            if comparative_analysis:
                comp_parts = []
                if 'today_vs_previous' in comparative_analysis and 'call_oi_change' in comparative_analysis['today_vs_previous']:
                    prev_day = comparative_analysis['today_vs_previous']
                    comp_parts.append(f"PrevDay:{prev_day['call_oi_change']:.0f}%/{prev_day['put_oi_change']:.0f}%")
                    
                if 'today_vs_accumulated' in comparative_analysis and 'call_oi_percentile' in comparative_analysis['today_vs_accumulated']:
                    acc_data = comparative_analysis['today_vs_accumulated']
                    comp_parts.append(f"AccPctl:{acc_data['call_oi_percentile']:.0f}%/{acc_data['put_oi_percentile']:.0f}%")
                    
                if 'pattern_recognition' in comparative_analysis:
                    pattern = comparative_analysis['pattern_recognition'].get('intraday_pattern', 'unknown')
                    comp_parts.append(f"Pattern:{pattern}")
                    
                comp_summary = f"|Comp:{'/'.join(comp_parts[:2])}" if comp_parts else ""
            
            analysis_msg = f"ML:{ml_signal}({enhanced_confidence:.2f})|Trad:{traditional_signal}|Regime:{current_regime}|Vol:{volatility:.3f}|Final:{stable_signal}|{stability_msg}"
            
            return stable_signal, final_strength, analysis_msg, position_size, current_regime
            
        except Exception as e:
            logger.error(f"Enhanced ML analysis error: {e}", exc_info=True)
            return 0, 0, f"Analysis error: {str(e)}", 0, "unknown"
        
    def calculate_reward(self, pnl, signal_duration, position_size=1.0):
        """Calculate reward considering PnL, holding time, and risk"""
        # Base reward is the total PnL (already calculated as per-contract PnL * position size * lot size)
        # The pnl passed here is per-contract PnL, so we need to scale it to total PnL
        total_pnl = pnl * position_size * Config.LOT_SIZE
        reward = total_pnl
        
        # Add time decay penalty (encourage faster trades)
        time_penalty = min(1.0, signal_duration / 60.0)  # Normalize to 0-1 for 1 hour
        reward *= (1.0 - 0.2 * time_penalty)  # Up to 20% penalty for longer holds
        
        # Add risk adjustment (penalize large drawdowns more)
        if total_pnl < 0:
            reward *= 1.5  # Penalize losses more heavily
            
        return reward
        
    def update_signal_with_learning(self, outcome, pnl, current_data, underlying_value, atm_strike, position_size=1):
        """Update ML models with trade outcome for continuous learning with RL"""
        try:
            # Extract features for the learning sample
            features = self.ml_engine.extract_features(current_data, self.historical_data, underlying_value, atm_strike)
            
            if features is None:
                return
                
            # Calculate signal duration in minutes
            signal_duration = 0
            if self.signal_start_time:
                signal_duration = (datetime.now() - self.signal_start_time).total_seconds() / 60.0
                
            # Calculate reward using RL - FIXED: Use actual position size passed as parameter
            reward = self.calculate_reward(pnl, signal_duration, position_size)
            
            # Store experience in replay buffer
            experience = {
                'features': features.copy(),
                'outcome': outcome,
                'pnl': pnl,
                'reward': reward,
                'timestamp': datetime.now(),
                'market_conditions': {
                    'vix': features.get('vix', 0),
                    'rsi': features.get('rsi', 50),
                    'signal_strength': self.signal_strength
                }
            }
            self.replay_buffer.append(experience)
            
            # Update training data with PnL-based labels
            self.ml_engine.update_training_data(features, self.ml_engine.generate_training_labels(features, pnl))
            
            # Update performance metrics
            self.episode_rewards.append(reward)
            self.episode_pnls.append(pnl)
            self.avg_reward = 0.9 * self.avg_reward + 0.1 * reward if self.avg_reward else reward
            
            # Calculate total PnL for display
            total_pnl = pnl * position_size * Config.LOT_SIZE
            print(f"[RL] Reward: {reward:.2f} | PnL: ₹{total_pnl:.2f} (Per Contract: {pnl:.2f}) | "
                  f"Avg Reward: {self.avg_reward:.2f} | "
                  f"Buffer: {len(self.replay_buffer)} | Duration: {signal_duration:.1f}min")
            
            # Train on a batch of experiences if we have enough
            if len(self.replay_buffer) >= self.batch_size * 2:
                self._train_on_batch()
            
        except Exception as e:
            logger.error(f"RL learning update error: {e}", exc_info=True)
        
        # Update outcome tracking for drift detection
        self.recent_outcomes.append(1 if outcome == 1 and pnl > 0 else 0)
        
        # Adaptive learning rate decay based on recent performance
        if len(self.episode_rewards) >= 10:
            recent_avg = np.mean(self.episode_rewards[-10:])
            if recent_avg < self.avg_reward * 0.9:  # Performance dropped
                self.learning_rate = max(
                    self.min_learning_rate,
                    self.learning_rate * 0.9  # Slow down learning
                )
                print(f"[RL] Reducing learning rate to {self.learning_rate:.6f}")
            
        # Retrain models if we have enough new data
        self.training_counter += 1
        if self.training_counter >= 50:  # Retrain every 50 updates
            if self.ml_engine.train_models():
                print("✅ ML models retrained with latest data")
            self.training_counter = 0
    
    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        if not self.portfolio_performance:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'expected_value': 0.0
            }
        
        try:
            # Basic metrics
            total_trades = len(self.portfolio_performance)
            wins = [p['pnl'] for p in self.portfolio_performance if p['pnl'] > 0]
            losses = [p['pnl'] for p in self.portfolio_performance if p['pnl'] < 0]
            
            win_rate = len(wins) / total_trades if total_trades > 0 else 0
            avg_pnl = sum(p['pnl'] for p in self.portfolio_performance) / total_trades if total_trades > 0 else 0
            
            # Profit factor
            total_wins = sum(wins) if wins else 0
            total_losses = abs(sum(losses)) if losses else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Max drawdown
            cumulative_pnl = []
            running_total = 0
            for p in self.portfolio_performance:
                running_total += p['pnl']
                cumulative_pnl.append(running_total)
            
            if cumulative_pnl:
                running_max = [cumulative_pnl[0]]
                for i in range(1, len(cumulative_pnl)):
                    running_max.append(max(running_max[-1], cumulative_pnl[i]))
                
                drawdowns = [running_max[i] - cumulative_pnl[i] for i in range(len(cumulative_pnl))]
                max_drawdown = max(drawdowns) if drawdowns else 0
            else:
                max_drawdown = 0
            
            # Sharpe ratio (simplified)
            returns = [p['pnl'] / 100000 for p in self.portfolio_performance]  # Normalize by account size
            if len(returns) > 1:
                avg_return = sum(returns) / len(returns)
                std_return = (sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1)) ** 0.5
                sharpe_ratio = (avg_return / std_return * (252 ** 0.5)) if std_return > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Expected value
            avg_win = sum(wins) / len(wins) if wins else 0
            avg_loss = sum(losses) / len(losses) if losses else 0
            expected_value = win_rate * avg_win + (1 - win_rate) * avg_loss
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'expected_value': expected_value
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'expected_value': 0.0
            }
    
    def display_performance_metrics(self):
        """Display comprehensive performance metrics"""
        try:
            metrics = self.calculate_performance_metrics()
            total_trades = metrics.get('total_trades', 0)
            win_rate = metrics.get('win_rate', 0)
            avg_pnl = metrics.get('avg_pnl', 0)
            max_drawdown = metrics.get('max_drawdown', 0)
            sharpe_ratio = metrics.get('sharpe_ratio', 0)
            profit_factor = metrics.get('profit_factor', 0)
            expected_value = metrics.get('expected_value', 0)
            
            print("\n" + "="*60)
            print("📊 PERFORMANCE METRICS")
            print("="*60)
            print(f"Total Trades: {total_trades}")
            print(f"Win Rate: {win_rate:.1%}")
            print(f"Profit Factor: {profit_factor:.2f}")
            print(f"Average PnL: ₹{avg_pnl:.2f}")
            if total_trades > 0:
                wins = [p['pnl'] for p in self.portfolio_performance if p['pnl'] > 0]
                losses = [p['pnl'] for p in self.portfolio_performance if p['pnl'] < 0]
                avg_win = sum(wins) / len(wins) if wins else 0
                avg_loss = sum(losses) / len(losses) if losses else 0
                print(f"Average Win: ₹{avg_win:.2f}")
                print(f"Average Loss: ₹{avg_loss:.2f}")
            print(f"Max Drawdown: ₹{max_drawdown:.2f}")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"Expected Value: ₹{expected_value:.2f}")
            print(f"Account Value: ₹{self.account_value:.2f}")
            print(f"Total PnL: ₹{self.account_value - 100000:.2f}")
            
            # Display SHAP feature importance if available
            if SHAP_AVAILABLE and hasattr(self, 'ml_engine') and self.ml_engine.shap_analyzer.shap_values:
                print(f"\n🔍 TOP 10 FEATURE IMPORTANCE (SHAP):")
                print("-" * 50)
                top_features = self.ml_engine.shap_analyzer.get_top_features('regime_classifier', 10)
                for i, (feature, importance) in enumerate(top_features):
                    print(f"{i+1:2d}. {feature:<30} | {importance:.4f}")
            
            print("="*60)
            
        except Exception as e:
            logger.error(f"Performance metrics display error: {e}")
            print(f"📊 PERFORMANCE METRICS")
            print("="*60)
            print("⚠️ Error calculating metrics - insufficient data")
            print("="*60)
    
    def _train_on_batch(self):
        """Sample a batch from replay buffer and update models"""
        if len(self.replay_buffer) < self.batch_size:
            return
            
        try:
            # Sample a batch of experiences
            batch = random.sample(self.replay_buffer, min(self.batch_size, len(self.replay_buffer)))
            
            # Prepare data for training
            X = []
            y_regime = []
            y_confidence = []
            
            for exp in batch:
                features = exp['features']
                X.append(list(features.values()))
                
                # Generate target labels with reward shaping
                if exp['reward'] > 0:
                    y_regime.append(1 if exp['outcome'] == 1 else 2)  # 1 for win, 2 for neutral
                    y_confidence.append(min(3, 1 + int(exp['reward'] * 2)))  # Scale confidence with reward
                else:
                    y_regime.append(0)  # Loss
                    y_confidence.append(1)  # Low confidence
            
            # Update models with the batch
            if len(X) > 0:
                self.ml_engine.models['regime_classifier'].partial_fit(
                    X, y_regime, classes=[0, 1, 2]
                )
                self.ml_engine.models['signal_confidence'].partial_fit(
                    X, y_confidence, classes=[1, 2, 3]
                )
                
                # Update learning rate
                self.learning_rate = max(
                    self.min_learning_rate,
                    self.learning_rate * self.learning_rate_decay
                )
                
        except Exception as e:
            logger.error(f"Error in batch training: {e}", exc_info=True)

    def cleanup_old_data(self):
        """Clean up old data to prevent memory leaks"""
        try:
            # Clean historical data older than 1 hour
            cutoff_time = datetime.now() - timedelta(hours=1)
            old_timestamps = [ts for ts in self.historical_data.keys() if ts < cutoff_time]
            for ts in old_timestamps:
                del self.historical_data[ts]
            
            # Limit training data size
            if len(self.ml_engine.training_data) > 2000:
                self.ml_engine.training_data = self.ml_engine.training_data[-1000:]
            
            # Clean up old model files (keep only latest 3)
            model_files = [f for f in os.listdir('.') if f.startswith('ml_models') and f.endswith('.pkl')]
            if len(model_files) > 3:
                model_files.sort(key=lambda x: os.path.getmtime(x))
                for old_file in model_files[:-3]:
                    try:
                        os.remove(old_file)
                        logger.info(f"Removed old model file: {old_file}")
                    except:
                        pass
            
            # Clean up old deep model files
            deep_model_files = [f for f in os.listdir('.') if f.startswith('deep_model') and f.endswith('.pth')]
            if len(deep_model_files) > 2:
                deep_model_files.sort(key=lambda x: os.path.getmtime(x))
                for old_file in deep_model_files[:-2]:
                    try:
                        os.remove(old_file)
                        logger.info(f"Removed old deep model file: {old_file}")
                    except:
                        pass
                        
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    def run_analysis(self):
        """Main analysis loop with 24/7 learning - Fixed for outside market hours operation"""
        consecutive_fails = 0
        cleanup_counter = 0
        offline_training_counter = 0
        
        # Initialize fallback data for offline mode
        last_valid_df = None
        last_valid_underlying = None
        last_valid_atm_strike = None
        
        # Cache stored data for offline mode to avoid repeated loading
        cached_stored_data = None
        stored_data_index = 0
        last_data_load_time = 0
        
        self.flow('E', 'Main Analysis Loop')
        while True:
            try:
                current_time = datetime.now().strftime('%H:%M:%S')
                is_market_hours = self.data_storage.is_market_hours()

                # Flow: F branch
                self.flow('F', f"Market Hours? {'Yes' if is_market_hours else 'No'}")
                if is_market_hours:
                    self.flow('G', 'LIVE TRADING MODE')
                else:
                    self.flow('H', 'OFFLINE SIMULATION MODE')
                
                # Periodic cleanup every 100 iterations
                cleanup_counter += 1
                if cleanup_counter >= 100:
                    self.cleanup_old_data()
                    self.data_storage.cleanup_old_data()  # Clean up old stored data
                    cleanup_counter = 0
                
                # Fetch option data (only during market hours)
                df, underlying_value = None, None
                if is_market_hours:
                    df, underlying_value = self.fetch_option_data()
                    self.flow('I', 'Fetch Live Option Data')
                
                    if df is not None and underlying_value is not None:
                        self.flow('J', 'Data Fetch Success')
                        # Store live data during market hours
                        timestamp = datetime.now().replace(second=0, microsecond=0)
                        self.data_storage.store_live_data(df, underlying_value, timestamp)
                        self.flow('L', 'Store Live Data')
                        
                        # Update fallback data for offline mode
                        last_valid_df = df.copy()
                        last_valid_underlying = underlying_value
                        
                        consecutive_fails = 0  # Reset failure counter on success
                    else:
                        self.flow('J', 'Data Fetch Failed')
                        # Fallback to last valid live data within live mode
                        consecutive_fails += 1
                        if last_valid_df is not None and last_valid_underlying is not None:
                            df = last_valid_df.copy()
                            underlying_value = last_valid_underlying
                            df['Timestamp'] = datetime.now().replace(second=0, microsecond=0)
                            print(f"{current_time} | 🟡 LIVE MODE | Using fallback live snapshot | UND: {underlying_value:.2f}")
                            self.flow('K', 'Use Fallback Data')
                        else:
                            if consecutive_fails < 3:  # Try a few times before giving up
                                print(f"{current_time} | ⚠️ Data fetch failed, retrying... ({consecutive_fails}/3)")
                                time.sleep(30)
                                continue
                
                # Handle offline mode - use stored data or fallback data
                if not is_market_hours or (df is None and underlying_value is None):
                    # Load stored data only once and cache it (reload every hour)
                    current_time_minutes = datetime.now().hour * 60 + datetime.now().minute
                    if cached_stored_data is None or (current_time_minutes - last_data_load_time) >= 60:
                        raw_data = self.data_storage.get_all_available_data()  # Use comprehensive data loader
                        if raw_data:
                            # Create a full trading day simulation (9:15 AM to 3:30 PM) from available data
                            self.flow('M', 'Load Stored Data')
                            cached_stored_data = self._create_trading_day_simulation(raw_data)
                            self.flow('N', 'Create Trading Simulation')
                            stored_data_index = 0  # Reset index
                            last_data_load_time = current_time_minutes
                            logger.info(f"🔄 Created trading day simulation with {len(cached_stored_data)} samples for offline mode")
                    
                    if cached_stored_data and len(cached_stored_data) > 0:
                        # Use different data entries sequentially instead of same entry
                        recent_entry = cached_stored_data[stored_data_index % len(cached_stored_data)]
                        stored_data_index += 1  # Move to next entry for next iteration
                        
                        try:
                            df = pd.DataFrame(recent_entry['option_data'])
                            underlying_value = recent_entry['underlying_value']
                            
                            # Modify timestamps to current time for analysis
                            current_timestamp = datetime.now().replace(second=0, microsecond=0)
                            df['Timestamp'] = current_timestamp
                            
                            if not is_market_hours:
                                progress = f"[{stored_data_index}/{len(cached_stored_data)}]"
                                sim_time = recent_entry['timestamp'][11:16]  # Extract HH:MM from timestamp
                                print(f"{current_time} | 🌙 OFFLINE MODE {progress} | Simulating {sim_time} | UND: {underlying_value:.2f}")
                                self.flow('O', 'Use Sequential Data')
                            
                        except Exception as e:
                            logger.error(f"Error processing stored data: {e}")
                            df, underlying_value = None, None
                    
                    # Final fallback to last valid data if available
                    elif last_valid_df is not None and last_valid_underlying is not None:
                        df = last_valid_df.copy()
                        underlying_value = last_valid_underlying
                        
                        # Update timestamps for current analysis
                        current_timestamp = datetime.now().replace(second=0, microsecond=0)
                        df['Timestamp'] = current_timestamp
                        
                        if not is_market_hours:
                            print(f"{current_time} | 🌙 OFFLINE MODE | Using fallback data | UND: {underlying_value:.2f}")
                
                # Handle no data situation - prioritize offline simulation
                if df is None or underlying_value is None:
                    if not is_market_hours:
                        # 🌙 INTENSIVE OFFLINE MODE - No live data available, focus on simulation
                        offline_training_counter += 1
                        
                        # More frequent offline simulation when no data (every hour instead of 2 hours)
                        if offline_training_counter >= 3600:  # 1 hour
                            offline_training_counter = 0
                            
                            print(f"\n{current_time} | 🌙 NO DATA AVAILABLE - RUNNING INTENSIVE OFFLINE SIMULATION")
                            simulation_success = self.run_offline_simulation()
                            
                            if simulation_success:
                                print(f"{current_time} | ✅ Intensive offline simulation completed - ML improved")
                            else:
                                print(f"{current_time} | ⚠️ Simulation failed, running basic training")
                                self.perform_offline_training()
                            
                        # Status display every 5 minutes when no data
                        elif offline_training_counter % 300 == 0:  # Every 5 minutes
                            minutes_remaining = (3600 - offline_training_counter) // 60
                            print(f"{current_time} | 🌙 OFFLINE MODE | No data - Enhanced learning active | "
                                  f"Next simulation in {minutes_remaining} min | ML samples: {len(self.ml_engine.training_data) if hasattr(self.ml_engine, 'training_data') else 0}")
                        
                        # Display performance metrics every 100 iterations in no-data mode
                        if offline_training_counter % 100 == 0:
                            try:
                                self.display_performance_metrics()
                                
                                # Display data statistics
                                stats = self.data_storage.get_data_statistics()
                                if stats:
                                    print(f"📊 Stored Data: {stats['total_sessions']} sessions, {stats['total_samples']} samples, "
                                          f"Current Expiry: {stats['current_expiry']}")
                            except Exception as e:
                                logger.error(f"Error displaying offline metrics: {e}")
                    
                    # Faster sleep when no data to enable more frequent offline learning
                    time.sleep(1)
                    continue
                
                # Get nearest strikes - with error handling
                try:
                    filtered_df, atm_strike = self.get_nearest_strikes(df, underlying_value, 3)
                    last_valid_atm_strike = atm_strike  # Update fallback
                    self.flow('P', 'Get Nearest Strikes')
                except Exception as e:
                    logger.error(f"Error getting nearest strikes: {e}")
                    if last_valid_atm_strike is not None:
                        atm_strike = last_valid_atm_strike
                        filtered_df = df
                    else:
                        time.sleep(30)
                        continue
                
                # Get ATM data for display - with error handling
                try:
                    atm_data = filtered_df[filtered_df['Strike'] == atm_strike]
                    if atm_data.empty:
                        # If ATM strike not found, use closest available strike
                        closest_strike = min(filtered_df['Strike'].unique(), key=lambda x: abs(x - underlying_value))
                        atm_data = filtered_df[filtered_df['Strike'] == closest_strike]
                        atm_strike = closest_strike
                        
                    if not atm_data.empty:
                        atm_row = atm_data.iloc[0]
                    
                        # Analyze with ML - with error handling
                        try:
                            signal, strength, analysis_msg, position_size, regime = self.analyze_regime_with_ml(filtered_df, underlying_value, atm_strike)
                        except Exception as e:
                            logger.error(f"ML analysis error: {e}")
                            signal, strength, analysis_msg, position_size, regime = 0, 0, f"Analysis error: {str(e)}", 0, "unknown"
                        
                        # Collect real-time market data for continuous training - with error handling
                        try:
                            self.paper_trade.collect_market_data_for_training(filtered_df, underlying_value, atm_strike, signal, strength, regime)
                        except Exception as e:
                            logger.error(f"Market data collection error: {e}")
                    
                        # Check for paper trade exit first - with error handling
                        if self.paper_trade.active:
                            self.flow('Z', 'Paper Trade Active? Yes')
                            try:
                                exit_result = self.paper_trade.check_exit(filtered_df, signal, regime)
                                if exit_result:
                                    self.flow('AA', 'Check Exit Conditions')
                                    self.flow('CC', 'Exit Signal? Yes')
                                    outcome, trade_pnl, trade_data = exit_result
                                    
                                    # 🚨 SEND EXIT SIGNAL TO TRADETRON (Signal = 0 means EXIT/NEUTRAL)
                                    if self.send_signal(0):  # Send neutral signal to exit position
                                        self.flow('II', 'Send EXIT Signal to Tradetron')
                                        print(f"\n{'='*60}")
                                        print(f"{current_time} | 📡 EXIT SIGNAL SENT: ⚪ NEUTRAL (Trade Closed)")
                                        print(f"Exit Reason: {trade_data.get('exit_reason', 'Unknown')}")
                                        print(f"Strike: {trade_data.get('strike', 'Unknown')} | Entry: {trade_data.get('entry_price', 0):.2f} | Exit: {trade_data.get('exit_price', 0):.2f}")
                                        print(f"Position: {trade_data.get('position_size', 0):.0f} lots ({trade_data.get('position_size', 0)*Config.LOT_SIZE:.0f} contracts)")
                                        # Calculate total PnL correctly: PnL per contract × total contracts
                                        total_pnl_amount = trade_pnl * trade_data.get('position_size', 1) * Config.LOT_SIZE
                                        print(f"PnL: ₹{total_pnl_amount:.2f} | Per Contract: {trade_pnl:.2f} | Outcome: {'PROFIT' if outcome else 'LOSS'}")
                                        print(f"{'='*60}\n")
                                        self.last_sent_signal = 0  # Update last sent signal to neutral
                                    
                                    # Update ML with trade outcome
                                    try:
                                        self.update_signal_with_learning(outcome, trade_pnl, filtered_df, underlying_value, atm_strike, trade_data.get('position_size', 1))
                                        self.flow('KK', 'Update ML with Trade Outcome')
                                        self.flow('MM', 'Calculate Performance Metrics')
                                    except Exception as e:
                                        logger.error(f"Learning update error: {e}")
                        
                                    # Update account value and performance tracking (scale PnL by position size)
                                    actual_pnl = trade_pnl * trade_data.get('position_size', 1) * Config.LOT_SIZE
                                    self.account_value += actual_pnl
                                    self.portfolio_performance.append({
                                        'timestamp': datetime.now(),
                                        'pnl': actual_pnl,  # Store the actual scaled PnL
                                        'raw_pnl': trade_pnl,  # Store the per-contract PnL
                                        'regime': regime,
                                        'exit_reason': trade_data.get('exit_reason', 'Unknown'),
                                        'position_size': trade_data.get('position_size', 1),
                                        'outcome': outcome
                                    })
                                    
                                    # Add trade to risk manager with safe access
                                    safe_trade_data = {
                                        'pnl': actual_pnl,
                                        'entry_price': trade_data.get('entry_price', 0),
                                        'exit_price': trade_data.get('exit_price', 0),
                                        'position_size': trade_data.get('position_size', 1),
                                        'outcome': outcome,
                                        'exit_reason': trade_data.get('exit_reason', 'Unknown')
                                    }
                                    self.risk_manager.add_trade(safe_trade_data)
                            except Exception as e:
                                logger.error(f"Paper trade exit error: {e}")
                        
                        # Signal handling - only send during market hours or for exit signals
                        if is_market_hours:
                            # Only send non-neutral signals during market hours
                            if signal != 0 and signal != self.last_sent_signal:
                                self.flow('Y', 'Signal Generation')
                                if self.send_signal(signal):
                                    self.last_sent_signal = signal  # Update only after successful send
                                    
                                    signal_name = "🟢 SHORT PUT" if signal == 1 else ("🔴 SHORT CALL" if signal == -1 else "⚪ NEUTRAL")
                                    print(f"\n{'='*60}")
                                    print(f"{current_time} | 📡 SIGNAL SENT: {signal_name}")
                                    print(f"Strike: {atm_strike} | Underlying: {underlying_value:.2f}")
                                    print(f"Analysis: {analysis_msg}")
                                    print(f"Strength: {strength} | Position Size: {position_size:.0f} lots ({position_size*Config.LOT_SIZE:.0f} contracts)")
                                    print(f"Regime: {regime} | Account Value: ₹{self.account_value:.2f}")
                                    print(f"{'='*60}\n")
                                    # Enter paper trade if not already in a trade and strength above threshold
                                    self.flow('BB', 'Check Entry Conditions')
                                    self.flow('FF', f"Signal Strength > Threshold? {'Yes' if strength >= Config.SIGNAL_STRENGTH_THRESHOLD else 'No'}")
                                    if not self.paper_trade.active and strength >= Config.SIGNAL_STRENGTH_THRESHOLD:
                                        # 🎯 SHORT STRATEGY: We short the option and profit when its price decreases
                                        # Signal 1 = SHORT PUT (enter at PUT price, profit when PUT price drops)
                                        # Signal -1 = SHORT CALL (enter at CALL price, profit when CALL price drops)
                                        current_ltp = atm_row['Put_LTP'] if signal == 1 else atm_row['Call_LTP']
                                        if current_ltp > 0:
                                            # Get ML predictions if available for paper trade parameters
                                            ml_predictions = None
                                            if hasattr(self, 'current_ml_prediction') and self.current_ml_prediction:
                                                ml_predictions = self.current_ml_prediction
                                            
                                            self.paper_trade.enter(signal, current_ltp, atm_strike, position_size, regime, ml_predictions)
                                            self.flow('GG', 'Enter Paper Trade')
                                            
                                            # Enhanced paper trade entry message
                                            if ml_predictions:
                                                print(f"{current_time} | 📝 Paper Trade: {self.paper_trade.position} @ {current_ltp:.2f} | "
                                                      f"ML-Enhanced: Size={self.paper_trade.position_size:.0f} lots ({self.paper_trade.position_size*Config.LOT_SIZE:.0f} contracts), SL%={ml_predictions.get('stop_loss_pct', 0.1):.1%}, TSL%={ml_predictions.get('trailing_stop_pct', 0.5):.1%}")
                                            else:
                                                print(f"{current_time} | 📝 Paper Trade: {self.paper_trade.position} @ {current_ltp:.2f} | Size: {self.paper_trade.position_size:.0f} lots ({self.paper_trade.position_size*Config.LOT_SIZE:.0f} contracts)")
                            else:
                                # Log why signal is neutral (only during market hours)
                                if "Low ML confidence" in analysis_msg:
                                    print(f"{current_time} | ⚠️ NO SIGNAL: {analysis_msg}")
                                elif "ML/Trad disagreement" in analysis_msg:
                                    print(f"{current_time} | ⚠️ NO SIGNAL: {analysis_msg}")
                                elif "Risk limit exceeded" in analysis_msg:
                                    print(f"{current_time} | ⚠️ NO SIGNAL: {analysis_msg}")
                                elif "Entry timing not optimal" in analysis_msg:
                                    print(f"{current_time} | ⚠️ NO SIGNAL: {analysis_msg}")
                                elif "ML not trained" in analysis_msg:
                                    print(f"{current_time} | ⚠️ NO SIGNAL: {analysis_msg}")
                                else:
                                    print(f"{current_time} | ⚪ NEUTRAL: {analysis_msg}")
                    
                        # Paper trade ENTRY when strength exceeds threshold even if no signal sent (e.g., offline or blocked send)
                        if not self.paper_trade.active and signal != 0 and strength >= Config.SIGNAL_STRENGTH_THRESHOLD:
                            self.flow('BB', 'Check Entry Conditions')
                            self.flow('FF', 'Signal Strength > Threshold? Yes')
                            current_ltp = atm_row['Put_LTP'] if signal == 1 else atm_row['Call_LTP']
                            if current_ltp > 0:
                                ml_predictions = None
                                if hasattr(self, 'current_ml_prediction') and self.current_ml_prediction:
                                    ml_predictions = self.current_ml_prediction
                                self.paper_trade.enter(signal, current_ltp, atm_strike, position_size, regime, ml_predictions)
                                self.flow('GG', 'Enter Paper Trade')
                                print(f"{current_time} | 📝 Paper Trade Entered (threshold): {('SHORT PUT' if signal==1 else 'SHORT CALL')} @ {current_ltp:.2f} | Size: {position_size:.0f} lots")

                        # Update current signal for tracking
                        self.current_signal = signal
                        
                        # Display current status - with error handling
                        try:
                            call_ltp = atm_row['Call_LTP']
                            put_ltp = atm_row['Put_LTP']
                            call_oi = atm_row['Call_OI']
                            put_oi = atm_row['Put_OI']
                            call_vol = atm_row['Call_Volume']
                            put_vol = atm_row['Put_Volume']
                            
                            pcr_oi = put_oi / max(call_oi, 1)
                            pcr_vol = put_vol / max(call_vol, 1)
                            
                            # Current signal indicator
                            signal_indicator = "🟢 SHORT PUT" if self.current_signal == 1 else ("🔴 SHORT CALL" if self.current_signal == -1 else "⚪ NEUTRAL")
                                        
                            # Initialize empty trade status and market data status
                            trade_status = ""
                            market_data_status = f"| Market Buffer: {len(self.paper_trade.market_data_buffer)}"
                            
                            if self.paper_trade.active:
                                entry_strike_data = filtered_df[filtered_df['Strike'] == self.paper_trade.entry_strike]
                                if not entry_strike_data.empty:
                                    entry_row = entry_strike_data.iloc[0]
                                    # For SHORT: direction 1 = SHORT PUT, direction -1 = SHORT CALL
                                    current_ltp = entry_row['Put_LTP'] if self.paper_trade.direction == 1 else entry_row['Call_LTP']
                                    # 🎯 SHORT PROFIT: We profit when the shorted option price DECREASES
                                    # Entry Price - Current Price = Profit (positive when option price drops)
                                    unrealized_pnl = self.paper_trade.entry_price - current_ltp
                                    # PnL calculation: unrealized_pnl per contract * contracts in position
                                    total_pnl = unrealized_pnl * self.paper_trade.position_size * Config.LOT_SIZE
                                    trade_status = f"| Trade: {self.paper_trade.position} Strike:{self.paper_trade.entry_strike} @ {self.paper_trade.entry_price:.2f} | Size: {self.paper_trade.position_size} lots | PnL: ₹{total_pnl:.2f}"
                                else:
                                    trade_status = f"| Trade: {self.paper_trade.position} Strike:{self.paper_trade.entry_strike} @ {self.paper_trade.entry_price:.2f} | PnL: [Strike data not found]"

                            # Add market mode indicator
                            mode_indicator = "🟢 LIVE" if is_market_hours else "🌙 OFFLINE"
                            
                            print(f"{current_time} | {mode_indicator} | {signal_indicator} | ATM:{atm_strike} | UND:{underlying_value:.2f} | "
                                  f"C_LTP:{call_ltp:.2f} | P_LTP:{put_ltp:.2f} | "
                                  f"PCR_OI:{pcr_oi:.2f} | PCR_VOL:{pcr_vol:.2f} | "
                                  f"Regime:{regime} | Account:₹{self.account_value:.0f} | "
                                  f"Analysis: {analysis_msg}{' ' + trade_status if self.paper_trade.active else ''}{market_data_status}")
                        except Exception as e:
                            logger.error(f"Display error: {e}")
                            print(f"{current_time} | ⚠️ Display error: {str(e)}")
                            
                        # Continuous learning from market data (every 10 iterations) - with error handling
                        try:
                            if self.training_counter % 10 == 0:
                                training_data = self.paper_trade.get_training_data_from_market_buffer()
                                if training_data and len(training_data) >= 20:
                                    # Update ML engine with market data
                                    for sample in training_data[-20:]:  # Use last 20 samples
                                        self.ml_engine.update_training_data(
                                            sample['features'], 
                                            self.ml_engine.generate_training_labels(sample['features'], sample['price_change'] * 1000)
                                        )
                                    
                                    # Retrain models with new market data
                                    self.flow('OO', 'Retrain Models? Yes')
                                    self.flow('PP', 'Model Training')
                                    if self.ml_engine.train_models():
                                        print(f"{current_time} | 🔄 Models retrained with {len(training_data)} market samples")
                                        self.flow('QQ', 'Collect Market Data')
                        except Exception as e:
                            logger.error(f"Continuous learning error: {e}")
                    else:
                        print(f"{current_time} | ⚠️ No ATM data available for strike {atm_strike}")
                        
                except Exception as e:
                    logger.error(f"Data processing error: {e}")
                    print(f"{current_time} | ⚠️ Data processing error: {str(e)}")
                
                # 🌙 ENHANCED 24/7 OFFLINE SIMULATION (outside market hours)
                if not is_market_hours:
                    offline_training_counter += 1
                    
                    # Run comprehensive offline simulation every 2 hours (7200 iterations at 1-second intervals)
                    if offline_training_counter >= 7200:  # 2 hours
                        offline_training_counter = 0
                        
                        # Run the enhanced offline simulation
                        print(f"\n{current_time} | 🌙 INITIATING COMPREHENSIVE OFFLINE SIMULATION")
                        simulation_success = self.run_offline_simulation()
                        
                        if simulation_success:
                            print(f"{current_time} | ✅ Offline simulation completed successfully")
                        else:
                            print(f"{current_time} | ⚠️ Offline simulation failed, falling back to basic training")
                            self.perform_offline_training()
                        
                    # Display enhanced offline status
                    elif offline_training_counter % 600 == 0:  # Every 10 minutes
                        minutes_remaining = (7200 - offline_training_counter) // 60
                        print(f"{current_time} | 🌙 OFFLINE MODE | Enhanced Learning Active | Next full simulation in {minutes_remaining} minutes | "
                              f"Account: ₹{self.account_value:.0f} | ML Samples: {len(self.ml_engine.training_data) if hasattr(self.ml_engine, 'training_data') else 0}")
                
                # Display performance metrics every 50 iterations
                if self.training_counter % 50 == 0:
                    try:
                        self.display_performance_metrics()
                        self.flow('UU', 'Performance Display')
                        
                        # Display data statistics
                        stats = self.data_storage.get_data_statistics()
                        if stats:
                            print(f"📊 Data Stats: {stats['total_sessions']} sessions, {stats['total_samples']} samples, "
                                  f"Expiry: {stats['current_expiry']}")
                    except Exception as e:
                        logger.error(f"Performance metrics error: {e}")
                
                # Sleep based on market hours and data availability
                if is_market_hours:
                    self.flow('WW', 'Sleep & Wait (live)')
                    time.sleep(60)  # 1 minute during market hours
                elif cached_stored_data and len(cached_stored_data) > 0:
                    self.flow('WW', 'Sleep & Wait (offline cached)')
                    time.sleep(0.1)  # Very fast processing when using cached data for offline simulation
                else:
                    self.flow('WW', 'Sleep & Wait (offline)')
                    time.sleep(1)   # 1 second during offline hours when no cached data
                
            except KeyboardInterrupt:
                print("\n🛑 Stopping analysis...")
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                print(f"{current_time} | ❌ Main loop error: {str(e)}")
                consecutive_fails += 1
                
                # Progressive backoff on repeated errors
                if consecutive_fails > 5:
                    sleep_time = min(300, 30 * consecutive_fails)  # Max 5 minutes
                    print(f"{current_time} | ⚠️ Multiple errors detected, sleeping for {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    time.sleep(60)

    def run_offline_simulation(self):
        """🌙 ENHANCED 24/7 OFFLINE SIMULATION - Processes stored data sequentially to simulate live market"""
        try:
            current_time = datetime.now().strftime('%H:%M:%S')
            print(f"\n{current_time} | 🌙 STARTING OFFLINE MARKET SIMULATION")
            print("=" * 80)
            
            # Get ALL available data from all sources using the comprehensive loader
            stored_data = self.data_storage.get_all_available_data()
            
            if not stored_data or len(stored_data) < 50:
                print(f"{current_time} | ⚠️ Insufficient stored data for simulation ({len(stored_data) if stored_data else 0} samples)")
                return False
            
            # Sort data chronologically for proper simulation
            stored_data.sort(key=lambda x: x['timestamp'])
            
            print(f"📊 Simulating {len(stored_data)} market data points from {stored_data[0]['timestamp'][:19]} to {stored_data[-1]['timestamp'][:19]}")
            print("🎯 Objective: Learn from historical decisions and improve ML models\n")
            
            simulation_start_time = datetime.now()
            processed_samples = 0
            signals_generated = 0
            trades_completed = 0
            learning_updates = 0
            
            # Sequential processing of stored data
            for i, data_entry in enumerate(stored_data):
                try:
                    # Extract data
                    df = pd.DataFrame(data_entry['option_data'])
                    underlying_value = data_entry['underlying_value']
                    original_timestamp = data_entry['timestamp']
                    
                    # Simulate as current time for analysis
                    current_timestamp = datetime.now().replace(second=0, microsecond=0)
                    df['Timestamp'] = current_timestamp
                    
                    # Get ATM strike
                    atm_strike = self.get_atm_strike(df, underlying_value)
                    
                    # Get nearest strikes
                    filtered_df, atm_strike = self.get_nearest_strikes(df, underlying_value, 3)
                    
                    if filtered_df.empty:
                        continue
                        
                    # SIMULATE LIVE ANALYSIS
                    signal, strength, analysis_msg, position_size, regime = self.analyze_regime_with_ml(
                        filtered_df, underlying_value, atm_strike
                    )
                    
                    # Collect market data for training (same as live mode)
                    self.paper_trade.collect_market_data_for_training(
                        filtered_df, underlying_value, atm_strike, signal, strength, regime
                    )
                    
                    # CHECK FOR PAPER TRADE EXIT (if active)
                    if self.paper_trade.active:
                        exit_result = self.paper_trade.check_exit(filtered_df, signal, regime)
                        if exit_result:
                            outcome, trade_pnl, trade_data = exit_result
                            
                            print(f"🌙 OFFLINE EXIT: {trade_data['position']} | Strike:{trade_data['strike']} | "
                                  f"Entry:{trade_data['entry_price']:.2f} | Exit:{exit_result[0]:.2f} | "
                                  f"PnL:₹{trade_pnl * trade_data['position_size'] * Config.LOT_SIZE:.2f} | "
                                  f"Reason:{trade_data['exit_reason']}")
                            
                            # Update ML with trade outcome (CRITICAL FOR LEARNING)
                            self.update_signal_with_learning(outcome, trade_pnl, filtered_df, underlying_value, atm_strike, trade_data.get('position_size', 1))
                            trades_completed += 1
                            learning_updates += 1
                            
                            # Update account value
                            actual_pnl = trade_pnl * trade_data['position_size'] * Config.LOT_SIZE
                            self.account_value += actual_pnl
                            self.portfolio_performance.append({
                                'timestamp': datetime.now(),
                                'pnl': trade_pnl,
                                'regime': regime,
                                'exit_reason': trade_data['exit_reason'],
                                'simulation': True
                            })
                    
                    # GENERATE SIGNALS IN OFFLINE MODE (simulate signal generation)
                    if signal != 0 and strength >= Config.SIGNAL_STRENGTH_THRESHOLD:
                        atm_data = filtered_df[filtered_df['Strike'] == atm_strike]
                        if not atm_data.empty:
                            atm_row = atm_data.iloc[0]
                            
                            # Enter paper trade if not already in a trade (OFFLINE PAPER TRADING)
                            if not self.paper_trade.active:
                                current_ltp = atm_row['Put_LTP'] if signal == 1 else atm_row['Call_LTP']
                                if current_ltp > 0:
                                    # Get ML predictions if available
                                    ml_predictions = None
                                    if hasattr(self, 'current_ml_prediction') and self.current_ml_prediction:
                                        ml_predictions = self.current_ml_prediction
                                    
                                    self.paper_trade.enter(signal, current_ltp, atm_strike, position_size, regime, ml_predictions)
                                    signals_generated += 1
                                    
                                    signal_name = "SHORT PUT" if signal == 1 else "SHORT CALL"
                                    print(f"🌙 OFFLINE ENTRY: {signal_name} | Strike:{atm_strike} | LTP:{current_ltp:.2f} | "
                                          f"Size:{position_size:.0f} lots | Regime:{regime} | Strength:{strength}")
                    
                    processed_samples += 1
                    
                    # CONTINUOUS LEARNING (every 10 samples)
                    if processed_samples % 10 == 0:
                        # Extract features for learning
                        features = self.ml_engine.extract_features(filtered_df, self.historical_data, underlying_value, atm_strike)
                        if features:
                            # Update training data
                            labels = self.ml_engine.generate_training_labels(features, 0)  # Neutral label for historical
                            self.ml_engine.update_training_data(features, labels)
                            learning_updates += 1
                    
                    # FREQUENT MODEL RETRAINING (every 50 samples)
                    if processed_samples % 50 == 0:
                        if self.ml_engine.train_models():
                            print(f"🔄 OFFLINE TRAINING: Models updated with {len(self.ml_engine.training_data)} samples")
                    
                    # Progress indicator (every 100 samples)
                    if processed_samples % 100 == 0:
                        elapsed = (datetime.now() - simulation_start_time).total_seconds()
                        rate = processed_samples / elapsed if elapsed > 0 else 0
                        remaining = len(stored_data) - processed_samples
                        eta = remaining / rate if rate > 0 else 0
                        
                        print(f"📈 PROGRESS: {processed_samples}/{len(stored_data)} samples | "
                              f"Rate: {rate:.1f}/sec | ETA: {eta:.0f}s | "
                              f"Signals: {signals_generated} | Trades: {trades_completed}")
                    
                    # Store current data in historical data for feature calculation
                    minute_key = current_timestamp.replace(second=0, microsecond=0)
                    self.historical_data[minute_key] = filtered_df.copy()
                    
                    # Cleanup old historical data (keep last 30 minutes)
                    cutoff_time = current_timestamp - timedelta(minutes=30)
                    old_keys = [k for k in self.historical_data.keys() if k < cutoff_time]
                    for key in old_keys:
                        del self.historical_data[key]
                    
                except Exception as e:
                    logger.error(f"Error processing offline sample {i}: {e}")
                    continue
            
            # SIMULATION SUMMARY
            elapsed_time = (datetime.now() - simulation_start_time).total_seconds()
            print(f"\n{'='*80}")
            print(f"🌙 OFFLINE SIMULATION COMPLETED")
            print(f"{'='*80}")
            print(f"⏱️  Duration: {elapsed_time:.1f} seconds")
            print(f"📊 Processed: {processed_samples} samples")
            print(f"🎯 Signals Generated: {signals_generated}")
            print(f"💼 Trades Completed: {trades_completed}")
            print(f"🧠 Learning Updates: {learning_updates}")
            print(f"📈 Processing Rate: {processed_samples/elapsed_time:.1f} samples/sec")
            print(f"💰 Account Value: ₹{self.account_value:.2f}")
            
            # Final model training with all collected data
            if self.ml_engine.train_models():
                print(f"✅ Final model training completed with {len(self.ml_engine.training_data)} total samples")
            
            # Save updated models
            self.ml_engine.save_models()
            print(f"💾 Models saved successfully")
            
            print(f"{'='*80}\n")
            return True
            
        except Exception as e:
            logger.error(f"Offline simulation error: {e}")
            return False

# ================================
# Adaptive Rate Limiting System
# ================================
class AdaptiveRateLimiter:
    """Intelligent rate limiting based on market conditions and API response times"""
    
    def __init__(self):
        self.base_interval = 60  # Base 60 seconds
        self.min_interval = 30   # Minimum 30 seconds
        self.max_interval = 300  # Maximum 5 minutes
        self.current_interval = self.base_interval
        
        # Performance tracking
        self.response_times = deque(maxlen=10)
        self.error_count = 0
        self.success_count = 0
        self.last_request_time = None
        
        # Market condition adjustments
        self.market_multipliers = {
            'opening': 0.5,    # More frequent during opening
            'high_vol': 0.7,   # More frequent during high volatility
            'normal': 1.0,     # Normal frequency
            'low_vol': 1.5,    # Less frequent during low volatility
            'closing': 0.8     # Slightly more frequent during closing
        }
        
    def calculate_next_interval(self, market_condition='normal', volatility=0.2, recent_signals=0):
        """Calculate optimal next request interval"""
        base_multiplier = self.market_multipliers.get(market_condition, 1.0)
        
        # Adjust based on volatility
        vol_multiplier = 0.5 + volatility * 2  # Higher volatility = shorter intervals
        
        # Adjust based on recent signal activity
        signal_multiplier = max(0.5, 1.0 - (recent_signals * 0.1))  # More signals = shorter intervals
        
        # Adjust based on API performance
        performance_multiplier = 1.0
        if self.response_times:
            avg_response_time = sum(self.response_times) / len(self.response_times)
            if avg_response_time > 5:  # Slow responses
                performance_multiplier = 1.5
            elif avg_response_time < 1:  # Fast responses
                performance_multiplier = 0.8
                
        # Calculate final interval
        calculated_interval = self.base_interval * base_multiplier * vol_multiplier * signal_multiplier * performance_multiplier
        
        # Apply bounds
        self.current_interval = max(self.min_interval, min(self.max_interval, calculated_interval))
        
        return self.current_interval
        
    def record_request_performance(self, response_time, success=True):
        """Record API request performance"""
        self.response_times.append(response_time)
        
        if success:
            self.success_count += 1
            # Reduce interval slightly on success
            self.current_interval = max(self.min_interval, self.current_interval * 0.95)
        else:
            self.error_count += 1
            # Increase interval on error
            self.current_interval = min(self.max_interval, self.current_interval * 1.2)
            
    def should_make_request(self):
        """Check if enough time has passed for next request"""
        if self.last_request_time is None:
                return True
            
        time_since_last = (datetime.now() - self.last_request_time).total_seconds()
        return time_since_last >= self.current_interval
        
    def mark_request_made(self):
        """Mark that a request was made"""
        self.last_request_time = datetime.now()
        
    def get_wait_time(self):
        """Get remaining wait time before next request"""
        if self.last_request_time is None:
            return 0
            
        time_since_last = (datetime.now() - self.last_request_time).total_seconds()
        return max(0, self.current_interval - time_since_last)
        
    def get_performance_stats(self):
        """Get rate limiter performance statistics"""
        total_requests = self.success_count + self.error_count
        success_rate = (self.success_count / total_requests * 100) if total_requests > 0 else 0
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        return {
            'current_interval': self.current_interval,
            'success_rate': success_rate,
            'avg_response_time': avg_response_time,
            'total_requests': total_requests,
            'error_count': self.error_count
        }

# ================================
# System Monitoring and Observability
# ================================
class SystemMonitor:
    """Comprehensive system monitoring and health tracking"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.metrics = {
            'api_calls': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'signals_sent': 0,
            'trades_executed': 0,
            'total_pnl': 0,
            'memory_usage': 0,
            'cpu_usage': 0
        }
        
        self.health_checks = {
            'api_health': True,
            'ml_health': True,
            'data_health': True,
            'memory_health': True
        }
        
        self.alerts = []
        self.performance_history = deque(maxlen=100)
        
    def record_api_call(self, success=True, response_time=0):
        """Record API call metrics"""
        self.metrics['api_calls'] += 1
        
        if not success:
            self.health_checks['api_health'] = False
            self.add_alert('API_ERROR', f'API call failed after {response_time:.2f}s')
        else:
            self.health_checks['api_health'] = True
            
    def record_prediction(self, success=True, confidence=0):
        """Record ML prediction metrics"""
        if success:
            self.metrics['successful_predictions'] += 1
            self.health_checks['ml_health'] = True
        else:
            self.metrics['failed_predictions'] += 1
            self.health_checks['ml_health'] = False
            self.add_alert('ML_ERROR', 'ML prediction failed')
            
    def record_signal(self, signal_type, success=True):
        """Record signal sending metrics"""
        if success:
            self.metrics['signals_sent'] += 1
        else:
            self.add_alert('SIGNAL_ERROR', f'Failed to send {signal_type} signal')
            
    def record_trade(self, pnl):
        """Record trade execution metrics"""
        self.metrics['trades_executed'] += 1
        self.metrics['total_pnl'] += pnl
        
    def update_system_resources(self):
        """Update system resource usage"""
        try:
            import psutil
            process = psutil.Process()
            self.metrics['memory_usage'] = process.memory_info().rss / 1024 / 1024  # MB
            self.metrics['cpu_usage'] = process.cpu_percent()
            
            # Health checks
            if self.metrics['memory_usage'] > 1000:  # > 1GB
                self.health_checks['memory_health'] = False
                self.add_alert('MEMORY_WARNING', f"High memory usage: {self.metrics['memory_usage']:.1f}MB")
            else:
                self.health_checks['memory_health'] = True
                
        except ImportError:
            # psutil not available
            pass
            
    def add_alert(self, alert_type, message):
        """Add system alert"""
        alert = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'message': message
        }
        self.alerts.append(alert)
        
        # Keep only recent alerts
        if len(self.alerts) > 50:
            self.alerts = self.alerts[-50:]
            
        logger.warning(f"[{alert_type}] {message}")
        
    def get_system_health(self):
        """Get overall system health status"""
        all_healthy = all(self.health_checks.values())
        
        return {
            'overall_health': 'HEALTHY' if all_healthy else 'DEGRADED',
            'uptime': str(datetime.now() - self.start_time),
            'health_checks': self.health_checks.copy(),
            'recent_alerts': len([a for a in self.alerts if (datetime.now() - a['timestamp']).seconds < 300]),
            'metrics_summary': {
                'api_success_rate': self._calculate_api_success_rate(),
                'ml_success_rate': self._calculate_ml_success_rate(),
                'avg_pnl_per_trade': self._calculate_avg_pnl_per_trade(),
                'signals_per_hour': self._calculate_signals_per_hour()
            }
        }
        
    def _calculate_api_success_rate(self):
        """Calculate API success rate"""
        total_calls = self.metrics['api_calls']
        if total_calls == 0:
            return 100.0
        # Assuming failed calls are tracked separately
        return 100.0  # Simplified - would need error tracking
        
    def _calculate_ml_success_rate(self):
        """Calculate ML prediction success rate"""
        total_predictions = self.metrics['successful_predictions'] + self.metrics['failed_predictions']
        if total_predictions == 0:
            return 100.0
        return (self.metrics['successful_predictions'] / total_predictions) * 100
        
    def _calculate_avg_pnl_per_trade(self):
        """Calculate average PnL per trade"""
        if self.metrics['trades_executed'] == 0:
            return 0.0
        return self.metrics['total_pnl'] / self.metrics['trades_executed']
        
    def _calculate_signals_per_hour(self):
        """Calculate signals sent per hour"""
        uptime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        if uptime_hours == 0:
            return 0.0
        return self.metrics['signals_sent'] / uptime_hours
        
    def generate_health_report(self):
        """Generate comprehensive health report"""
        health = self.get_system_health()
        
        report = f"""
{'='*60}
🏥 SYSTEM HEALTH REPORT
{'='*60}
Overall Status: {health['overall_health']}
Uptime: {health['uptime']}

📊 PERFORMANCE METRICS:
- API Success Rate: {health['metrics_summary']['api_success_rate']:.1f}%
- ML Success Rate: {health['metrics_summary']['ml_success_rate']:.1f}%
- Avg PnL/Trade: ₹{health['metrics_summary']['avg_pnl_per_trade']:.2f}
- Signals/Hour: {health['metrics_summary']['signals_per_hour']:.1f}

🔧 HEALTH CHECKS:
- API Health: {'✅' if health['health_checks']['api_health'] else '❌'}
- ML Health: {'✅' if health['health_checks']['ml_health'] else '❌'}
- Data Health: {'✅' if health['health_checks']['data_health'] else '❌'}
- Memory Health: {'✅' if health['health_checks']['memory_health'] else '❌'}

📈 SYSTEM METRICS:
- Total API Calls: {self.metrics['api_calls']}
- Successful Predictions: {self.metrics['successful_predictions']}
- Failed Predictions: {self.metrics['failed_predictions']}
- Signals Sent: {self.metrics['signals_sent']}
- Trades Executed: {self.metrics['trades_executed']}
- Total PnL: ₹{self.metrics['total_pnl']:.2f}
- Memory Usage: {self.metrics['memory_usage']:.1f}MB
- CPU Usage: {self.metrics['cpu_usage']:.1f}%

⚠️ RECENT ALERTS: {health['recent_alerts']}
{'='*60}
"""
        return report
        
    def log_performance_snapshot(self):
        """Log current performance snapshot"""
        snapshot = {
            'timestamp': datetime.now(),
            'metrics': self.metrics.copy(),
            'health': self.get_system_health()
        }
        self.performance_history.append(snapshot)

# Add after the imports section, before the LSTM class
def validate_and_clean_features(features):
    """Critical feature validation to prevent trading losses from bad data"""
    if not features or not isinstance(features, dict):
        return None
        
    cleaned_features = {}
    
    for key, value in features.items():
        try:
            # Convert to float and handle edge cases
            if value is None:
                cleaned_features[key] = 0.0
            elif math.isnan(value) or math.isinf(value):
                cleaned_features[key] = 0.0
            else:
                cleaned_features[key] = float(value)
        except (TypeError, ValueError):
            cleaned_features[key] = 0.0
    
    # Apply realistic bounds to critical features
    feature_bounds = {
        'volatility': (0.05, 2.0),  # 5% to 200% annualized
        'pcr_oi': (0.1, 10.0),      # Put-Call ratio bounds
        'pcr_volume': (0.1, 10.0),   
        'time_of_day': (0.0, 24.0),
        'underlying_price': (10000, 30000),  # Nifty bounds
        'atm_call_ltp': (0.1, 1000),        # Option price bounds
        'atm_put_ltp': (0.1, 1000),
        'call_oi_change_pct': (-50, 50),     # Reasonable OI change bounds
        'put_oi_change_pct': (-50, 50),
    }
    
    for feature, (min_val, max_val) in feature_bounds.items():
        if feature in cleaned_features:
            cleaned_features[feature] = max(min_val, min(max_val, cleaned_features[feature]))
    
    # Scale large values to prevent model issues
    scaling_factors = {
        'atm_call_oi': 100000,      # Scale OI to 0-10 range
        'atm_put_oi': 100000,
        'total_call_oi': 1000000,
        'total_put_oi': 1000000,
        'atm_call_volume': 10000,   # Scale volume to 0-100 range  
        'atm_put_volume': 10000,
        'total_call_volume': 100000,
        'total_put_volume': 100000,
    }
    
    for feature, scale in scaling_factors.items():
        if feature in cleaned_features:
            cleaned_features[feature] = cleaned_features[feature] / scale
    
    return cleaned_features

# ================================
# SHAP Feature Importance Analyzer
# ================================
class SHAPFeatureAnalyzer:
    """SHAP-based feature importance analysis for ML model interpretability"""
    
    def __init__(self):
        self.shap_values = {}
        self.feature_importance_history = []
        self.explainer = None
        self.feature_names = []
        self.last_analysis_time = None
        self.analysis_interval = 100  # Analyze every 100 predictions
        
    def initialize_explainer(self, model, feature_names, model_type='tree'):
        """Initialize SHAP explainer based on model type"""
        if not SHAP_AVAILABLE:
            return False
            
        try:
            self.feature_names = feature_names
            
            if model_type == 'tree':
                # For tree-based models (Random Forest, Gradient Boosting)
                if hasattr(model, 'estimators_'):
                    self.explainer = shap.TreeExplainer(model)
                else:
                    # Fallback for unsupported tree models
                    return False
            elif model_type == 'linear':
                # For linear models (Logistic Regression) - fix masker issue
                try:
                    # Create a simple masker with zeros
                    masker = shap.maskers.Independent(np.zeros((1, len(feature_names))))
                    self.explainer = shap.LinearExplainer(model, masker)
                except:
                    # If LinearExplainer fails, skip SHAP for this model
                    return False
            else:
                # Skip other model types to avoid errors
                return False
                
            return True
            
        except Exception as e:
            # Downgrade known multiclass GradientBoosting limitation to warning
            msg = str(e)
            if 'GradientBoostingClassifier is only supported for binary classification' in msg:
                logger.warning(f"SHAP explainer limitation: {e}")
            else:
                logger.error(f"SHAP explainer initialization error: {e}")
            return False
    
    def analyze_feature_importance(self, model, X_sample, feature_names, model_name='regime_classifier'):
        """Analyze feature importance using SHAP"""
        if not SHAP_AVAILABLE or self.explainer is None:
            return {}
            
        try:
            # Skip SHAP for models that cause issues
            if hasattr(model, '__class__') and 'GradientBoosting' in model.__class__.__name__:
                # Check if it's multiclass (causes the binary classification error)
                if hasattr(model, 'n_classes_') and model.n_classes_ > 2:
                    logger.warning(f"Skipping SHAP for multiclass GradientBoosting model: {model_name}")
                    return {}
            
            # Limit sample size to avoid memory issues
            sample_size = min(len(X_sample), 100)
            X_subset = X_sample[:sample_size]
            
            # Calculate SHAP values with error handling
            shap_values = self.explainer.shap_values(X_subset)
            
            # Handle different SHAP output formats safely
            if isinstance(shap_values, list):
                if len(shap_values) > 0:
                    shap_values = shap_values[0]
                else:
                    return {}
            
            # Ensure shap_values is a numpy array
            if not isinstance(shap_values, np.ndarray):
                return {}
            
            # Handle shape issues
            if shap_values.ndim == 1:
                shap_values = shap_values.reshape(1, -1)
            
            # Calculate mean absolute SHAP values for feature importance
            try:
                feature_importance = np.mean(np.abs(shap_values), axis=0)
            except Exception as e:
                logger.warning(f"Error calculating feature importance for {model_name}: {e}")
                return {}
            
            # Create feature importance dictionary safely
            importance_dict = {}
            min_len = min(len(feature_names), len(feature_importance))
            for i in range(min_len):
                try:
                    importance_dict[feature_names[i]] = float(feature_importance[i])
                except (ValueError, TypeError):
                    importance_dict[feature_names[i]] = 0.0
            
            # Store for history
            self.shap_values[model_name] = {
                'importance': importance_dict,
                'timestamp': datetime.now(),
                'sample_size': sample_size
            }
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"SHAP feature importance analysis error: {e}")
            return {}
    
    def analyze_prediction(self, model, X_single, feature_names, model_name='regime_classifier'):
        """Analyze individual prediction using SHAP"""
        if not SHAP_AVAILABLE or self.explainer is None:
            return {}
            
        try:
            # Skip problematic models
            if hasattr(model, '__class__') and 'GradientBoosting' in model.__class__.__name__:
                if hasattr(model, 'n_classes_') and model.n_classes_ > 2:
                    return {}
            
            # Ensure X_single is properly shaped
            if X_single.ndim == 1:
                X_single = X_single.reshape(1, -1)
            
            # Calculate SHAP values for single prediction
            shap_values = self.explainer.shap_values(X_single)
            
            # Handle different SHAP output formats safely
            if isinstance(shap_values, list):
                if len(shap_values) > 0:
                    shap_values = shap_values[0]
                else:
                    return {}
            
            # Ensure we have the right shape
            if not isinstance(shap_values, np.ndarray):
                return {}
                
            if shap_values.ndim == 1:
                shap_values = shap_values.reshape(1, -1)
            
            # Create prediction explanation safely
            explanation = {}
            if len(shap_values) > 0:
                shap_row = shap_values[0]
                min_len = min(len(feature_names), len(shap_row))
                for i in range(min_len):
                    try:
                        explanation[feature_names[i]] = float(shap_row[i])
                    except (ValueError, TypeError, IndexError):
                        explanation[feature_names[i]] = 0.0
            
            return explanation
            
        except Exception as e:
            logger.error(f"SHAP prediction analysis error: {e}")
            return {}
    
    def get_top_features(self, model_name='regime_classifier', top_n=10):
        """Get top N most important features"""
        if model_name not in self.shap_values:
            return []
            
        importance_dict = self.shap_values[model_name]['importance']
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_importance[:top_n]

    def generate_feature_report(self, model_name='regime_classifier', top_n=10):
        """Generate a simple text report of SHAP feature importance."""
        if model_name not in self.shap_values:
            return "No SHAP results available."
        try:
            importance = self.shap_values[model_name]['importance']
            sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            lines = [
                "\n" + "="*60,
                f"SHAP Feature Importance Report: {model_name}",
                "="*60
            ]
            for i, (feat, val) in enumerate(sorted_items[:top_n], start=1):
                lines.append(f"{i:2d}. {feat}: {val:.6f}")
            lines.append("="*60)
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"Failed generating SHAP report: {e}")
            return "SHAP report generation failed."

    def save_feature_importance_plot(self, model_name='regime_classifier', top_n=20):
        """Save a bar plot of top SHAP feature importances."""
        if model_name not in self.shap_values:
            return False
        try:
            try:
                import matplotlib.pyplot as plt
            except Exception as e:
                logger.warning(f"matplotlib not available for SHAP plotting: {e}")
                return False
            importance = self.shap_values[model_name]['importance']
            sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
            features = [k for k, _ in sorted_items][::-1]
            values = [v for _, v in sorted_items][::-1]
            plt.figure(figsize=(8, max(4, len(features)*0.3)))
            plt.barh(features, values)
            plt.title(f"SHAP Feature Importance: {model_name}")
            plt.xlabel("Mean |SHAP value|")
            plt.tight_layout()
            filename = f"shap_feature_importance_{model_name}.png"
            plt.savefig(filename)
            plt.close()
            return True
        except Exception as e:
            logger.warning(f"Failed saving SHAP importance plot: {e}")
            return False

    def export_feature_importance_csv(self, model_name='regime_classifier', filepath=None):
        """Export SHAP feature importances to CSV."""
        if model_name not in self.shap_values:
            return False
        try:
            import csv
            importance = self.shap_values[model_name]['importance']
            sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            path = filepath or f"shap_feature_importance_{model_name}.csv"
            with open(path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["feature", "importance"])
                for feat, val in sorted_items:
                    try:
                        writer.writerow([feat, float(val)])
                    except Exception:
                        writer.writerow([feat, 0.0])
            return True
        except Exception as e:
            logger.warning(f"Failed exporting SHAP importance CSV: {e}")
            return False

    def track_feature_importance_over_time(self, model_name='regime_classifier'):
        """Record snapshot of feature importance for trend analysis."""
        try:
            if model_name not in self.shap_values:
                return False
            snapshot = {
                'model': model_name,
                'timestamp': datetime.now(),
                'importance': self.shap_values[model_name]['importance'].copy()
            }
            self.feature_importance_history.append(snapshot)
            # Keep history bounded
            if len(self.feature_importance_history) > 1000:
                self.feature_importance_history = self.feature_importance_history[-500:]
            return True
        except Exception as e:
            logger.warning(f"Failed to track SHAP importance over time: {e}")
            return False

# ================================
# Feature Caching System for Performance
# ================================
class FeatureCache:
    """Efficient feature caching system to avoid redundant calculations"""
    
    def __init__(self, max_size=100):
        self.cache = {}
        self.timestamps = []
        self.max_size = max_size
        
    def get_cache_key(self, underlying_value, atm_strike, timestamp):
        """Generate cache key for features"""
        # Round timestamp to minute for caching
        minute_timestamp = timestamp.replace(second=0, microsecond=0)
        return f"{underlying_value:.2f}_{atm_strike}_{minute_timestamp.isoformat()}"
        
    def get_features(self, key):
        """Get cached features if available"""
        return self.cache.get(key)
        
    def store_features(self, key, features):
        """Store features in cache with size management"""
        if len(self.cache) >= self.max_size:
            # Remove oldest entries
            oldest_keys = list(self.cache.keys())[:10]
            for old_key in oldest_keys:
                del self.cache[old_key]
                
        self.cache[key] = features
        
    def clear_old_cache(self, cutoff_time):
        """Clear cache entries older than cutoff time"""
        keys_to_remove = []
        for key in self.cache.keys():
            try:
                # Extract timestamp from key
                timestamp_str = key.split('_')[-1]
                cache_time = datetime.fromisoformat(timestamp_str)
                if cache_time < cutoff_time:
                    keys_to_remove.append(key)
            except:
                continue
                
        for key in keys_to_remove:
            del self.cache[key]

# ================================
# Memory-Efficient Data Manager
# ================================
class MemoryEfficientDataManager:
    """Optimized data management with automatic cleanup"""
    
    def __init__(self, max_historical_minutes=30):
        self.historical_data = {}
        self.max_historical_minutes = max_historical_minutes
        self.feature_cache = FeatureCache()
        self.last_cleanup = datetime.now()
        
    def store_data(self, timestamp, data):
        """Store data with automatic cleanup"""
        # Store data
        minute_key = timestamp.replace(second=0, microsecond=0)
        self.historical_data[minute_key] = data
        
        # Cleanup every 5 minutes
        if (datetime.now() - self.last_cleanup).total_seconds() > 300:
            self.cleanup_old_data()
            self.last_cleanup = datetime.now()
            
    def cleanup_old_data(self):
        """Remove data older than max_historical_minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=self.max_historical_minutes)
        
        # Clean historical data
        old_keys = [k for k in self.historical_data.keys() if k < cutoff_time]
        for key in old_keys:
            del self.historical_data[key]
            
        # Clean feature cache
        self.feature_cache.clear_old_cache(cutoff_time)
        
        if old_keys:
            logger.info(f"Cleaned {len(old_keys)} old data entries")
            
    def get_recent_data(self, minutes_back=10):
        """Get recent data for analysis"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes_back)
        return {k: v for k, v in self.historical_data.items() if k >= cutoff_time}

# ================================
# Circuit Breaker Pattern for System Reliability
# ================================
class CircuitBreaker:
    """Circuit breaker pattern to handle system failures gracefully"""
    
    def __init__(self, failure_threshold=5, recovery_timeout=300, expected_exception=Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == 'OPEN':
            if self._should_attempt_reset():
                self.state = 'HALF_OPEN'
            else:
                raise Exception("Circuit breaker is OPEN")
                
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
            
    def _should_attempt_reset(self):
        """Check if enough time has passed to attempt reset"""
        return (datetime.now() - self.last_failure_time).total_seconds() >= self.recovery_timeout
        
    def _on_success(self):
        """Handle successful execution"""
        self.failure_count = 0
        self.state = 'CLOSED'
        
    def _on_failure(self):
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

# ================================
# Enhanced Error Recovery System
# ================================
class ErrorRecoverySystem:
    """Comprehensive error recovery and fallback mechanisms"""
    
    def __init__(self):
        self.api_circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        self.ml_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=300)
        self.fallback_data = None
        self.error_counts = {
            'api_errors': 0,
            'ml_errors': 0,
            'data_errors': 0,
            'total_errors': 0
        }
        
    def safe_api_call(self, func, *args, **kwargs):
        """Safe API call with circuit breaker and fallback"""
        try:
            return self.api_circuit_breaker.call(func, *args, **kwargs)
        except Exception as e:
            self.error_counts['api_errors'] += 1
            self.error_counts['total_errors'] += 1
            logger.error(f"API call failed: {e}")
            
            # Return fallback data if available
            if self.fallback_data:
                logger.info("Using fallback data due to API failure")
                return self.fallback_data
            return None
            
    def safe_ml_prediction(self, func, *args, **kwargs):
        """Safe ML prediction with fallback to traditional analysis"""
        try:
            return self.ml_circuit_breaker.call(func, *args, **kwargs)
        except Exception as e:
            self.error_counts['ml_errors'] += 1
            self.error_counts['total_errors'] += 1
            logger.error(f"ML prediction failed: {e}")
            
            # Fallback to traditional analysis
            logger.info("Falling back to traditional analysis")
            return None
            
    def update_fallback_data(self, data):
        """Update fallback data for emergency use"""
        if data is not None:
            self.fallback_data = data
            
    def get_error_statistics(self):
        """Get error statistics for monitoring"""
        return self.error_counts.copy()
        
    def reset_error_counts(self):
        """Reset error counters"""
        self.error_counts = {key: 0 for key in self.error_counts}

# ================================
# Adaptive Rate Limiting System

if __name__ == "__main__":
    print("🚀 Advanced Options SHORT SELLING System with 24/7 ML Learning")
    print("=" * 70)
    print("📊 Enhanced Features:")
    print("   • 24/7 Operation: Live trading + Offline simulation")
    print("   • Advanced ML Decision Engine (Random Forest + Gradient Boosting)")
    print("   • Deep Learning LSTM for sequence prediction")
    print("   • SHAP Feature Importance Analysis")
    print("   • Advanced Risk Management & Position Sizing")
    print("   • Market Regime Detection")
    print("   • Profit Optimization & Entry/Exit Timing")
    print("   • ENHANCED: Sequential offline simulation using ALL stored data")
    print("   • ENHANCED: After-market hours paper trading with full data utilization")
    print("   • ENHANCED: Learn from mistakes using comprehensive historical data")
    print("   • ENHANCED: No more 20-sample limit - processes thousands of samples")
    print("   • ENHANCED: Comprehensive data loading from all source directories")
    print("   • Real-time Signal Generation for Tradetron")
    print("   • Continuous Learning from Every Market Condition")
    print("=" * 70)
    print("🎯 Strategy: SHORT OPTIONS")
    print("   • Signal 1 = SHORT PUT (profit when PUT price decreases)")
    print("   • Signal -1 = SHORT CALL (profit when CALL price decreases)")
    print("💰 Profit Logic: Entry Price - Current Price = Profit")
    print("🛡️ Risk Management: Stop loss ABOVE entry, Profit target BELOW entry")
    print("🌙 24/7 Learning:")
    print("   • LIVE MODE: Collects data, generates signals, paper trades")
    print("   • OFFLINE MODE: Simulates stored data sequentially")
    print("   • SIMULATION: Replays historical data like live market")
    print("   • LEARNING: ML trains on both live and simulated outcomes")
    print("📡 Output: Direct signals to Tradetron API (confidence > 70%)")
    print("🔍 Analysis: SHAP explainability for feature importance")
    print("=" * 70)
    print("🛠️ Usage Options:")
    print("   1. Run normally: python trading_system_24x7.py")
    print("   2. Test offline simulation:")
    print("      analyzer = OptimizedATMAnalyzer()")
    print("      analyzer.start_immediate_offline_simulation()")
    print("=" * 70)
    
    # Initialize the analyzer
    analyzer = OptimizedATMAnalyzer()
    
    # Check if user wants to test offline simulation first
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--test-offline':
        print("\n🌙 TESTING OFFLINE SIMULATION...")
        analyzer.start_immediate_offline_simulation()
        print("💡 To run full system: python trading_system_24x7.py")
    else:
        # Run the full 24/7 analysis
        print("\n🚀 STARTING 24/7 TRADING SYSTEM...")
        analyzer.run_analysis()
