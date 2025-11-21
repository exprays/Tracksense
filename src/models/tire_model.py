"""
Tire degradation prediction model
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import pickle
import logging

from ..utils.constants import TIRE_DEGRADATION
from ..utils.helpers import calculate_degradation_rate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TireDegradationModel:
    """Predict tire degradation based on lap data"""
    
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features for tire degradation prediction
        
        Args:
            df: DataFrame with race data
        
        Returns:
            Feature array and feature names
        """
        features = []
        feature_names = []
        
        # Basic features
        if 'LAP_NUMBER' in df.columns:
            features.append(df['LAP_NUMBER'].values)
            feature_names.append('lap_number')
        
        if 'LAPS_IN_STINT' in df.columns:
            features.append(df['LAPS_IN_STINT'].values)
            feature_names.append('laps_in_stint')
        
        # Speed and pace features
        if 'KPH' in df.columns:
            features.append(df['KPH'].values)
            feature_names.append('avg_speed')
        
        if 'TOP_SPEED' in df.columns:
            features.append(df['TOP_SPEED'].values)
            feature_names.append('top_speed')
        
        # Sector times (indicate cornering load)
        for sector in ['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']:
            if sector in df.columns:
                features.append(df[sector].values)
                feature_names.append(sector.lower())
        
        # Weather conditions
        if 'AIR_TEMP' in df.columns:
            features.append(df['AIR_TEMP'].values)
            feature_names.append('air_temp')
        
        if 'TRACK_TEMP' in df.columns:
            features.append(df['TRACK_TEMP'].values)
            feature_names.append('track_temp')
        
        if 'WIND_SPEED' in df.columns:
            features.append(df['WIND_SPEED'].values)
            feature_names.append('wind_speed')
        
        # Consistency indicators
        if 'LAP_TIME_ROLLING_STD' in df.columns:
            features.append(df['LAP_TIME_ROLLING_STD'].fillna(0).values)
            feature_names.append('lap_time_std')
        
        if len(features) == 0:
            raise ValueError("No features available for model training")
        
        X = np.column_stack(features)
        
        return X, feature_names
    
    def prepare_target(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare target variable (tire life estimate)
        
        Args:
            df: DataFrame with race data
        
        Returns:
            Target array
        """
        if 'TIRE_LIFE_ESTIMATE' in df.columns:
            return df['TIRE_LIFE_ESTIMATE'].values
        
        # Calculate target from lap time degradation
        if 'LAP_TIME_SECONDS' in df.columns and 'LAP_NUMBER' in df.columns:
            lap_times = df['LAP_TIME_SECONDS'].values
            lap_numbers = df['LAP_NUMBER'].values
            
            # Estimate tire life based on lap time increase
            best_time = np.nanmin(lap_times)
            time_deltas = (lap_times - best_time) / best_time
            
            # Inverse relationship: higher delta = lower tire life
            tire_life = 1.0 - (time_deltas * 2)  # Scale factor
            tire_life = np.clip(tire_life, 0, 1)
            
            return tire_life
        
        raise ValueError("Cannot prepare target variable")
    
    def train(self, training_data: List[pd.DataFrame]) -> Dict[str, float]:
        """
        Train the tire degradation model
        
        Args:
            training_data: List of DataFrames from different races
        
        Returns:
            Training metrics
        """
        logger.info(f"Training tire degradation model on {len(training_data)} races")
        
        all_X = []
        all_y = []
        
        for df in training_data:
            if df.empty:
                continue
            
            try:
                X, feature_names = self.prepare_features(df)
                y = self.prepare_target(df)
                
                # Remove NaN values
                valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
                X_valid = X[valid_mask]
                y_valid = y[valid_mask]
                
                if len(X_valid) > 0:
                    all_X.append(X_valid)
                    all_y.append(y_valid)
                    
            except Exception as e:
                logger.warning(f"Error processing training data: {e}")
                continue
        
        if len(all_X) == 0:
            raise ValueError("No valid training data available")
        
        # Combine all data
        X_train = np.vstack(all_X)
        y_train = np.concatenate(all_y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.feature_names = feature_names
        self.is_trained = True
        
        # Calculate training metrics
        train_score = self.model.score(X_train_scaled, y_train)
        predictions = self.model.predict(X_train_scaled)
        mae = np.mean(np.abs(predictions - y_train))
        rmse = np.sqrt(np.mean((predictions - y_train) ** 2))
        
        metrics = {
            'r2_score': train_score,
            'mae': mae,
            'rmse': rmse,
            'n_samples': len(X_train),
            'n_features': len(feature_names)
        }
        
        logger.info(f"Model trained: R²={train_score:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict tire life for new data
        
        Args:
            df: DataFrame with current race data
        
        Returns:
            Predicted tire life values (0-1)
        """
        if not self.is_trained:
            logger.warning("Model not trained, using simple estimation")
            return self._simple_estimate(df)
        
        try:
            X, _ = self.prepare_features(df)
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            
            # Clip to valid range
            predictions = np.clip(predictions, 0, 1)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._simple_estimate(df)
    
    def _simple_estimate(self, df: pd.DataFrame) -> np.ndarray:
        """
        Simple tire life estimation (fallback)
        
        Args:
            df: DataFrame with race data
        
        Returns:
            Estimated tire life values
        """
        if 'LAPS_IN_STINT' in df.columns:
            laps = df['LAPS_IN_STINT'].values
        elif 'LAP_NUMBER' in df.columns:
            laps = df['LAP_NUMBER'].values
        else:
            return np.ones(len(df))
        
        tire_life = 1.0 - (laps * TIRE_DEGRADATION['base_degradation_rate'])
        return np.clip(tire_life, 0, 1)
    
    def predict_next_laps(self, current_df: pd.DataFrame, n_laps: int = 5) -> Dict:
        """
        Predict tire life for next N laps
        
        Args:
            current_df: Current race data
            n_laps: Number of laps to predict
        
        Returns:
            Dictionary with predictions
        """
        if current_df.empty:
            return {'laps': [], 'tire_life': [], 'warnings': []}
        
        current_lap = current_df['LAP_NUMBER'].iloc[-1]
        current_tire_life = self.predict(current_df.iloc[[-1]])[0]
        
        # Simple projection based on degradation rate
        if 'DEGRADATION_RATE' in current_df.columns:
            deg_rate = current_df['DEGRADATION_RATE'].iloc[-1]
        else:
            deg_rate = TIRE_DEGRADATION['base_degradation_rate']
        
        future_laps = []
        future_tire_life = []
        warnings = []
        
        for i in range(1, n_laps + 1):
            lap = current_lap + i
            # Project tire life degradation
            tire_life = current_tire_life - (deg_rate * i * 0.01)  # Convert to percentage
            tire_life = max(0, tire_life)
            
            future_laps.append(lap)
            future_tire_life.append(tire_life)
            
            if tire_life < TIRE_DEGRADATION['wear_threshold']:
                warnings.append(f"Lap {lap}: Critical tire wear ({tire_life*100:.1f}%)")
        
        return {
            'laps': future_laps,
            'tire_life': future_tire_life,
            'warnings': warnings,
            'recommended_pit_lap': future_laps[0] if future_tire_life[0] < 0.7 else None
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from trained model
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    def save(self, filepath: str):
        """Save model to disk"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {filepath}")
