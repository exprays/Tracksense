"""
Comprehensive ML Model Training Pipeline
Trains tire degradation, pit strategy, and driver fingerprinting models
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import logging
import json
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Try to import XGBoost, use RandomForest as fallback
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except (ImportError, ValueError) as e:
    HAS_XGBOOST = False
    xgb = None

from .tire_model import TireDegradationModel
from .pit_optimizer import PitStopOptimizer
from ..data.loader import RaceDataLoader
from ..data.preprocessor import RaceDataPreprocessor
from ..utils.constants import TRACKS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and evaluate all race strategy models"""
    
    def __init__(self, data_path: str, models_output_path: str):
        """
        Initialize model trainer
        
        Args:
            data_path: Path to race datasets
            models_output_path: Path to save trained models
        """
        self.data_path = Path(data_path)
        self.models_path = Path(models_output_path)
        self.models_path.mkdir(exist_ok=True, parents=True)
        
        self.loader = RaceDataLoader(str(self.data_path))
        self.preprocessor = RaceDataPreprocessor()
        
        self.tire_model = TireDegradationModel()
        self.pit_strategy_model = None
        self.driver_fingerprint_model = None
        
        self.training_history = {
            'timestamp': None,
            'tire_model': {},
            'pit_strategy_model': {},
            'driver_model': {},
            'races_used': []
        }
    
    def load_all_race_data(self) -> List[Tuple[str, int, pd.DataFrame]]:
        """
        Load all available race data from all tracks
        
        Returns:
            List of tuples (track, race_num, processed_data)
        """
        all_data = []
        
        for track in ['barber', 'cota', 'indianapolis', 'sebring']:
            for race_num in [1, 2]:
                try:
                    logger.info(f"Loading {track} Race {race_num}...")
                    
                    # Get all drivers for this race
                    drivers = self.loader.get_available_drivers(track, race_num)
                    
                    if not drivers:
                        logger.warning(f"No drivers found for {track} R{race_num}")
                        continue
                    
                    # Load data for all drivers
                    for driver in drivers:
                        driver_data = self.loader.get_driver_data(track, race_num, driver)
                        
                        if driver_data.get('sectors') is not None and not driver_data['sectors'].empty:
                            processed = self.preprocessor.process_driver_data(driver_data)
                            
                            if not processed.empty:
                                # Add metadata
                                processed['TRACK'] = track
                                processed['RACE'] = race_num
                                processed['DRIVER'] = driver
                                
                                all_data.append((track, race_num, driver, processed))
                                logger.info(f"Loaded {len(processed)} laps for driver {driver}")
                    
                except Exception as e:
                    logger.error(f"Error loading {track} R{race_num}: {e}")
                    continue
        
        logger.info(f"Total datasets loaded: {len(all_data)}")
        return all_data
    
    def train_tire_degradation_model(self, race_data: List[Tuple]) -> Dict:
        """
        Train tire degradation prediction model
        
        Args:
            race_data: List of (track, race, driver, dataframe) tuples
        
        Returns:
            Training metrics and results
        """
        logger.info("=" * 60)
        logger.info("TRAINING TIRE DEGRADATION MODEL")
        logger.info("=" * 60)
        
        # Extract dataframes
        all_dfs = [data[3] for data in race_data]
        
        # Train the tire model
        metrics = self.tire_model.train(all_dfs)
        
        # Save model
        model_path = self.models_path / 'tire_degradation_model.pkl'
        self.tire_model.save(str(model_path))
        
        # Get feature importance
        feature_importance = self.tire_model.get_feature_importance()
        
        results = {
            'metrics': metrics,
            'feature_importance': feature_importance,
            'model_path': str(model_path),
            'training_races': len(race_data),
            'total_laps': sum(len(df) for df in all_dfs)
        }
        
        self.training_history['tire_model'] = results
        
        logger.info(f"✓ Tire model trained on {results['total_laps']} laps")
        logger.info(f"  R² Score: {metrics['r2_score']:.4f}")
        logger.info(f"  MAE: {metrics['mae']:.4f}")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        
        return results
    
    def train_pit_strategy_model(self, race_data: List[Tuple]) -> Dict:
        """
        Train pit stop timing prediction model
        Uses XGBoost to predict optimal pit lap
        
        Args:
            race_data: List of (track, race, driver, dataframe) tuples
        
        Returns:
            Training metrics and results
        """
        logger.info("=" * 60)
        logger.info("TRAINING PIT STRATEGY MODEL")
        logger.info("=" * 60)
        
        # Prepare training data
        X_list = []
        y_list = []
        
        for track, race, driver, df in race_data:
            if len(df) < 10:  # Need sufficient laps
                continue
            
            # Features for pit strategy
            features = []
            
            # Current race state
            features.append(df['LAP_NUMBER'].values)
            features.append(df['TIRE_LIFE_ESTIMATE'].values)
            features.append(df['LAPS_OF_FUEL'].values)
            features.append(df['DEGRADATION_RATE'].values)
            features.append(df['CONSISTENCY_SCORE'].values)
            
            # Track-specific
            features.append(np.full(len(df), 1 if track == 'barber' else 0))
            
            # Weather
            if 'TRACK_TEMP' in df.columns:
                features.append(df['TRACK_TEMP'].fillna(df['TRACK_TEMP'].mean()).values)
            
            if 'AIR_TEMP' in df.columns:
                features.append(df['AIR_TEMP'].fillna(df['AIR_TEMP'].mean()).values)
            
            X = np.column_stack(features)
            
            # Target: binary classification of whether to pit
            # Pit if tire life < 0.7 OR fuel < 5 laps OR in optimal window (laps 8-12)
            should_pit = (
                (df['TIRE_LIFE_ESTIMATE'] < 0.7) | 
                (df['LAPS_OF_FUEL'] < 5) |
                df['IN_OPTIMAL_PIT_WINDOW']
            ).astype(int).values
            
            X_list.append(X)
            y_list.append(should_pit)
        
        if not X_list:
            logger.warning("No data available for pit strategy model")
            return {}
        
        X_train = np.vstack(X_list)
        y_train = np.concatenate(y_list)
        
        # Remove NaN values
        valid_mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]
        
        # Split for validation
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Train classifier (XGBoost if available, RandomForest as fallback)
        if HAS_XGBOOST:
            logger.info("Training with XGBoost classifier")
            self.pit_strategy_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
            
            self.pit_strategy_model.fit(
                X_train_split, y_train_split,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            logger.info("Using RandomForest classifier (XGBoost not available)")
            self.pit_strategy_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
            self.pit_strategy_model.fit(X_train_split, y_train_split)
        
        # Evaluate
        train_accuracy = self.pit_strategy_model.score(X_train_split, y_train_split)
        val_accuracy = self.pit_strategy_model.score(X_val, y_val)
        
        # Predictions for detailed metrics
        y_pred = self.pit_strategy_model.predict(X_val)
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        # Save model
        model_path = self.models_path / 'pit_strategy_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.pit_strategy_model, f)
        
        results = {
            'train_accuracy': float(train_accuracy),
            'val_accuracy': float(val_accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'n_samples': len(X_train),
            'model_path': str(model_path)
        }
        
        self.training_history['pit_strategy_model'] = results
        
        logger.info(f"✓ Pit strategy model trained on {len(X_train)} samples")
        logger.info(f"  Training Accuracy: {train_accuracy:.4f}")
        logger.info(f"  Validation Accuracy: {val_accuracy:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        
        return results
    
    def train_driver_fingerprint_model(self, race_data: List[Tuple]) -> Dict:
        """
        Train driver fingerprinting/style classification model
        Learns unique driving characteristics per driver
        
        Args:
            race_data: List of (track, race, driver, dataframe) tuples
        
        Returns:
            Training metrics and results
        """
        logger.info("=" * 60)
        logger.info("TRAINING DRIVER FINGERPRINT MODEL")
        logger.info("=" * 60)
        
        # Group by driver
        driver_profiles = {}
        
        for track, race, driver, df in race_data:
            if len(df) < 5:  # Need minimum laps
                continue
            
            if driver not in driver_profiles:
                driver_profiles[driver] = []
            
            # Extract driver characteristics
            profile = {
                'avg_lap_time': df['LAP_TIME_SECONDS'].mean(),
                'consistency': df['CONSISTENCY_SCORE'].mean(),
                'aggression': df['DEGRADATION_RATE'].mean(),  # Higher = more aggressive
                'sector1_strength': df['S1_SECONDS'].mean() if 'S1_SECONDS' in df.columns else 0,
                'sector2_strength': df['S2_SECONDS'].mean() if 'S2_SECONDS' in df.columns else 0,
                'sector3_strength': df['S3_SECONDS'].mean() if 'S3_SECONDS' in df.columns else 0,
                'top_speed': df['TOP_SPEED'].max() if 'TOP_SPEED' in df.columns else 0,
                'avg_speed': df['KPH'].mean() if 'KPH' in df.columns else 0,
            }
            
            driver_profiles[driver].append(profile)
        
        # Build training data
        X_list = []
        y_list = []
        driver_names = []
        
        for driver, profiles in driver_profiles.items():
            if len(profiles) < 2:  # Need multiple races
                continue
            
            for profile in profiles:
                features = [
                    profile['avg_lap_time'],
                    profile['consistency'],
                    profile['aggression'],
                    profile['sector1_strength'],
                    profile['sector2_strength'],
                    profile['sector3_strength'],
                    profile['top_speed'],
                    profile['avg_speed']
                ]
                
                X_list.append(features)
                y_list.append(driver)
                driver_names.append(driver)
        
        if len(X_list) < 10:
            logger.warning("Insufficient data for driver fingerprinting")
            return {}
        
        X_train = np.array(X_list)
        
        # Encode driver labels
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_list)
        
        # Remove NaN
        valid_mask = ~np.isnan(X_train).any(axis=1)
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train Random Forest for driver classification
        self.driver_fingerprint_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.driver_fingerprint_model.fit(X_train_scaled, y_train)
        
        # Cross-validation score (use 2-fold minimum to avoid errors with small datasets)
        n_splits = min(3, len(set(y_train)), len(X_train) // 2)
        if n_splits < 2:
            # Not enough data for cross-validation, use train score only
            cv_scores = np.array([self.driver_fingerprint_model.score(X_train_scaled, y_train)])
            logger.warning(f"Insufficient data for cross-validation, using train score")
        else:
            cv_scores = cross_val_score(
                self.driver_fingerprint_model, 
                X_train_scaled, 
                y_train, 
                cv=n_splits
            )
        
        # Feature importance
        feature_names = ['avg_lap_time', 'consistency', 'aggression', 
                        'sector1', 'sector2', 'sector3', 'top_speed', 'avg_speed']
        feature_importance = dict(zip(
            feature_names, 
            self.driver_fingerprint_model.feature_importances_
        ))
        
        # Save model
        model_path = self.models_path / 'driver_fingerprint_model.pkl'
        model_data = {
            'model': self.driver_fingerprint_model,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'feature_names': feature_names
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        results = {
            'cv_accuracy_mean': float(cv_scores.mean()),
            'cv_accuracy_std': float(cv_scores.std()),
            'n_drivers': len(set(y_list)),
            'n_samples': len(X_train),
            'feature_importance': feature_importance,
            'model_path': str(model_path),
            'drivers': list(label_encoder.classes_)
        }
        
        self.training_history['driver_model'] = results
        
        logger.info(f"✓ Driver fingerprint model trained on {len(X_train)} samples")
        logger.info(f"  Drivers: {len(set(y_list))}")
        logger.info(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return results
    
    def train_all_models(self) -> Dict:
        """
        Train all models and save results
        
        Returns:
            Complete training results
        """
        logger.info("=" * 60)
        logger.info("STARTING COMPREHENSIVE MODEL TRAINING")
        logger.info("=" * 60)
        
        self.training_history['timestamp'] = datetime.now().isoformat()
        
        # Load all race data
        race_data = self.load_all_race_data()
        
        if not race_data:
            logger.error("No race data available for training!")
            return {}
        
        self.training_history['races_used'] = [
            f"{track}_R{race}_D{driver}" for track, race, driver, _ in race_data
        ]
        
        # Train each model
        tire_results = self.train_tire_degradation_model(race_data)
        pit_results = self.train_pit_strategy_model(race_data)
        driver_results = self.train_driver_fingerprint_model(race_data)
        
        # Save training history
        history_path = self.models_path / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
        
        logger.info("=" * 60)
        logger.info("✓ ALL MODELS TRAINED SUCCESSFULLY")
        logger.info("=" * 60)
        
        return self.training_history
    
    def generate_training_report(self) -> str:
        """
        Generate a comprehensive training report
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("RACE STRATEGY ML MODELS - TRAINING REPORT")
        report.append("=" * 80)
        report.append(f"Training Date: {self.training_history.get('timestamp', 'Unknown')}")
        report.append(f"Races Used: {len(self.training_history.get('races_used', []))}")
        report.append("")
        
        # Tire Model
        tire = self.training_history.get('tire_model', {})
        if tire:
            report.append("1. TIRE DEGRADATION MODEL")
            report.append("-" * 40)
            report.append(f"   R² Score:      {tire.get('metrics', {}).get('r2_score', 0):.4f}")
            report.append(f"   MAE:           {tire.get('metrics', {}).get('mae', 0):.4f}")
            report.append(f"   RMSE:          {tire.get('metrics', {}).get('rmse', 0):.4f}")
            report.append(f"   Total Laps:    {tire.get('total_laps', 0)}")
            report.append(f"   Model Path:    {tire.get('model_path', 'N/A')}")
            report.append("")
        
        # Pit Strategy Model
        pit = self.training_history.get('pit_strategy_model', {})
        if pit:
            report.append("2. PIT STRATEGY MODEL")
            report.append("-" * 40)
            report.append(f"   Train Accuracy: {pit.get('train_accuracy', 0):.4f}")
            report.append(f"   Val Accuracy:   {pit.get('val_accuracy', 0):.4f}")
            report.append(f"   Precision:      {pit.get('precision', 0):.4f}")
            report.append(f"   Recall:         {pit.get('recall', 0):.4f}")
            report.append(f"   F1 Score:       {pit.get('f1_score', 0):.4f}")
            report.append(f"   Samples:        {pit.get('n_samples', 0)}")
            report.append("")
        
        # Driver Model
        driver = self.training_history.get('driver_model', {})
        if driver:
            report.append("3. DRIVER FINGERPRINT MODEL")
            report.append("-" * 40)
            report.append(f"   CV Accuracy:    {driver.get('cv_accuracy_mean', 0):.4f} ± {driver.get('cv_accuracy_std', 0):.4f}")
            report.append(f"   Drivers:        {driver.get('n_drivers', 0)}")
            report.append(f"   Samples:        {driver.get('n_samples', 0)}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    # Set paths
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / 'dataset'
    models_path = project_root / 'models'
    
    # Initialize trainer
    trainer = ModelTrainer(str(data_path), str(models_path))
    
    # Train all models
    results = trainer.train_all_models()
    
    # Print report
    print(trainer.generate_training_report())
