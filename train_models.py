"""
Standalone script to train all ML models
Run this before using the dashboard for best performance
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.models.model_trainer import ModelTrainer


def main():
    """Train all models"""
    print("=" * 80)
    print("TOYOTA GR CUP - ML MODEL TRAINING")
    print("=" * 80)
    print()
    
    # Setup paths
    project_root = Path(__file__).parent
    data_path = project_root / 'dataset'
    models_path = project_root / 'models'
    
    print(f"Data path: {data_path}")
    print(f"Models output path: {models_path}")
    print()
    
    if not data_path.exists():
        print(f"❌ ERROR: Data path not found: {data_path}")
        return 1
    
    # Create models directory
    models_path.mkdir(exist_ok=True, parents=True)
    
    # Initialize trainer
    print("Initializing model trainer...")
    trainer = ModelTrainer(str(data_path), str(models_path))
    
    # Train all models
    print("\nStarting training process...")
    print("This will take 1-2 minutes...\n")
    
    try:
        results = trainer.train_all_models()
        
        # Print report
        print("\n")
        print(trainer.generate_training_report())
        
        print("\n✓ Training complete! Models saved to:", models_path)
        print("\nYou can now run the dashboard with: streamlit run app.py")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
