"""
Inspect Pickle Model Files
View the contents of the saved model and preprocessor
"""

import joblib
import pandas as pd

print("=" * 80)
print("CABLE FAILURE MODEL INSPECTION")
print("=" * 80)

# ============================================================================
# Load the Random Forest Model
# ============================================================================
print("\n📦 Loading Random Forest Model...")
try:
    rf = joblib.load('cable_failure_rf_model.pkl')
    print("   ✓ Model loaded successfully")
    print(f"\n   Model Type: {type(rf).__name__}")
    print(f"   Number of Trees: {rf.n_estimators}")
    print(f"   Max Depth: {rf.max_depth}")
    print(f"   Number of Features: {rf.n_features_in_}")
    print(f"   Number of Classes: {rf.n_classes_}")
    print(f"   Classes: {rf.classes_}")
except FileNotFoundError:
    print("   ❌ Model file not found. Run run_model.py first.")
    exit(1)

# ============================================================================
# Load the Preprocessor
# ============================================================================
print("\n🔧 Loading Preprocessor Pipeline...")
try:
    preprocessor = joblib.load('cable_preprocessor.pkl')
    print("   ✓ Preprocessor loaded successfully")
    print(f"\n   Pipeline Type: {type(preprocessor).__name__}")
    print(f"   Number of Transformers: {len(preprocessor.transformers_)}")
    
    print("\n   Transformer Details:")
    for name, transformer, columns in preprocessor.transformers_:
        print(f"      • {name}: {len(columns)} features")
        if name == 'num':
            print(f"        Features: {columns[:5]}..." if len(columns) > 5 else f"        Features: {columns}")
        elif name == 'cat':
            print(f"        Features: {columns}")
except FileNotFoundError:
    print("   ❌ Preprocessor file not found.")
    exit(1)

# ============================================================================
# Feature Importance (if available)
# ============================================================================
print("\n🔍 Feature Importance Analysis...")
if hasattr(rf, 'feature_importances_'):
    importances = rf.feature_importances_
    
    # Get feature names
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    numeric_features = preprocessor.transformers_[0][2]
    cat_features = preprocessor.transformers_[1][2]
    ohe_feature_names = list(ohe.get_feature_names_out(cat_features))
    all_feature_names = list(numeric_features) + ohe_feature_names
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\n   Top 15 Most Important Features:")
    print("   " + "-" * 60)
    for idx, row in importance_df.head(15).iterrows():
        bar_length = int(row['Importance'] * 100)
        bar = '█' * (bar_length // 2)
        print(f"   {row['Feature']:25s} │ {bar} {row['Importance']:.4f}")
    
    # Save to CSV for easier viewing
    importance_df.to_csv('feature_importance.csv', index=False)
    print("\n   ✓ Feature importance saved to: feature_importance.csv")

# ============================================================================
# Model Parameters
# ============================================================================
print("\n⚙️  Model Parameters:")
print("   " + "-" * 60)
params = rf.get_params()
for key, value in sorted(params.items())[:10]:
    print(f"   {key:25s}: {value}")

# ============================================================================
# How to Use the Model
# ============================================================================
print("\n" + "=" * 80)
print("HOW TO USE THE MODEL")
print("=" * 80)

print("""
📝 To predict failure probability for new cables:

1. Prepare your data (same format as training data):
   
   import pandas as pd
   new_data = pd.read_excel('new_cables.xlsx')

2. Apply the same feature engineering as in run_model.py:
   - Calculate age_index, loading_pct, joint_density, etc.
   - Create seasonal features

3. Load model and preprocessor:
   
   import joblib
   model = joblib.load('cable_failure_rf_model.pkl')
   prep = joblib.load('cable_preprocessor.pkl')

4. Preprocess and predict:
   
   # Select same features used in training
   features = ['CableSize', 'CableType', 'Temperature', ...]
   X_new = new_data[features]
   
   # Transform
   X_new_prep = prep.transform(X_new)
   
   # Predict probability
   probabilities = model.predict_proba(X_new_prep)[:, 1]
   new_data['failure_probability'] = probabilities
   
   # Classify
   predictions = model.predict(X_new_prep)
   new_data['predicted_status'] = predictions  # 0=Healthy, 1=Failed

5. Export results:
   
   new_data.to_excel('predictions.xlsx', index=False)

📊 To retrain the model with new data:
   - Add new failure records to Failure_Data.xlsx
   - Add new healthy records to Healthy_Data.xlsx
   - Run: python run_model.py

📈 To view feature importance:
   - Open feature_importance.csv in Excel
   - Sort by 'Importance' column (descending)
""")

print("=" * 80)
print("✅ INSPECTION COMPLETE")
print("=" * 80)
