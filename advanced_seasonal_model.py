"""
ADVANCED CABLE FAILURE PREDICTION - SEASON-SPECIFIC MODELS
Train separate models for Summer/Monsoon/Winter with optimized hyperparameters
and detailed cable type analysis
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False

print("=" * 100)
print("ADVANCED CABLE FAILURE PREDICTION - SEASON-SPECIFIC MODELS")
print("=" * 100)
print()

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("📂 Step 1: Loading data...")
df_fail = pd.read_excel("Failure_Data.xlsx")
df_healthy = pd.read_excel("Healthy_Data.xlsx")

df_fail['target'] = 1
df_healthy['target'] = 0

# ============================================================================
# STEP 2: COMPREHENSIVE DATA CLEANING
# ============================================================================
print("\n🧹 Step 2: Comprehensive data cleaning...")

rename_map = {
    'Switch ID1': 'SwitchID', 'Cablesize': 'CableSize',
    'Cable Type (PILC/XLPE/Mixed)': 'CableType',
    'Age of XLPE Cable (MFY)': 'Age_XLPE', 'Age of PILC Cable (MFY)': 'Age_PILC',
    'OEM Rating (XLPE)': 'OEM_Rating',
    'AEML Derated Limit (XLPE)': 'AEML_Derated_Limit',
    'No. of Cables in Duct': 'No_of_Cables_in_Duct',
    'Cable Length': 'Cable_Length', ' No.of Joints': 'No_of_Joints',
    'Total length of Section (kM)': 'Total_Length_km',
    'Cable Condition': 'Cable_Condition', 'Age of the joint': 'Joint_Age',
    'Joint Age': 'Joint_Age', 'Section Order Number': 'Section_Order_Number',
    'Loading of Sections (amps)': 'Loading_amps'
}

df_fail = df_fail.rename(columns=rename_map)
df_healthy = df_healthy.rename(columns=rename_map)
df = pd.concat([df_fail, df_healthy], ignore_index=True, sort=False)

# Clean strings
for col in ['CableType', 'CableSize', 'Cable_Condition', 'Category']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True).str.upper()

# Convert numeric
numeric_cols = ['Temperature', 'Age_XLPE', 'Age_PILC', 'OEM_Rating', 'AEML_Derated_Limit',
                'No_of_Cables_in_Duct', 'Cable_Length', 'No_of_Joints', 'Total_Length_km',
                'Joint_Age', 'Loading_amps', 'Month']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"   ✓ Combined: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================================
# STEP 3: ADVANCED FEATURE ENGINEERING
# ============================================================================
print("\n⚙️  Step 3: Advanced feature engineering...")

# Season mapping
def month_to_season(m):
    if pd.isna(m): return 'UNKNOWN'
    m = int(m)
    if m in [3, 4, 5, 6]: return 'SUMMER'
    elif m in [7, 8, 9]: return 'MONSOON'
    elif m in [10, 11, 12, 1, 2]: return 'WINTER'
    return 'UNKNOWN'

df['season'] = df['Month'].apply(month_to_season)

# Age features
df['age_xlpe_filled'] = df['Age_XLPE'].fillna(0)
df['age_pilc_filled'] = df['Age_PILC'].fillna(0)
df['age_index'] = df[['age_xlpe_filled', 'age_pilc_filled']].replace(0, np.nan).mean(axis=1)
df['age_index_max'] = df[['age_xlpe_filled', 'age_pilc_filled']].max(axis=1)
df['cable_age_category'] = pd.cut(df['age_index_max'], bins=[0, 10, 20, 30, 100], 
                                   labels=['NEW', 'MEDIUM', 'OLD', 'VERY_OLD'])

# Loading features
df['loading_pct'] = df.apply(
    lambda r: (r['Loading_amps'] / r['AEML_Derated_Limit'] * 100)
    if pd.notnull(r['Loading_amps']) and pd.notnull(r['AEML_Derated_Limit']) and r['AEML_Derated_Limit'] > 0
    else np.nan, axis=1
)
df['loading_category'] = pd.cut(df['loading_pct'], bins=[0, 60, 80, 100, 200], 
                                 labels=['LOW', 'MEDIUM', 'HIGH', 'OVERLOAD'])

# Joint features
df['total_length_km_filled'] = df['Total_Length_km'].replace(0, np.nan)
df['joint_density'] = df.apply(
    lambda r: r['No_of_Joints'] / r['total_length_km_filled']
    if pd.notnull(r['No_of_Joints']) and pd.notnull(r['total_length_km_filled'])
    else np.nan, axis=1
)
df['joint_category'] = pd.cut(df['joint_density'], bins=[0, 5, 10, 20, 100], 
                               labels=['LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH'])

# Derating features
df['derating_ratio'] = df.apply(
    lambda r: (r['AEML_Derated_Limit'] / r['OEM_Rating'])
    if pd.notnull(r['AEML_Derated_Limit']) and pd.notnull(r['OEM_Rating']) and r['OEM_Rating'] > 0
    else np.nan, axis=1
)
df['derating_severity'] = pd.cut(df['derating_ratio'], bins=[0, 0.7, 0.85, 1.0, 2.0], 
                                  labels=['SEVERE', 'MODERATE', 'MILD', 'NONE'])

# Temperature features
df['temp_stress'] = df['Temperature'] / 40.0
df['temp_category'] = pd.cut(df['Temperature'], bins=[0, 30, 35, 40, 100], 
                              labels=['NORMAL', 'WARM', 'HOT', 'CRITICAL'])

# Cable type detailed analysis
df['CableType_clean'] = df['CableType'].fillna('UNKNOWN')
df['has_PILC'] = df['CableType_clean'].str.contains('PILC', na=False).astype(int)
df['has_XLPE'] = df['CableType_clean'].str.contains('XLPE', na=False).astype(int)
df['is_mixed'] = ((df['has_PILC'] == 1) & (df['has_XLPE'] == 1)).astype(int)
df['cable_type_simplified'] = df.apply(
    lambda r: 'MIXED' if r['is_mixed'] == 1 else ('PILC' if r['has_PILC'] == 1 else ('XLPE' if r['has_XLPE'] == 1 else 'OTHER')), 
    axis=1
)

# Joint age features
df['old_joint_flag'] = df['Joint_Age'].apply(lambda x: 1 if pd.notnull(x) and x >= 10 else 0)
df['joint_age_category'] = pd.cut(df['Joint_Age'], bins=[0, 5, 10, 15, 100], 
                                   labels=['NEW', 'MEDIUM', 'OLD', 'VERY_OLD'])

# Cable condition
cable_condition_map = {
    'SEVERELY DETERIORATED': 'SEVERE', 'DETERIORATED': 'DETERIORATED',
    'GOOD': 'GOOD', 'HEALTHY': 'GOOD', 'NORMAL': 'GOOD', 'SATISFACTORY': 'GOOD'
}
df['Cable_Condition_simple'] = df['Cable_Condition'].str.upper().map(cable_condition_map).fillna('UNKNOWN')

# Interaction features
df['age_temp_interaction'] = df['age_index_max'] * df['temp_stress']
df['loading_temp_interaction'] = df['loading_pct'] * df['temp_stress']
df['age_loading_interaction'] = df['age_index_max'] * df['loading_pct']

# Seasonal binary flags
df['is_summer'] = (df['season'] == 'SUMMER').astype(int)
df['is_monsoon'] = (df['season'] == 'MONSOON').astype(int)
df['is_winter'] = (df['season'] == 'WINTER').astype(int)

# Season-specific weighted features
df['temp_summer_weighted'] = df['temp_stress'] * df['is_summer'] * 1.5
df['loading_summer_weighted'] = df['loading_pct'] * df['is_summer'] * 1.3
df['joints_monsoon_weighted'] = df['joint_density'] * df['is_monsoon'] * 1.5
df['condition_monsoon_weighted'] = df['is_monsoon'] * 1.2
df['age_winter_weighted'] = df['age_index_max'] * df['is_winter'] * 1.3
df['loading_winter_weighted'] = df['loading_pct'] * df['is_winter'] * 1.2

print(f"   ✓ Created 30+ advanced features")

# ============================================================================
# STEP 4: PREPARE FEATURES
# ============================================================================
print("\n🔧 Step 4: Preparing features for modeling...")

features = [
    # Original numeric
    'Temperature', 'Loading_amps', 'AEML_Derated_Limit', 'OEM_Rating',
    'No_of_Cables_in_Duct', 'No_of_Joints', 'Total_Length_km', 'Joint_Age',
    
    # Derived numeric
    'age_index', 'age_index_max', 'loading_pct', 'joint_density', 
    'temp_stress', 'derating_ratio', 'old_joint_flag',
    
    # Interaction features
    'age_temp_interaction', 'loading_temp_interaction', 'age_loading_interaction',
    
    # Seasonal flags
    'is_summer', 'is_monsoon', 'is_winter',
    
    # Weighted seasonal features
    'temp_summer_weighted', 'loading_summer_weighted', 'joints_monsoon_weighted',
    'condition_monsoon_weighted', 'age_winter_weighted', 'loading_winter_weighted',
    
    # Cable type flags
    'has_PILC', 'has_XLPE', 'is_mixed',
    
    # Categorical
    'CableSize', 'cable_type_simplified', 'Cable_Condition_simple', 'season',
    'cable_age_category', 'loading_category', 'joint_category', 
    'derating_severity', 'temp_category', 'joint_age_category'
]

features = [f for f in features if f in df.columns]
X = df[features].copy()
y = df['target'].copy()

numeric_feats = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_feats = [c for c in X.columns if c not in numeric_feats]

print(f"   ✓ Total features: {len(features)}")
print(f"   ✓ Numeric: {len(numeric_feats)}, Categorical: {len(cat_feats)}")

# ============================================================================
# STEP 5: BUILD PREPROCESSING PIPELINE
# ============================================================================
print("\n🔀 Step 5: Building preprocessing pipeline...")

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_feats),
    ('cat', categorical_transformer, cat_feats)
])

# ============================================================================
# STEP 6: TRAIN GLOBAL MODEL (OPTIMIZED)
# ============================================================================
print("\n🤖 Step 6: Training GLOBAL optimized model...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

# SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train_prep, y_train)

# Hyperparameter tuning for Random Forest
print("   → Optimizing hyperparameters...")
rf_params = {
    'n_estimators': [300, 500],
    'max_depth': [10, 15, 20],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4]
}

rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_grid = GridSearchCV(rf_base, rf_params, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0)
rf_grid.fit(X_res, y_res)

rf_global = rf_grid.best_estimator_
print(f"   ✓ Best params: {rf_grid.best_params_}")

# Train XGBoost if available
if HAS_XGB:
    print("   → Training XGBoost...")
    xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=10, learning_rate=0.1, 
                                  random_state=42, n_jobs=-1)
    xgb_model.fit(X_res, y_res)
    print("   ✓ XGBoost trained")

# Evaluate global model
y_pred_global = rf_global.predict(X_test_prep)
y_pred_proba_global = rf_global.predict_proba(X_test_prep)[:, 1]

print("\n" + "=" * 100)
print("GLOBAL MODEL PERFORMANCE (All Seasons Combined)")
print("=" * 100)
print(f"Accuracy: {accuracy_score(y_test, y_pred_global):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, y_pred_proba_global):.4f}")
print("\n" + classification_report(y_test, y_pred_global, target_names=['Healthy', 'Failed']))

if HAS_XGB:
    y_pred_xgb = xgb_model.predict(X_test_prep)
    y_pred_proba_xgb = xgb_model.predict_proba(X_test_prep)[:, 1]
    print(f"\nXGBoost ROC-AUC: {roc_auc_score(y_test, y_pred_proba_xgb):.4f}")

# ============================================================================
# STEP 7: TRAIN SEASON-SPECIFIC MODELS
# ============================================================================
print("\n" + "=" * 100)
print("SEASON-SPECIFIC MODELS (Separate models for each season)")
print("=" * 100)

season_models = {}
season_preprocessors = {}
season_feature_importance = {}

for season_name in ['SUMMER', 'MONSOON', 'WINTER']:
    print(f"\n🌦️  Training {season_name} Model...")
    
    # Filter data for this season
    df_season = df[df['season'] == season_name].copy()
    
    if len(df_season) < 20:
        print(f"   ⚠️  Not enough data for {season_name} ({len(df_season)} samples) - skipping")
        continue
    
    X_season = df_season[features].copy()
    y_season = df_season['target'].copy()
    
    print(f"   → Data: {len(df_season)} samples ({y_season.sum()} failures)")
    
    # Train-test split
    if y_season.sum() < 5 or (len(y_season) - y_season.sum()) < 5:
        print(f"   ⚠️  Insufficient class samples - using simple model")
        continue
    
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_season, y_season, test_size=0.25, stratify=y_season, random_state=42
    )
    
    # Preprocessing
    prep_season = ColumnTransformer([
        ('num', numeric_transformer, numeric_feats),
        ('cat', categorical_transformer, cat_feats)
    ])
    
    X_train_s_prep = prep_season.fit_transform(X_train_s)
    X_test_s_prep = prep_season.transform(X_test_s)
    
    # SMOTE
    smote_s = SMOTE(random_state=42)
    X_res_s, y_res_s = smote_s.fit_resample(X_train_s_prep, y_train_s)
    
    # Train Random Forest
    rf_season = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_split=5, 
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    rf_season.fit(X_res_s, y_res_s)
    
    # Evaluate
    y_pred_s = rf_season.predict(X_test_s_prep)
    y_pred_proba_s = rf_season.predict_proba(X_test_s_prep)[:, 1]
    
    acc_s = accuracy_score(y_test_s, y_pred_s)
    auc_s = roc_auc_score(y_test_s, y_pred_proba_s)
    
    print(f"   ✓ {season_name} Model Performance:")
    print(f"      Accuracy: {acc_s:.4f}, ROC-AUC: {auc_s:.4f}")
    
    # Store models
    season_models[season_name] = rf_season
    season_preprocessors[season_name] = prep_season
    
    # Extract feature importance
    ohe = prep_season.named_transformers_['cat'].named_steps['onehot']
    ohe_feature_names = list(ohe.get_feature_names_out(cat_feats))
    all_feature_names = numeric_feats + ohe_feature_names
    
    fi_df = pd.DataFrame({
        'feature': all_feature_names,
        'importance': rf_season.feature_importances_
    }).sort_values('importance', ascending=False)
    
    season_feature_importance[season_name] = fi_df
    
    print(f"\n   🔑 Top 10 Features for {season_name}:")
    for idx, row in fi_df.head(10).iterrows():
        print(f"      {row['feature']:30s}: {row['importance']:.4f}")

# ============================================================================
# STEP 8: CABLE TYPE ANALYSIS
# ============================================================================
print("\n" + "=" * 100)
print("CABLE TYPE ANALYSIS")
print("=" * 100)

cable_type_analysis = df.groupby('cable_type_simplified').agg({
    'target': ['count', 'sum', 'mean'],
    'age_index_max': 'mean',
    'loading_pct': 'mean',
    'Temperature': 'mean',
    'joint_density': 'mean'
}).round(3)

cable_type_analysis.columns = ['_'.join(col) for col in cable_type_analysis.columns]
print("\nCable Type Failure Analysis:")
print(cable_type_analysis)

# ============================================================================
# STEP 9: SEASONAL FACTOR WEIGHTAGE REPORT
# ============================================================================
print("\n" + "=" * 100)
print("SEASONAL FACTOR WEIGHTAGE RECOMMENDATIONS")
print("=" * 100)

for season_name in ['SUMMER', 'MONSOON', 'WINTER']:
    if season_name not in season_feature_importance:
        continue
    
    fi_df = season_feature_importance[season_name]
    
    # Group by category
    categories = {
        'Temperature': ['Temperature', 'temp_stress', 'temp_summer_weighted', 'temp_category'],
        'Loading': ['Loading_amps', 'loading_pct', 'loading_summer_weighted', 'loading_winter_weighted', 'loading_category'],
        'Age': ['age_index', 'age_index_max', 'Age_XLPE', 'Age_PILC', 'age_winter_weighted', 'cable_age_category'],
        'Joints': ['No_of_Joints', 'joint_density', 'Joint_Age', 'old_joint_flag', 'joints_monsoon_weighted', 'joint_category', 'joint_age_category'],
        'Derating': ['derating_ratio', 'AEML_Derated_Limit', 'OEM_Rating', 'derating_severity'],
        'CableType': ['has_PILC', 'has_XLPE', 'is_mixed', 'cable_type_simplified'],
        'Condition': ['Cable_Condition_simple', 'condition_monsoon_weighted']
    }
    
    print(f"\n🌦️  {season_name} - Factor Weightage:")
    print("   " + "-" * 80)
    
    for cat_name, cat_features in categories.items():
        cat_imp = fi_df[fi_df['feature'].str.contains('|'.join(cat_features), case=False, na=False)]['importance'].sum()
        if cat_imp > 0:
            bar_length = int(cat_imp * 100)
            bar = '█' * min(bar_length, 50)
            print(f"   {cat_name:15s} │ {bar} {cat_imp:.4f}")

# ============================================================================
# STEP 10: SAVE MODELS AND RESULTS
# ============================================================================
print("\n💾 Step 10: Saving models and results...")

import joblib

# Save global model
joblib.dump(rf_global, 'cable_model_global_optimized.pkl')
joblib.dump(preprocessor, 'cable_preprocessor_global.pkl')
print("   ✓ Global model saved")

# Save season-specific models
for season_name, model in season_models.items():
    joblib.dump(model, f'cable_model_{season_name.lower()}.pkl')
    joblib.dump(season_preprocessors[season_name], f'cable_preprocessor_{season_name.lower()}.pkl')
    print(f"   ✓ {season_name} model saved")

# Save feature importance
global_fi_df = pd.DataFrame({
    'feature': numeric_feats + list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_feats)),
    'importance': rf_global.feature_importances_
}).sort_values('importance', ascending=False)

global_fi_df.to_csv('feature_importance_global.csv', index=False)

for season_name, fi_df in season_feature_importance.items():
    fi_df.to_csv(f'feature_importance_{season_name.lower()}.csv', index=False)

print("   ✓ Feature importance CSV files saved")

# Save cable type analysis
cable_type_analysis.to_csv('cable_type_analysis.csv')
print("   ✓ Cable type analysis saved")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 100)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 100)

print(f"""
✅ MODELS TRAINED:
   • Global Optimized Model (All seasons): ROC-AUC = {roc_auc_score(y_test, y_pred_proba_global):.4f}
   • Season-Specific Models: {len(season_models)} models (SUMMER, MONSOON, WINTER)
   
📊 FILES SAVED:
   • cable_model_global_optimized.pkl
   • cable_model_summer.pkl, cable_model_monsoon.pkl, cable_model_winter.pkl
   • feature_importance_global.csv (all factors)
   • feature_importance_summer/monsoon/winter.csv (season-specific)
   • cable_type_analysis.csv
   
🔑 TOP RECOMMENDATIONS BY SEASON:
""")

for season_name in ['SUMMER', 'MONSOON', 'WINTER']:
    if season_name in season_feature_importance:
        fi = season_feature_importance[season_name]
        print(f"\n   {season_name}:")
        for idx, row in fi.head(3).iterrows():
            print(f"      {idx+1}. {row['feature']}: {row['importance']:.4f}")

print("\n" + "=" * 100)
print("✅ ADVANCED MODEL TRAINING COMPLETE!")
print("=" * 100)
