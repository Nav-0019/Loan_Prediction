import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import os

# --- Constants ---
RISK_BINS = [0, 0.10, 0.15, 1.0]
RISK_LABELS = ['stable zone', 'risk zone', 'high risk zone']

PROFESSION_GROUPS = {
    'Physician': 'High_Income_Specialized', 'Surgeon': 'High_Income_Specialized', 
    'Dentist': 'High_Income_Specialized', 'Chartered_Accountant': 'High_Income_Specialized', 
    'Lawyer': 'High_Income_Specialized', 'Magistrate': 'High_Income_Specialized', 
    'Architect': 'High_Income_Specialized', 'Financial_Analyst': 'High_Income_Specialized', 
    'Software_Developer': 'STEM_Technical', 'Mechanical_engineer': 'STEM_Technical', 
    'Chemical_engineer': 'STEM_Technical', 'Civil_engineer': 'STEM_Technical', 
    'Industrial_Engineer': 'STEM_Technical', 'Design_Engineer': 'STEM_Technical', 
    'Computer_hardware_engineer': 'STEM_Technical', 'Biomedical_Engineer': 'STEM_Technical', 
    'Petroleum_Engineer': 'STEM_Technical', 'Engineer': 'STEM_Technical', 
    'Scientist': 'STEM_Technical', 'Technology_specialist': 'STEM_Technical', 
    'Economist': 'STEM_Technical', 'Civil_servant': 'Government_Civil', 
    'Police_officer': 'Government_Civil', 'Firefighter': 'Government_Civil', 
    'Army_officer': 'Government_Civil', 'Official': 'Government_Civil', 
    'Librarian': 'Government_Civil', 'Technician': 'Service_Mid_Range', 
    'Secretary': 'Service_Mid_Range', 'Drafter': 'Service_Mid_Range', 
    'Computer_operator': 'Service_Mid_Range', 'Hotel_Manager': 'Service_Mid_Range', 
    'Surveyor': 'Service_Mid_Range', 'Geologist': 'Service_Mid_Range', 
    'Microbiologist': 'Service_Mid_Range', 'Analyst': 'Analyst_Consulting', 
    'Statistician': 'Analyst_Consulting', 'Consultant': 'Analyst_Consulting', 
    'Psychologist': 'Analyst_Consulting', 'Artist': 'Creative_Hospitality', 
    'Designer': 'Creative_Hospitality', 'Graphic_Designer': 'Creative_Hospitality', 
    'Web_designer': 'Creative_Hospitality', 'Fashion_Designer': 'Creative_Hospitality', 
    'Chef': 'Creative_Hospitality', 'Comedian': 'Creative_Hospitality', 
    'Technical_writer': 'Creative_Hospitality', 'Flight_attendant': 'Aviation_Operations', 
    'Air_traffic_controller': 'Aviation_Operations', 'Aviator': 'Aviation_Operations', 
    'Politician': 'Aviation_Operations'
}

ALL_PROFESSION_GROUPS = list(set(PROFESSION_GROUPS.values()))

def calculate_risk_maps(df_train):
    state_risk_map = df_train.groupby('STATE')['Risk_Flag'].mean()
    city_risk_map = df_train.groupby('CITY')['Risk_Flag'].mean()
    return state_risk_map, city_risk_map

def apply_transformations(df, state_map=None, city_map=None, is_training=True):
    df = pd.DataFrame(df).copy()
    
    # Binary encodings
    df["Married/Single"] = df["Married/Single"].map({'single':0, 'married':1}).fillna(0).astype(int)
    df["Car_Ownership"] = df["Car_Ownership"].map({'no':0, 'yes':1}).fillna(0).astype(int)

    # House OHE
    House_map = {'rented':1, 'owned':2, 'norent_noown':0}
    df['House_Ownership'] = df['House_Ownership'].map(House_map).fillna(0)
    df['House_Ownership_owned'] = (df['House_Ownership'] == 2).astype(int)
    df['House_Ownership_rented'] = (df['House_Ownership'] == 1).astype(int)

    # Profession group + OHE
    df['Profession_Group'] = df['Profession'].map(PROFESSION_GROUPS).fillna(ALL_PROFESSION_GROUPS[0])
    Prof_Encoder = pd.get_dummies(df['Profession_Group'], prefix='Profession')
    df = pd.concat([df, Prof_Encoder], axis=1)

    # State/City risk scores
    if is_training:
        df['State_Risk_Score'] = df['STATE'].map(state_map).fillna(df['STATE'].map(state_map).mean())
        df['City_Risk_Score'] = df['CITY'].map(city_map).fillna(df['CITY'].map(city_map).mean())
    else:
        mean_state_risk = state_map.mean() if not state_map.empty else 0.125
        mean_city_risk = city_map.mean() if not city_map.empty else 0.125
        df['State_Risk_Score'] = df['STATE'].map(state_map).fillna(mean_state_risk)
        df['City_Risk_Score'] = df['CITY'].map(city_map).fillna(mean_city_risk)

    # Risk zones
    df['State_Risk_Zone'] = pd.cut(df['State_Risk_Score'], bins=RISK_BINS, labels=RISK_LABELS, right=True, include_lowest=True)
    df['City_Risk_Zone'] = pd.cut(df['City_Risk_Score'], bins=RISK_BINS, labels=RISK_LABELS, right=True, include_lowest=True)
    Encoder_State = pd.get_dummies(df['State_Risk_Zone'], prefix='Zone_Risk_State', drop_first=True)
    Encoder_City = pd.get_dummies(df['City_Risk_Zone'], prefix='Zone_Risk_City', drop_first=True)
    df = pd.concat([df, Encoder_State, Encoder_City], axis=1)

    # Drop unnecessary columns
    cols_to_drop = ['CITY', 'STATE', 'Profession', 'Profession_Group', 'Profession_Analyst_Consulting',
                    'State_Risk_Score', 'City_Risk_Score', 'State_Risk_Zone', 'City_Risk_Zone',
                    'House_Ownership', 'Id']
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    return df

def main():
    print("Loading data...")
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_FILE_PATH = os.path.join(SCRIPT_DIR, 'Training Data.csv')
    
    try:
        data = pd.read_csv(DATA_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: 'Training Data.csv' not found in {SCRIPT_DIR}")
        return

    print("Splitting data...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        data.drop(columns=['Risk_Flag']),
        data['Risk_Flag'],
        test_size=0.2,
        random_state=42,
        stratify=data['Risk_Flag']
    )

    print("Calculating risk maps...")
    state_map_train, city_map_train = calculate_risk_maps(pd.concat([X_train_raw, y_train], axis=1))

    print("Transforming data...")
    X_train = apply_transformations(X_train_raw, state_map_train, city_map_train, is_training=True)
    
    # Capture columns
    TRAINING_COLUMNS = X_train.columns.tolist()
    X_train = X_train.reindex(columns=TRAINING_COLUMNS, fill_value=0)

    print("Training model...")
    X_train = X_train.astype(float)
    model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    print("Saving artifacts...")
    artifacts = {
        'model': model,
        'state_risk_map': state_map_train,
        'city_risk_map': city_map_train,
        'training_columns': TRAINING_COLUMNS,
        'profession_groups': PROFESSION_GROUPS,
        'risk_bins': RISK_BINS,
        'risk_labels': RISK_LABELS,
        'all_profession_groups': ALL_PROFESSION_GROUPS
    }
    
    with open('loan_risk_artifacts.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    
    print("Done! Artifacts saved to loan_risk_artifacts.pkl")

if __name__ == "__main__":
    main()
