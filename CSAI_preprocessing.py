import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import os
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Load datasets
# df_attack_samba = pd.read_csv(r"C:\Users\sreet\Downloads\MCAD-SDN\attack_samba.csv")

# df_os_port_scan = pd.read_csv(r"C:\Users\sreet\Downloads\MCAD-SDN\attack_os_port_scan.csv")
# df_attack_tcp = pd.read_csv(r"C:\Users\sreet\Downloads\MCAD-SDN\attack_ddos_tcp.csv")
# df_attack_sql_injection = pd.read_csv(r"C:\Users\sreet\Downloads\MCAD-SDN\attack_sql_injection.csv")
# df_attack_vnc = pd.read_csv(r"C:\Users\sreet\Downloads\MCAD-SDN\attack_vnc.csv")
# df_normal_internet2 = pd.read_csv(r"C:\Users\sreet\Downloads\MCAD-SDN\normal_internet2.csv")
# df_normal_iperf = pd.read_csv(r"C:\Users\sreet\Downloads\MCAD-SDN\normal_iperf.csv")
# #df_attack_samba = pd.read_csv(r"C:\Users\sreet\Downloads\MCAD-SDN\attack_samba.csv")
# df_attack_bruteforce = pd.read_csv(r"C:\Users\sreet\Downloads\MCAD-SDN\attack_bruteforce.csv")
# df_attack_ddos_udp = pd.read_csv(r"C:\Users\sreet\Downloads\MCAD-SDN\attack_ddos_udp.csv")
# df_attack_scapy_new = pd.read_csv(r"C:\Users\sreet\Downloads\MCAD-SDN\ddos_attack_scapy_new.csv")
# df_attack_cmd = pd.read_csv(r"C:\Users\sreet\Downloads\MCAD-SDN\attack_cmd.csv")


# # Assign labels: 1 for attack, 0 for normal
# df_os_port_scan['target'] = 1
# df_attack_sql_injection['target'] = 1
# df_attack_vnc['target'] = 1
# df_normal_internet2 ['target'] = 0
# df_normal_iperf['target'] = 0
# df_attack_samba['target'] = 1
# df_attack_bruteforce['target'] = 1
# df_attack_cmd['target'] = 1

# Concatenate the datasets
# df = pd.concat([df_os_port_scan, df_attack_sql_injection, df_attack_vnc, df_normal_internet2,df_normal_iperf, df_attack_samba, df_attack_bruteforce, df_attack_cmd ])

d1 = pd.read_csv(r"C:\Users\sreet\Downloads\Python\work.csv")
d2 = pd.read_csv(r"C:\Users\sreet\Downloads\Python\work2.csv")

# Concatenate datasets
df = pd.concat([d1,d2], axis=0, ignore_index=True)
# Separate numeric and non-numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop('target')
non_numeric_cols = df.select_dtypes(exclude=['float64', 'int64']).columns

# Step 1: Convert non-numeric data (like IP addresses) using Label Encoding
label_enc = LabelEncoder()
for col in non_numeric_cols:
    df[col] = label_enc.fit_transform(df[col].astype(str))

# Step 2: Handle missing values by replacing with the median for numeric columns
imputer = SimpleImputer(strategy='median')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# Step 3: Normalize the features using StandardScaler (exclude the target)
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Step 4: Handle class imbalance using SMOTE (ensure 'target' is not scaled)
X = df.drop('target', axis=1)  # Features only
y = df['target']  # Target (attack/normal)

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Combine the resampled data into a final DataFrame
df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=['target'])], axis=1)

# Step 5: Feature selection using Random Forest/XGBoost (if necessary)
# ...

df_resampled.head()

# Save the preprocessed data to a new CSV file
output_path = os.path.abspath('work3.csv')
df_resampled.to_csv(output_path, index=False)

# Print the path of the saved CSV file
print(f"Preprocessed data saved at: {output_path}")
