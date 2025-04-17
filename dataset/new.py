import pandas as pd

# Load the original split datasets
train_df = pd.read_csv("train_pos.csv")
val_df = pd.read_csv("val_pos.csv")
test_df = pd.read_csv("test_pos.csv")

# Load the new datasets
df_train = pd.read_csv('train4.csv')
df_test = pd.read_csv('test4.csv')
df_val = pd.read_csv('val4.csv')

# Debugging: Check column names
print("Columns in train_df:", train_df.columns)
print("Columns in df_train:", df_train.columns)

# Function to add the 'positional_sensitivity' column
def add_positional_sensitivity(df, odf, left_on, right_on):
    # Debugging: Check for matching rows
    print("Sample rows from df:", df[[left_on]].head())
    print("Sample rows from odf:", odf[[right_on]].head())
    
    # Merge the DataFrames on the specified columns
    merged_df = df.merge(odf[[right_on, 'positional_sensitivity']], left_on=left_on, right_on=right_on, how='left')
    print("Merged DataFrame sample:", merged_df.head())  # Debugging
    
    # Update the original DataFrame with the new column
    df['positional_sensitivity'] = merged_df['positional_sensitivity'].fillna(0).astype(int)  # Ensure integers

# Add the column to each DataFrame
add_positional_sensitivity(df_train, train_df, left_on='emoji', right_on='sent1')
add_positional_sensitivity(df_val, val_df, left_on='emoji', right_on='sent1')
add_positional_sensitivity(df_test, test_df, left_on='emoji', right_on='sent1')

# Debugging: Check the updated DataFrames
print("Updated df_train sample:", df_train.head())

# Save the updated DataFrames to new CSV files
df_train.to_csv('train5.csv', index=False)
df_test.to_csv('test5.csv', index=False)
df_val.to_csv('val5.csv', index=False)