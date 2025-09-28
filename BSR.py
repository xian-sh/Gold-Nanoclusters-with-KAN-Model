import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

# Read CSV data
data = pd.read_csv('LUMO-99NEW.csv')

# Convert all data to numeric, fill NaN with 0
data = data.apply(pd.to_numeric, errors='coerce').fillna(0)

# Remove duplicated columns (columns with identical content)
data = data.T.drop_duplicates().T

# Extract target variable y and predictors X
# Assumes y is the 2nd column, predictors start from 3rd column
y = data.iloc[:, 1]
X = data.iloc[:, 2:]

# Add intercept to predictors
X_with_const = sm.add_constant(X)

# Stepwise regression (bidirectional) function
def stepwise_selection(X, y, initial_list=[], threshold_in=0.05, threshold_out=0.05, verbose=True):
    """Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns:
        List of selected features 
    """
    included = list(initial_list)
    while True:
        changed = False
        # Forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(dtype=float, index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        if not new_pval.empty:
            best_pval = new_pval.min()
            if best_pval < threshold_in:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed = True
                if verbose:
                    print(f'Add {best_feature} with p-value {best_pval:.6f}')
        # Backward step
        if included:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
            # all coefs except intercept
            pvalues = model.pvalues.iloc[1:]
            worst_pval = pvalues.max()
            if worst_pval > threshold_out:
                changed = True
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                if verbose:
                    print(f'Drop {worst_feature} with p-value {worst_pval:.6f}')
        if not changed:
            break
    return included

# Perform stepwise regression
selected_features = stepwise_selection(X, y)

print("Selected features (Stepwise):", selected_features)

# Save selected features and target to a new CSV
feature_df = pd.DataFrame({
    'True Values': y
})

for feature in selected_features:
    feature_df[feature] = X[feature]

feature_df.to_csv('LUMO-99NEW4_swt.csv', index=False)

# Model fitting and evaluation function
def model_evaluation(X, y, selected_features):
    """Train OLS model and return R², MAE, RMSE, predictions"""
    X_selected = sm.add_constant(X[selected_features])
    model = sm.OLS(y, X_selected).fit()
    y_pred = model.predict(X_selected)
    r2 = model.rsquared
    mae = mean_absolute_error(y, y_pred)
    rmse = sqrt(mean_squared_error(y, y_pred))
    return r2, mae, rmse, y_pred

# Evaluate model
r2, mae, rmse, predictions = model_evaluation(X, y, selected_features)

print("\nModel Evaluation (R², MAE, RMSE):")
print(f"Stepwise: {(r2, mae, rmse)}")

# Scatter plot: True vs. Predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y, predictions, alpha=0.5, label='Predictions vs True Values')

# y=x reference line (solid)
plt.plot(y, y, color='black', linestyle='-', linewidth=1, label='y=x')
# y=x+0.2 (dashed)
plt.plot(y, y + 0.2, color='black', linestyle='--', linewidth=1, label='y=x+0.2')
# y=x-0.2 (dashed)
plt.plot(y, y - 0.2, color='black', linestyle='--', linewidth=1, label='y=x-0.2')

plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Regression Scatter Plot')
plt.legend()
plt.tight_layout()
plt.show()
