# import pandas as pd
# import statsmodels.api as sm
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from math import sqrt
#
# # 读取数据
# data = pd.read_csv('jtt0710-mix-HOMO(1).csv')
#
# # 提取y和X
# y = data.iloc[:, 1]
# X = data.iloc[:, 2:]
#
# # 添加常数项
# X_with_const = sm.add_constant(X)
#
#
# # 定义逐步回归函数
# def stepwise_selection(X, y,
#                        initial_list=[],
#                        threshold_in=0.3,
#                        threshold_out=0.3,
#                        verbose=True):
#     included = list(initial_list)
#     while True:
#         changed = False
#         # forward step
#         excluded = list(set(X.columns) - set(included))
#         new_pval = pd.Series(dtype=float, index=excluded)  # 指定 dtype
#         for new_column in excluded:
#             model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
#             new_pval[new_column] = model.pvalues[new_column]
#         best_pval = new_pval.min()
#         if best_pval < threshold_in:
#             best_feature = new_pval.idxmin()
#             included.append(best_feature)
#             changed = True
#             if verbose:
#                 print(f'Add  {best_feature} with p-value {best_pval:.6}')
#
#         # backward step
#         model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
#         # use all coefs except intercept
#         pvalues = model.pvalues.iloc[1:]
#         worst_pval = pvalues.max()
#         if worst_pval > threshold_out:
#             changed = True
#             worst_feature = pvalues.idxmax()
#             included.remove(worst_feature)
#             if verbose:
#                 print(f'Drop {worst_feature} with p-value {worst_pval:.6}')
#
#         if not changed:
#             break
#
#     return included
#
#
# # 逐步双向回归
# selected_features = stepwise_selection(X, y)
#
#
# # 逐步正向回归
# def forward_selection(X, y, threshold_in=0.01, verbose=True):
#     included = []
#     while True:
#         changed = False
#         excluded = list(set(X.columns) - set(included))
#         new_pval = pd.Series(dtype=float, index=excluded)  # 指定 dtype
#         for new_column in excluded:
#             model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
#             new_pval[new_column] = model.pvalues[new_column]
#         best_pval = new_pval.min()
#         if best_pval < threshold_in:
#             best_feature = new_pval.idxmin()
#             included.append(best_feature)
#             changed = True
#             if verbose:
#                 print(f'Add  {best_feature} with p-value {best_pval:.6}')
#         if not changed:
#             break
#
#     return included
#
#
# selected_features_forward = forward_selection(X, y)
#
#
# # 逐步逆向回归
# def backward_elimination(X, y, threshold_out=0.001, verbose=True):
#     included = list(X.columns)
#     while True:
#         model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
#         pvalues = model.pvalues.iloc[1:]
#         worst_pval = pvalues.max()
#         if worst_pval > threshold_out:
#             changed = True
#             worst_feature = pvalues.idxmax()
#             included.remove(worst_feature)
#             if verbose:
#                 print(f'Drop {worst_feature} with p-value {worst_pval:.6}')
#         else:
#             changed = False
#         if not changed:
#             break
#
#     return included
#
#
# selected_features_backward = backward_elimination(X, y)
#
# # 打印各选择方法的参数
# print("Selected features (Stepwise):", selected_features)
# print("Selected features (Forward):", selected_features_forward)
# print("Selected features (Backward):", selected_features_backward)
#
#
# # 建立模型并计算R², MAE, RMSE
# def model_evaluation(X, y, selected_features):
#     X_selected = sm.add_constant(X[selected_features])
#     model = sm.OLS(y, X_selected).fit()
#     y_pred = model.predict(X_selected)
#     r2 = model.rsquared
#     mae = mean_absolute_error(y, y_pred)
#     rmse = sqrt(mean_squared_error(y, y_pred))
#     return r2, mae, rmse
#
#
# results = {}
# results['stepwise'] = model_evaluation(X, y, selected_features)
# results['forward'] = model_evaluation(X, y, selected_features_forward)
# results['backward'] = model_evaluation(X, y, selected_features_backward)
#
# print("\nModel Evaluation (R², MAE, RMSE):")
# print(f"Stepwise: {results['stepwise']}")
# print(f"Forward: {results['forward']}")
# print(f"Backward: {results['backward']}")


import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# 读取数据
data = pd.read_csv(r'D:\GAN1\SPMS\0718-HOMO.csv')

# 提取y和X
y = data.iloc[:, 1]
X = data.iloc[:, 2:]

# 添加常数项
X_with_const = sm.add_constant(X)


# 定义逐步回归函数
def stepwise_selection(X, y,
                       initial_list=[],
                       threshold_in=0.1,
                       threshold_out=0.1,
                       verbose=True):
    included = list(initial_list)
    while True:
        changed = False
        # forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(dtype=float, index=excluded)  # 指定 dtype
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print(f'Add  {best_feature} with p-value {best_pval:.6}')

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print(f'Drop {worst_feature} with p-value {worst_pval:.6}')

        if not changed:
            break

    return included


# 逐步双向回归
selected_features = stepwise_selection(X, y)


# 逐步正向回归
def forward_selection(X, y, threshold_in=0.1, verbose=True):
    included = []
    while True:
        changed = False
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(dtype=float, index=excluded)  # 指定 dtype
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print(f'Add  {best_feature} with p-value {best_pval:.6}')
        if not changed:
            break

    return included


selected_features_forward = forward_selection(X, y)


# 逐步逆向回归
def backward_elimination(X, y, threshold_out=0.01, verbose=True):
    included = list(X.columns)
    while True:
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print(f'Drop {worst_feature} with p-value {worst_pval:.6}')
        else:
            changed = False
        if not changed:
            break

    return included


selected_features_backward = backward_elimination(X, y)

# 打印各选择方法的参数
print("Selected features (Stepwise):", selected_features)
print("Selected features (Forward):", selected_features_forward)
print("Selected features (Backward):", selected_features_backward)


# 建立模型并计算R², MAE, RMSE
def model_evaluation(X, y, selected_features):
    X_selected = sm.add_constant(X[selected_features])
    model = sm.OLS(y, X_selected).fit()
    y_pred = model.predict(X_selected)
    r2 = model.rsquared
    mae = mean_absolute_error(y, y_pred)
    rmse = sqrt(mean_squared_error(y, y_pred))
    return r2, mae, rmse


results = {}
results['stepwise'] = model_evaluation(X, y, selected_features)
results['forward'] = model_evaluation(X, y, selected_features_forward)
results['backward'] = model_evaluation(X, y, selected_features_backward)

print("\nModel Evaluation (R², MAE, RMSE):")
print(f"Stepwise: {results['stepwise']}")
print(f"Forward: {results['forward']}")
print(f"Backward: {results['backward']}")





# import pandas as pd
# import statsmodels.api as sm
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from math import sqrt
#
# # Load the dataset
# data = pd.read_csv('jtt0710-mix-HOMO(1).csv')
#
# # Extract y and X
# y = data.iloc[:, 1]
# X = data.iloc[:, 2:]
#
# # Add constant
# X_with_const = sm.add_constant(X)
#
#
# # Forward selection function
# def forward_selection(X, y, threshold_in=0.01, verbose=False):
#     included = []
#     while True:
#         changed = False
#         excluded = list(set(X.columns) - set(included))
#         new_pval = pd.Series(dtype=float, index=excluded)
#         for new_column in excluded:
#             model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
#             new_pval[new_column] = model.pvalues[new_column]
#         best_pval = new_pval.min()
#         if best_pval < threshold_in:
#             best_feature = new_pval.idxmin()
#             included.append(best_feature)
#             changed = True
#         if not changed:
#             break
#     return included
#
#
# # Backward elimination function
# def backward_elimination(X, y, threshold_out=0.001, verbose=False):
#     included = list(X.columns)
#     while True:
#         model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
#         pvalues = model.pvalues.iloc[1:]
#         worst_pval = pvalues.max()
#         if worst_pval > threshold_out:
#             worst_feature = pvalues.idxmax()
#             included.remove(worst_feature)
#         else:
#             break
#     return included
#
#
# # Model evaluation function
# def model_evaluation(X, y, selected_features):
#     X_selected = sm.add_constant(X[selected_features])
#     model = sm.OLS(y, X_selected).fit()
#     y_pred = model.predict(X_selected)
#     r2 = model.rsquared
#     mae = mean_absolute_error(y, y_pred)
#     rmse = sqrt(mean_squared_error(y, y_pred))
#     return y_pred, r2, mae, rmse
#
#
# # Create subplots
# fig, axes = plt.subplots(2, 3, figsize=(18, 12))
# fig.suptitle('Forward and Backward Regression Analysis')
#
# # Forward selection thresholds
# forward_thresholds = [0.01, 0.05, 0.1]
# for i, threshold in enumerate(forward_thresholds):
#     selected_features = forward_selection(X, y, threshold_in=threshold)
#     y_pred, r2, mae, rmse = model_evaluation(X, y, selected_features)
#
#     ax = axes[0, i]
#     ax.scatter(y, y_pred, label='Data')
#
#     # Fit line and additional lines
#     ax.plot(y, y, color='blue', label='y = kx + b')
#     ax.plot(y, y + 0.2, color='red', linestyle='--', label='y = kx + b + 0.2')
#     ax.plot(y, y - 0.2, color='red', linestyle='--', label='y = kx + b - 0.2')
#
#     ax.set_title(f'Forward Selection (in={threshold})')
#     ax.set_xlabel('Actual')
#     ax.set_ylabel('Predicted')
#     ax.text(0.95, 0.05, f'Features: {len(selected_features)}\nR²: {r2:.2f}\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}',
#             transform=ax.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right')
#     ax.legend()
#
# # Backward elimination thresholds
# backward_thresholds = [0.001, 0.01, 0.05]
# for i, threshold in enumerate(backward_thresholds):
#     selected_features = backward_elimination(X, y, threshold_out=threshold)
#     y_pred, r2, mae, rmse = model_evaluation(X, y, selected_features)
#
#     ax = axes[1, i]
#     ax.scatter(y, y_pred, label='Data')
#
#     # Fit line and additional lines
#     ax.plot(y, y, color='blue', label='y = kx + b')
#     ax.plot(y, y + 0.2, color='red', linestyle='--', label='y = kx + b + 0.2')
#     ax.plot(y, y - 0.2, color='red', linestyle='--', label='y = kx + b - 0.2')
#
#     ax.set_title(f'Backward Elimination (out={threshold})')
#     ax.set_xlabel('Actual')
#     ax.set_ylabel('Predicted')
#     ax.text(0.95, 0.05, f'Features: {len(selected_features)}\nR²: {r2:.2f}\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}',
#             transform=ax.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right')
#     ax.legend()
#
# plt.tight_layout(rect=[0, 0.03, 1, 0.97])
# plt.savefig('fit.png', dpi=800)
# plt.show()