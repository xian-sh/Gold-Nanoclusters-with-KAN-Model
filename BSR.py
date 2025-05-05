# import pandas as pd
# import statsmodels.api as sm
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from math import sqrt
# import matplotlib.pyplot as plt
#
# # 读取数据
# data = pd.read_csv('wtt.csv')
#
# # 提取File Name, y和X
# file_names = data.iloc[:, 0]
# y = data.iloc[:, 1]
# X = data.iloc[:, 2:]
#
# # 添加常数项
# X_with_const = sm.add_constant(X)
#
# # 定义逐步回归函数
# def stepwise_selection(X, y, initial_list=[], threshold_in=0.25, threshold_out=0.25, verbose=True):
#     included = list(initial_list)
#     while True:
#         changed = False
#         # forward step
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
#             if verbose:
#                 print(f'Add {best_feature} with p-value {best_pval:.6}')
#
#         # backward step
#         model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
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
# # 逐步双向回归
# selected_features = stepwise_selection(X, y)
#
# # 打印所选特征
# print("Selected features (Stepwise):", selected_features)
#
# # 保存选择的特征为csv
# feature_df = pd.DataFrame({
#     'File Name': file_names,
#     'True Values': y
# })
#
# # 添加所选特征
# for feature in selected_features:
#     feature_df[feature] = X[feature]
#
# feature_df.to_csv('feature-E-ST.csv', index=False)
#
# # 建立模型并计算R², MAE, RMSE
# def model_evaluation(X, y, selected_features):
#     X_selected = sm.add_constant(X[selected_features])
#     model = sm.OLS(y, X_selected).fit()
#     y_pred = model.predict(X_selected)
#     r2 = model.rsquared
#     mae = mean_absolute_error(y, y_pred)
#     rmse = sqrt(mean_squared_error(y, y_pred))
#     return r2, mae, rmse, y_pred
#
# # 评估模型
# results = model_evaluation(X, y, selected_features)
#
# print("\nModel Evaluation (R², MAE, RMSE):")
# print(f"Stepwise: {results[:3]}")
#
# # 绘制预测散点图
# true_values = y
# predictions = results[3]
#
# plt.figure(figsize=(8, 6))
# plt.scatter(true_values, predictions, alpha=0.5, label='Predictions vs True Values')
#
# # Plot y=x line (black solid line)
# plt.plot(true_values, true_values, color='black', linestyle='-', linewidth=1, label='y=x')
#
# # Plot y=x+0.2 line (black dashed line)
# plt.plot(true_values, true_values + 0.2, color='black', linestyle='--', linewidth=1, label='y=x+0.2')
#
# # Plot y=x-0.2 line (black dashed line)
# plt.plot(true_values, true_values - 0.2, color='black', linestyle='--', linewidth=1, label='y=x-0.2')
#
# # Labels and title
# plt.xlabel('True Values')
# plt.ylabel('Predictions')
# plt.title('Regression Scatter Plot')
# plt.legend()
# plt.show()




# import pandas as pd
# import statsmodels.api as sm
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from math import sqrt
# import matplotlib.pyplot as plt
#
# # 读取数据
# data = pd.read_csv('LUMO-99NEW.csv')
#
# # 将数据转换为数字类型，缺失值填充为0
# data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
#
# # 提取y和X
# y = data.iloc[:, 1]
# X = data.iloc[:, 2:]
#
# # 添加常数项
# X_with_const = sm.add_constant(X)
#
# # 定义逐步回归函数
# def stepwise_selection(X, y, initial_list=[], threshold_in=0.03, threshold_out=0.03, verbose=True):
#     included = list(initial_list)
#     while True:
#         changed = False
#         # forward step
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
#             if verbose:
#                 print(f'Add {best_feature} with p-value {best_pval:.6}')
#
#         # backward step
#         model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
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
# # 逐步双向回归
# selected_features = stepwise_selection(X, y)
#
# # 打印所选特征
# print("Selected features (Stepwise):", selected_features)
#
# # 保存选择的特征为csv
# feature_df = pd.DataFrame({
#     'True Values': y
# })
#
# # 添加所选特征
# for feature in selected_features:
#     feature_df[feature] = X[feature]
#
# feature_df.to_csv('LUMO-99NEW4_swt.csv', index=False)
#
# # 建立模型并计算R², MAE, RMSE
# def model_evaluation(X, y, selected_features):
#     X_selected = sm.add_constant(X[selected_features])
#     model = sm.OLS(y, X_selected).fit()
#     y_pred = model.predict(X_selected)
#     r2 = model.rsquared
#     mae = mean_absolute_error(y, y_pred)
#     rmse = sqrt(mean_squared_error(y, y_pred))
#     return r2, mae, rmse, y_pred
#
# # 评估模型
# results = model_evaluation(X, y, selected_features)
#
# print("\nModel Evaluation (R², MAE, RMSE):")
# print(f"Stepwise: {results[:3]}")
#
# # 绘制预测散点图
# true_values = y
# predictions = results[3]
#
# plt.figure(figsize=(8, 6))
# plt.scatter(true_values, predictions, alpha=0.5, label='Predictions vs True Values')
#
# # Plot y=x line (black solid line)
# plt.plot(true_values, true_values, color='black', linestyle='-', linewidth=1, label='y=x')
#
# # Plot y=x+0.2 line (black dashed line)
# plt.plot(true_values, true_values + 0.2, color='black', linestyle='--', linewidth=1, label='y=x+0.2')
#
# # Plot y=x-0.2 line (black dashed line)
# plt.plot(true_values, true_values - 0.2, color='black', linestyle='--', linewidth=1, label='y=x-0.2')
#
# # Labels and title
# plt.xlabel('True Values')
# plt.ylabel('Predictions')
# plt.title('Regression Scatter Plot')
# plt.legend()
# plt.show()


import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('LUMO-99NEW.csv')

# 将数据转换为数字类型，缺失值填充为0
data = data.apply(pd.to_numeric, errors='coerce').fillna(0)

# 去除相同内容的列
data = data.T.drop_duplicates().T

# 提取y和X
y = data.iloc[:, 1]
X = data.iloc[:, 2:]

# 添加常数项
X_with_const = sm.add_constant(X)

# 定义逐步回归函数
def stepwise_selection(X, y, initial_list=[], threshold_in=0.05, threshold_out=0.05, verbose=True):
    included = list(initial_list)
    while True:
        changed = False
        # forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(dtype=float, index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print(f'Add {best_feature} with p-value {best_pval:.6}')

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
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

# 打印所选特征
print("Selected features (Stepwise):", selected_features)

# 保存选择的特征为csv
feature_df = pd.DataFrame({
    'True Values': y
})

# 添加所选特征
for feature in selected_features:
    feature_df[feature] = X[feature]

feature_df.to_csv('LUMO-99NEW4_swt.csv', index=False)

# 建立模型并计算R², MAE, RMSE
def model_evaluation(X, y, selected_features):
    X_selected = sm.add_constant(X[selected_features])
    model = sm.OLS(y, X_selected).fit()
    y_pred = model.predict(X_selected)
    r2 = model.rsquared
    mae = mean_absolute_error(y, y_pred)
    rmse = sqrt(mean_squared_error(y, y_pred))
    return r2, mae, rmse, y_pred

# 评估模型
results = model_evaluation(X, y, selected_features)

print("\nModel Evaluation (R², MAE, RMSE):")
print(f"Stepwise: {results[:3]}")

# 绘制预测散点图
true_values = y
predictions = results[3]

plt.figure(figsize=(8, 6))
plt.scatter(true_values, predictions, alpha=0.5, label='Predictions vs True Values')

# Plot y=x line (black solid line)
plt.plot(true_values, true_values, color='black', linestyle='-', linewidth=1, label='y=x')

# Plot y=x+0.2 line (black dashed line)
plt.plot(true_values, true_values + 0.2, color='black', linestyle='--', linewidth=1, label='y=x+0.2')

# Plot y=x-0.2 line (black dashed line)
plt.plot(true_values, true_values - 0.2, color='black', linestyle='--', linewidth=1, label='y=x-0.2')

# Labels and title
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Regression Scatter Plot')
plt.legend()
plt.show()
