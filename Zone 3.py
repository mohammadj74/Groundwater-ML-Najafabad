# ---- Libraries ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import warnings
from matplotlib import rcParams
from tabulate import tabulate
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as font_manager

# Setting font and style for plots
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['figure.figsize'] = 12, 8
rcParams['font.family'] = 'Times New Roman'  # Setting Times New Roman as font family
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 16
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 12

# Suppress warnings
warnings.filterwarnings('ignore')

# ---- Reading the original data ----
data = pd.read_csv('amirye+ab.txt', delimiter='\t', index_col='dateMiladi', parse_dates=True)
print("Original data loaded. Number of samples:", len(data))

# ---- Preparation ----
data['year1'] = data.index.year
data['month1'] = data.index.month
data['day1'] = data.index.day
data['date_formatted'] = data.index.strftime('%Y-%m')  # For use in plots
data['season'] = pd.cut(data['month1'],
                        bins=[0, 3, 6, 9, 12],
                        labels=['Winter', 'Spring', 'Summer', 'Fall'])

# ---- Improved plot rendering ----
# 1. Plot of water level decline over time
plt.figure(figsize=(14, 8))
sns.lineplot(x=data.index, y=data['oft'], linewidth=2.5, color='#1f77b4')
plt.fill_between(data.index, data['oft'], alpha=0.3, color='#1f77b4')
plt.title('Groundwater Level Decline Over Time', fontsize=18, fontweight='bold', fontfamily='Times New Roman')
plt.xlabel('Time', fontsize=14, fontweight='bold', fontfamily='Times New Roman')
plt.ylabel('Groundwater Level Decline (m)', fontsize=14, fontweight='bold', fontfamily='Times New Roman')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 2. Groundwater level trend plot
plt.figure(figsize=(14, 8))
sns.regplot(x=data['year1'], y=data['taraz'],
            scatter_kws={'alpha': 0.6, 's': 40, 'color': '#2ca02c'},
            line_kws={'color': 'red', 'linewidth': 2})
plt.title('Groundwater Level Trend', fontsize=18, fontweight='bold', fontfamily='Times New Roman')
plt.xlabel('Year', fontsize=14, fontweight='bold', fontfamily='Times New Roman')
plt.ylabel('Groundwater Level (meter)', fontsize=14, fontweight='bold', fontfamily='Times New Roman')
plt.xlim(min(data['year1']) - 0.5, max(data['year1']) + 0.5)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 3. Violin plot with embedded boxplot of decline by month
plt.figure(figsize=(14, 8))
ax = sns.violinplot(x='month1', y='oft', data=data, palette='viridis',
                    inner='box', scale='width', bw=0.2)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticklabels(months)
plt.title('Groundwater Level Decline Variations by Month', fontsize=18, fontweight='bold', fontfamily='Times New Roman')
plt.xlabel('Month', fontsize=14, fontweight='bold', fontfamily='Times New Roman')
plt.ylabel('Groundwater Level Decline (m)', fontsize=14, fontweight='bold', fontfamily='Times New Roman')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 4. Seasonal comparison of groundwater levels with violin plot
plt.figure(figsize=(12, 8))
sns.violinplot(x='season', y='taraz', data=data, palette='Set2',
               inner='box', scale='width', bw=0.2)
plt.title('Seasonal Comparison of Groundwater Levels', fontsize=18, fontweight='bold', fontfamily='Times New Roman')
plt.xlabel('Season', fontsize=14, fontweight='bold', fontfamily='Times New Roman')
plt.ylabel('Groundwater Level (m)', fontsize=14, fontweight='bold', fontfamily='Times New Roman')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# تغییر نام‌ها
renamed_data = data.rename(columns={
    'taraz': 'GwL',
    'abyari': 'IV',
    'brdstjshan': 'GwA',
    'oft': 'GwDL',
    # اگر oft متغیر دیگری است، تغییر نده
})

correlation_columns = ['GwL', 'T', 'P', 'IV','GwA','GwDL']
correlation_matrix = renamed_data[correlation_columns].corr()

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',  # یا 'plasma', 'cividis' برای جذابیت بصری
    fmt=".2f",
    mask=mask,
    annot_kws={"size": 12},
    cbar_kws={"shrink": 0.8},  # رنگ‌سنج فشرده
    square=True,
    linewidths=0
)

plt.grid(False)
plt.title('Correlation Between Main Variables', fontsize=18, fontweight='bold', fontfamily='Times New Roman')
plt.tight_layout()
plt.show()


# ---- Creating Time Series ----
seryzamani = data[['T', 'P', 'abyari', 'oft', 'brdstjshan']]

# ---- Split data into training and testing ----
train_end = '2018-07-20'
test_start = '2018-08-20'

x = seryzamani[['T', 'P', 'abyari', 'brdstjshan']]
y = seryzamani['oft']

train_mask = data.index <= train_end
test_mask = data.index >= test_start

xtrain = x[train_mask]
ytrain = y[train_mask]
xtest = x[test_mask]
ytest = y[test_mask]

print(f"Training data: {len(xtrain)} samples")
print(f"Testing data: {len(xtest)} samples")

# Distribution plot of training and testing data
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.hist(ytrain, bins=15, alpha=0.7, color='blue', label='Training')
plt.hist(ytest, bins=15, alpha=0.7, color='red', label='Testing')
plt.title('Distribution of Decline Values in Training and Testing Sets', fontsize=14, fontfamily='Times New Roman')
plt.xlabel('Decline (meter)', fontsize=12, fontfamily='Times New Roman')
plt.ylabel('Frequency', fontsize=12, fontfamily='Times New Roman')
plt.legend(prop={'family': 'Times New Roman'})
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
sns.kdeplot(ytrain, label='Training', color='blue')
sns.kdeplot(ytest, label='Testing', color='red')
plt.title('Density Distribution of Decline Values', fontsize=14, fontfamily='Times New Roman')
plt.xlabel('Decline (meter)', fontsize=12, fontfamily='Times New Roman')
plt.ylabel('Density', fontsize=12, fontfamily='Times New Roman')
plt.legend(prop={'family': 'Times New Roman'})
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# ---- Model Evaluation Function ----
def evaluate_model(model_name, y_true_train, y_pred_train, y_true_test, y_pred_test):
    """Calculate and return evaluation metrics for the model"""
    metrics = {}

    # RMSE
    metrics['Train RMSE'] = np.sqrt(mean_squared_error(y_true_train, y_pred_train))
    metrics['Test RMSE'] = np.sqrt(mean_squared_error(y_true_test, y_pred_test))

    # R²
    metrics['Train R²'] = r2_score(y_true_train, y_pred_train)
    metrics['Test R²'] = r2_score(y_true_test, y_pred_test)

    # MAE
    metrics['Train MAE'] = mean_absolute_error(y_true_train, y_pred_train)
    metrics['Test MAE'] = mean_absolute_error(y_true_test, y_pred_test)

    # MAPE
    metrics['Train MAPE'] = mean_absolute_percentage_error(y_true_train, y_pred_train)
    metrics['Test MAPE'] = mean_absolute_percentage_error(y_true_test, y_pred_test)

    # d (Index of Agreement)
    metrics['Train d'] = 1 - (np.sum((y_true_train - y_pred_train) ** 2) / np.sum(
        (np.abs(y_true_train - np.mean(y_true_train)) + np.abs(y_pred_train - np.mean(y_true_train))) ** 2))
    metrics['Test d'] = 1 - (np.sum((y_true_test - y_pred_test) ** 2) / np.sum(
        (np.abs(y_true_test - np.mean(y_true_test)) + np.abs(y_pred_test - np.mean(y_true_test))) ** 2))

    return metrics

# Dictionary to store results of all models
all_metrics = {}
all_predictions = {}

# ---- SVM Model ----
print("\n--- Support Vector Machine (SVM) Model ---")
svm_model = SVR(kernel='rbf', epsilon=0.07, C=22, gamma=0.4)
svm_model.fit(xtrain, ytrain)

trainsvm = svm_model.predict(xtrain)
testsvm = svm_model.predict(xtest)

# Save predictions
all_predictions['SVM'] = {'train': trainsvm, 'test': testsvm}

# Evaluate model
svm_metrics = evaluate_model('SVM', ytrain, trainsvm, ytest, testsvm)
all_metrics['SVM'] = svm_metrics

# ---- Random Forest Model ----
print("\n--- Random Forest Model ---")
rf_model = RandomForestRegressor(max_leaf_nodes=40, n_estimators=100, random_state=42)
rf_model.fit(xtrain, ytrain)

trainrf = rf_model.predict(xtrain)
testrf = rf_model.predict(xtest)

# Save predictions
all_predictions['Random Forest'] = {'train': trainrf, 'test': testrf}

# Evaluate model
rf_metrics = evaluate_model('Random Forest', ytrain, trainrf, ytest, testrf)
all_metrics['Random Forest'] = rf_metrics

# ---- XGBoost Model ----
print("\n--- XGBoost Model ---")
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=40,
    learning_rate=0.35,
    n_jobs=5,
    verbosity=0
)
xgb_model.fit(xtrain, ytrain)

trainxg = xgb_model.predict(xtrain)
testxg = xgb_model.predict(xtest)

# Save predictions
all_predictions['XGBoost'] = {'train': trainxg, 'test': testxg}

# Evaluate model
xgb_metrics = evaluate_model('XGBoost', ytrain, trainxg, ytest, testxg)
all_metrics['XGBoost'] = xgb_metrics

# ---- Display evaluation metrics table for each model ----
# Convert results to DataFrame for better display
metrics_df = pd.DataFrame(all_metrics).T

# Set columns for better display
metrics_columns = ['Train RMSE', 'Test RMSE', 'Train R²', 'Test R²', 'Train MAE', 'Test MAE',
                   'Train MAPE', 'Test MAPE', 'Train d', 'Test d']
metrics_df = metrics_df[metrics_columns]

# Display table in console
print("\n=== Evaluation Metrics Table for All Models ===")
print(tabulate(metrics_df, headers='keys', tablefmt='pretty', floatfmt='.4f'))

# Visual display of the table
plt.figure(figsize=(14, 8))
ax = plt.subplot(111, frame_on=False)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
table = plt.table(cellText=metrics_df.round(4).values,
                  rowLabels=metrics_df.index,
                  colLabels=metrics_df.columns,
                  cellLoc='center',
                  loc='center',
                  colWidths=[0.09] * len(metrics_df.columns))
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
plt.title('Evaluation Metrics Table for All Models', fontsize=18, fontweight='bold', fontfamily='Times New Roman', pad=20)
plt.tight_layout()
plt.show()

# ---- Model Comparison Plot ----
plt.figure(figsize=(14, 10))
plt.plot(data.loc[test_mask].index, ytest, '-o', label='Actual Values', linewidth=3, markersize=8, alpha=0.7)
plt.plot(data.loc[test_mask].index, testsvm, '--', label='SVM', linewidth=2.5)
plt.plot(data.loc[test_mask].index, testrf, '--', label='Random Forest', linewidth=2.5)
plt.plot(data.loc[test_mask].index, testxg, '--', label='XGBoost', linewidth=2.5)
plt.title('Performance Comparison of Different Models in Water Level Decline Prediction', fontsize=18, fontweight='bold', fontfamily='Times New Roman')
plt.xlabel('Date', fontsize=14, fontfamily='Times New Roman')
plt.ylabel('Water Level Decline (m)', fontsize=14, fontfamily='Times New Roman')
plt.legend(fontsize=12, prop={'family': 'Times New Roman'})
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ---- Scatter Plot of Actual vs Predicted Values ----
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
models = ['SVM', 'Random Forest', 'XGBoost']
predictions = [testsvm, testrf, testxg]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, (model, pred, color) in enumerate(zip(models, predictions, colors)):
    axes[i].scatter(ytest, pred, alpha=0.7, s=50, color=color)
    axes[i].plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], 'k--', lw=2)
    axes[i].set_title(f'{model} Model', fontsize=16, fontfamily='Times New Roman')
    axes[i].set_xlabel('Actual Values', fontsize=12, fontfamily='Times New Roman')
    axes[i].set_ylabel('Predicted Values', fontsize=12, fontfamily='Times New Roman')
    axes[i].grid(True, linestyle='--', alpha=0.7)

    r2 = r2_score(ytest, pred)
    rmse = np.sqrt(mean_squared_error(ytest, pred))
    axes[i].text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}',
                 transform=axes[i].transAxes,
                 fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 fontfamily='Times New Roman')

plt.tight_layout()
plt.show()

# ---- Feature Importance for Random Forest Model ----
rf_importance = pd.DataFrame({
    'Feature': x.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=rf_importance, palette='viridis')
plt.title('Feature Importance in Random Forest Model', fontsize=16, fontfamily='Times New Roman')
plt.xlabel('Relative Importance', fontsize=14, fontfamily='Times New Roman')
plt.ylabel('Feature', fontsize=14, fontfamily='Times New Roman')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ---- Feature Importance for XGBoost Model ----
xgb_importance = pd.DataFrame({
    'Feature': x.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=xgb_importance, palette='plasma')
plt.title('Feature Importance in XGBoost Model', fontsize=16, fontfamily='Times New Roman')
plt.xlabel('Relative Importance', fontsize=14, fontfamily='Times New Roman')
plt.ylabel('Feature', fontsize=14, fontfamily='Times New Roman')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ---- Feature Importance for SVM Model ----
from sklearn.inspection import permutation_importance

r = permutation_importance(svm_model, xtest, ytest, n_repeats=30, random_state=42)
svm_importance = pd.DataFrame({
    'Feature': x.columns,
    'Importance': r.importances_mean
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=svm_importance, palette='magma')
plt.title('Feature Importance in SVM Model', fontsize=16, fontfamily='Times New Roman')
plt.xlabel('Relative Importance', fontsize=14, fontfamily='Times New Roman')
plt.ylabel('Feature', fontsize=14, fontfamily='Times New Roman')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ---- Comparison of Feature Importance Between Models ----
# داده‌های نمونه به صورت فرضی
all_importance = pd.DataFrame({
    'Feature': x.columns,
    'SVM': svm_importance['Importance'].values,
    'Random Forest': rf_importance['Importance'].values,
    'XGBoost': xgb_importance['Importance'].values
})

# تغییر نام ویژگی‌ها
all_importance['Feature'] = all_importance['Feature'].replace({'abyari': 'IV', 'v': 'GwA'})

# نرمال‌سازی اهمیت ویژگی‌ها به درصد
all_importance = pd.DataFrame({
    'Feature': x.columns,
    'SVM': svm_importance['Importance'].values,
    'Random Forest': rf_importance['Importance'].values,
    'XGBoost': xgb_importance['Importance'].values
})

# تغییر نام ویژگی‌ها
all_importance['Feature'] = all_importance['Feature'].replace({'abyari': 'IV', 'brdstjshan': 'GwA'})

# مرتب‌سازی دستی ترتیب ویژگی‌ها: P, T, IV, GwA
desired_order = ['P', 'T', 'IV', 'GwA']
all_importance = all_importance.set_index('Feature').loc[desired_order].reset_index()

# نرمال‌سازی اهمیت ویژگی‌ها به درصد
all_importance_relative = all_importance.copy()
for col in ['SVM', 'Random Forest', 'XGBoost']:
    all_importance_relative[col] = (all_importance_relative[col] / all_importance_relative[col].sum()) * 100

print("\n=== Feature Importance for All Models (Percentage) ===")
print(tabulate(all_importance_relative, headers='keys', tablefmt='pretty', floatfmt='.2f'))

# رسم نمودار
plt.figure(figsize=(10, 6))
x_pos = np.arange(len(all_importance_relative['Feature']))
width = 0.15

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x_pos - width, all_importance_relative['SVM'], width, label='SVM', color='#800080')         # بنفش
ax.bar(x_pos, all_importance_relative['Random Forest'], width, label='Random Forest', color='#B76E79')  # رز گلد
ax.bar(x_pos + width, all_importance_relative['XGBoost'], width, label='XGBoost', color='#6B8E23')     # سبز کله‌غازی

ax.set_xticks(x_pos)
ax.set_xticklabels(all_importance_relative['Feature'], rotation=0, ha="center")
ax.set_ylabel('Relative Importance (%)', fontsize=14, fontfamily='Times New Roman', fontweight='bold')
ax.set_title('Comparison of Feature Importance in Different Models', fontsize=18, fontfamily='Times New Roman')

# راهنمای بولد و درشت
ax.legend(prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 12})

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ---- Best Models Based on Different Metrics ----
best_models = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'R²', 'MAPE', 'd'],
    'Best Model (Test)': [
        metrics_df['Test RMSE'].idxmin(),
        metrics_df['Test MAE'].idxmin(),
        metrics_df['Test R²'].idxmax(),
        metrics_df['Test MAPE'].idxmin(),
        metrics_df['Test d'].idxmax()
    ],
    'Best Value (Test)': [
        metrics_df['Test RMSE'].min(),
        metrics_df['Test MAE'].min(),
        metrics_df['Test R²'].max(),
        metrics_df['Test MAPE'].min(),
        metrics_df['Test d'].max()
    ]
})

print("\n=== Summary of Results: Best Models Based on Different Metrics ===")
print(tabulate(best_models, headers='keys', tablefmt='pretty', floatfmt='.4f'))

# Visual display of summary results
plt.figure(figsize=(12, 6))
ax = plt.subplot(111, frame_on=False)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
summary_table = plt.table(cellText=best_models.values,
                          colLabels=best_models.columns,
                          cellLoc='center',
                          loc='center',
                          colWidths=[0.2, 0.4, 0.2])
summary_table.auto_set_font_size(False)
summary_table.set_fontsize(12)
summary_table.scale(1.2, 1.5)
plt.title('Summary of Results: Best Models Based on Different Metrics', fontsize=18, fontweight='bold', fontfamily='Times New Roman', pad=20)
plt.tight_layout()
plt.show()

# Comparison of RMSE between models
rmse_data = pd.DataFrame({
    'Model': ['SVM', 'Random Forest', 'XGBoost'],
    'Train RMSE': [all_metrics['SVM']['Train RMSE'],
                   all_metrics['Random Forest']['Train RMSE'],
                   all_metrics['XGBoost']['Train RMSE']],
    'Test RMSE': [all_metrics['SVM']['Test RMSE'],
                  all_metrics['Random Forest']['Test RMSE'],
                  all_metrics['XGBoost']['Test RMSE']]
})

# Plot comparing RMSE between models
rmse_melted = pd.melt(rmse_data, id_vars=['Model'], value_vars=['Train RMSE', 'Test RMSE'],
                       var_name='Dataset', value_name='RMSE')

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='RMSE', hue='Dataset', data=rmse_melted, palette=['#1f77b4', '#ff7f0e'])
plt.title('RMSE Comparison Between Different Models', fontsize=16, fontfamily='Times New Roman')
plt.xlabel('Model', fontsize=14, fontfamily='Times New Roman')
plt.ylabel('RMSE', fontsize=14, fontfamily='Times New Roman')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='', prop={'family': 'Times New Roman'})
plt.tight_layout()
plt.show()

# Comparison of R² between models
r2_data = pd.DataFrame({
    'Model': ['SVM', 'Random Forest', 'XGBoost'],
    'Train R²': [all_metrics['SVM']['Train R²'],
                  all_metrics['Random Forest']['Train R²'],
                  all_metrics['XGBoost']['Train R²']],
    'Test R²': [all_metrics['SVM']['Test R²'],
                 all_metrics['Random Forest']['Test R²'],
                 all_metrics['XGBoost']['Test R²']]
})

r2_melted = pd.melt(r2_data, id_vars=['Model'], value_vars=['Train R²', 'Test R²'],
                     var_name='Dataset', value_name='R²')

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='R²', hue='Dataset', data=r2_melted, palette=['#2ca02c', '#d62728'])
plt.title('R² Comparison Between Different Models', fontsize=16, fontfamily='Times New Roman')
plt.xlabel('Model', fontsize=14, fontfamily='Times New Roman')
plt.ylabel('R²', fontsize=14, fontfamily='Times New Roman')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='', prop={'family': 'Times New Roman'})
plt.tight_layout()
plt.show()