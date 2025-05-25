# ---- Libraries ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.inspection import permutation_importance
import warnings
from matplotlib import rcParams
from tabulate import tabulate
import matplotlib.gridspec as gridspec

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

# ---- Reading Main Data ----
data = pd.read_csv("harimmean1390.txt", sep="\t")
print("Main data loaded. Number of samples:", len(data))

# Convert date column to datetime and set as index
data['dateMiladi'] = pd.to_datetime(data['dateMiladi'])
data_indexed = data.set_index('dateMiladi')

# Create a copy for adding time features
data2 = data.copy()
data2['dateMiladi'] = pd.to_datetime(data2['dateMiladi'])
data2['year1'] = data2['dateMiladi'].dt.year
data2['month1'] = data2['dateMiladi'].dt.month
data2['day1'] = data2['dateMiladi'].dt.day
data2['date_formatted'] = data2['dateMiladi'].dt.strftime('%Y-%m')  # For use in plots
data2['season'] = pd.cut(data2['month1'],
                         bins=[0, 3, 6, 9, 12],
                         labels=['Winter', 'Spring', 'Summer', 'Fall'])

# ---- Check for Missing Values ----
missing_values = data.isna().sum().sum()
print(f"Number of missing values: {missing_values}")

# ---- Descriptive Statistics ----
print("\n=== Descriptive Statistics ===")
print(data.describe().round(2))

# ---- Improved Plot Visualizations ----
# 1. Water level decline over time
plt.figure(figsize=(14, 8))
sns.lineplot(x=data2['dateMiladi'], y=data2['oft'], linewidth=2.5, color='#1f77b4')
plt.fill_between(data2['dateMiladi'], data2['oft'], alpha=0.3, color='#1f77b4')
plt.title('Groundwater Level Decline Over Time', fontsize=18, fontweight='bold')
plt.xlabel('Time', fontsize=14, fontweight='bold')
plt.ylabel('Groundwater Level Decline (m)', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('water_level_decline_over_time.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Groundwater level trend
plt.figure(figsize=(14, 8))
sns.regplot(x=data2['year1'], y=data2['taraz'],
            scatter_kws={'alpha': 0.6, 's': 40, 'color': '#2ca02c'},
            line_kws={'color': 'red', 'linewidth': 2})
plt.title('Groundwater Level Trend', fontsize=18, fontweight='bold')
plt.xlabel('Year', fontsize=14)
plt.ylabel('Groundwater Level (m)', fontsize=14)
plt.xlim(min(data2['year1']) - 0.5, max(data2['year1']) + 0.5)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('groundwater_level_trend.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Monthly water decline violin plot
plt.figure(figsize=(14, 8))
ax = sns.violinplot(x='month1', y='oft', data=data2, palette='viridis',
                    inner='box', scale='width', bw=0.2)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticklabels(months)
plt.title('Groundwater Level Decline Across Different Months', fontsize=18, fontweight='bold')
plt.xlabel('Month', fontsize=14, fontweight='bold')
plt.ylabel('Groundwater Level Decline (m)', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('monthly_water_decline_violinplot.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Seasonal comparison of groundwater levels
plt.figure(figsize=(12, 8))
sns.violinplot(x='season', y='taraz', data=data2, palette='Set2',
               inner='box', scale='width', bw=0.2)
plt.title('Seasonal Comparison of Groundwater Levels', fontsize=18, fontweight='bold')
plt.xlabel('Season', fontsize=14, fontweight='bold')
plt.ylabel('Groundwater Level (m)', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('seasonal_water_level_violinplot.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Correlation heatmap between variables
# Determine numeric columns for correlation calculation
# تغییر نام ستون‌ها
renamed_data = data.rename(columns={
    'taraz': 'GwL',
    'qlenj': 'RD',
    'brdstjshan': 'GwA',
    'oft': 'GwDL'
})

# ستون‌های مورد نظر برای محاسبه همبستگی (بدون qhabl و ghabl2)
correlation_columns = ['GwL', 'T', 'P', 'RD', 'GwA', 'GwDL']

# محاسبه ماتریس همبستگی
correlation_matrix = renamed_data[correlation_columns].corr()

# ایجاد ماسک برای نمایش فقط مثلث بالایی
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

# رسم نقشه حرارتی
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    mask=mask,
    annot_kws={"size": 12},
    cbar_kws={"shrink": 0.8},
    square=True,
    linewidths=0
)

plt.grid(False)
plt.title('Correlation Between Main Variables', fontsize=18, fontweight='bold', fontfamily='Times New Roman')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# ---- Creating Time Series ----
# Determine input variables and target based on data
if 'T' in data.columns and 'P' in data.columns and 'qlenj' in data.columns and 'brdstjshan' in data.columns:
    feature_columns = ['T', 'P', 'qlenj', 'brdstjshan']
else:
    # Default variables if specified columns don't exist
    available_numeric = data.select_dtypes(include=[np.number]).columns.tolist()
    available_numeric.remove('oft')  # Remove target variable
    available_numeric.remove('taraz') if 'taraz' in available_numeric else None  # Remove water level if exists
    feature_columns = available_numeric[:4] if len(available_numeric) >= 4 else available_numeric

print(f"\nModel input features: {feature_columns}")

# ---- Split Data into Training and Testing ----
train_end = '2019-07-20'  # Same as original code

x = data2[feature_columns]
y = data2['oft']

train_mask = data2['dateMiladi'] <= train_end
test_mask = data2['dateMiladi'] > train_end

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
plt.title('Distribution of Decline Values in Training and Testing Sets', fontsize=14)
plt.xlabel('Decline (meters)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
sns.kdeplot(ytrain, label='Training', color='blue')
sns.kdeplot(ytest, label='Testing', color='red')
plt.title('Density Distribution of Decline Values', fontsize=14)
plt.xlabel('Decline (meters)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('train_test_distribution.png', dpi=300, bbox_inches='tight')
plt.show()


# ---- Model Evaluation Function ----
def evaluate_model(model_name, y_true_train, y_pred_train, y_true_test, y_pred_test):
    """Calculate and return evaluation metrics for the model"""
    metrics = {}

    # RMSE - Root Mean Square Error
    metrics['Train RMSE'] = np.sqrt(mean_squared_error(y_true_train, y_pred_train))
    metrics['Test RMSE'] = np.sqrt(mean_squared_error(y_true_test, y_pred_test))

    # R² - Coefficient of Determination
    metrics['Train R²'] = r2_score(y_true_train, y_pred_train)
    metrics['Test R²'] = r2_score(y_true_test, y_pred_test)

    # MAE - Mean Absolute Error
    metrics['Train MAE'] = mean_absolute_error(y_true_train, y_pred_train)
    metrics['Test MAE'] = mean_absolute_error(y_true_test, y_pred_test)

    # MAPE - Mean Absolute Percentage Error
    metrics['Train MAPE'] = mean_absolute_percentage_error(y_true_train, y_pred_train)
    metrics['Test MAPE'] = mean_absolute_percentage_error(y_true_test, y_pred_test)

    # d - Index of Agreement
    metrics['Train d'] = 1 - (np.sum((y_true_train - y_pred_train) ** 2) / np.sum(
        (np.abs(y_true_train - np.mean(y_true_train)) + np.abs(y_pred_train - np.mean(y_true_train))) ** 2))
    metrics['Test d'] = 1 - (np.sum((y_true_test - y_pred_test) ** 2) / np.sum(
        (np.abs(y_true_test - np.mean(y_true_test)) + np.abs(y_pred_test - np.mean(y_true_test))) ** 2))

    # NSE - Nash-Sutcliffe Efficiency
    metrics['Train NSE'] = 1 - (
                np.sum((y_true_train - y_pred_train) ** 2) / np.sum((y_true_train - np.mean(y_true_train)) ** 2))
    metrics['Test NSE'] = 1 - (
                np.sum((y_true_test - y_pred_test) ** 2) / np.sum((y_true_test - np.mean(y_true_test)) ** 2))

    # KGE - Kling-Gupta Efficiency
    # KGE components
    r_train = np.corrcoef(y_true_train, y_pred_train)[0, 1]
    r_test = np.corrcoef(y_true_test, y_pred_test)[0, 1]

    alpha_train = np.std(y_pred_train) / np.std(y_true_train)
    alpha_test = np.std(y_pred_test) / np.std(y_true_test)

    beta_train = np.mean(y_pred_train) / np.mean(y_true_train)
    beta_test = np.mean(y_pred_test) / np.mean(y_true_test)

    metrics['Train KGE'] = 1 - np.sqrt((r_train - 1) ** 2 + (alpha_train - 1) ** 2 + (beta_train - 1) ** 2)
    metrics['Test KGE'] = 1 - np.sqrt((r_test - 1) ** 2 + (alpha_test - 1) ** 2 + (beta_test - 1) ** 2)

    return metrics


# Define dictionary to store results of all models
all_metrics = {}
all_predictions = {}

# ---- 1. SVM Model ----
print("\n--- Support Vector Machine (SVM) Model ---")
svm_model = SVR(kernel='rbf', epsilon=0.39, C=12, gamma=0.4)  # Parameters from original code
svm_model.fit(xtrain, ytrain)

trainsvm = svm_model.predict(xtrain)
testsvm = svm_model.predict(xtest)

# Store predictions
all_predictions['SVM'] = {'train': trainsvm, 'test': testsvm}

# Evaluate model
svm_metrics = evaluate_model('SVM', ytrain, trainsvm, ytest, testsvm)
all_metrics['SVM'] = svm_metrics

# Save outputs
pd.DataFrame(testsvm).to_csv('test_svm.csv', index=False)
pd.DataFrame(trainsvm).to_csv('train_svm.csv', index=False)

# Actual vs Predicted scatter plot for SVM
plt.figure(figsize=(10, 6))
plt.scatter(ytest, testsvm)
plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], 'k--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('SVM: Actual vs Predicted Values for Test Set')
plt.text(0.05, 0.95, f'R² = {svm_metrics["Test R²"]:.4f}\nRMSE = {svm_metrics["Test RMSE"]:.4f}',
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('svm_prediction_scatter.png', dpi=300)
plt.show()

# ---- 2. Random Forest Model ----
print("\n--- Random Forest Model ---")
rf_model = RandomForestRegressor(max_depth=20, n_estimators=6, random_state=42)  # Parameters from original code
rf_model.fit(xtrain, ytrain)

trainrf = rf_model.predict(xtrain)
testrf = rf_model.predict(xtest)

# Store predictions
all_predictions['Random Forest'] = {'train': trainrf, 'test': testrf}

# Evaluate model
rf_metrics = evaluate_model('Random Forest', ytrain, trainrf, ytest, testrf)
all_metrics['Random Forest'] = rf_metrics

# Save outputs
pd.DataFrame(testrf).to_csv('test_rf.csv', index=False)
pd.DataFrame(trainrf).to_csv('train_rf.csv', index=False)

# Actual vs Predicted scatter plot for Random Forest
plt.figure(figsize=(10, 6))
plt.scatter(ytest, testrf)
plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], 'k--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Random Forest: Actual vs Predicted Values for Test Set')
plt.text(0.05, 0.95, f'R² = {rf_metrics["Test R²"]:.4f}\nRMSE = {rf_metrics["Test RMSE"]:.4f}',
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('rf_prediction_scatter.png', dpi=300)
plt.show()

# Feature importance in Random Forest model
feature_importance = pd.DataFrame({
    'Feature': xtrain.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature importance in Random Forest model:")
print(feature_importance)

# Feature importance plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
plt.title('Feature Importance in Random Forest Model', fontsize=16)
plt.xlabel('Relative Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('rf_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# ---- 3. XGBoost Model ----
print("\n--- XGBoost Model ---")
xgb_model = xgb.XGBRegressor(
    n_estimators=10,
    max_depth=40,
    learning_rate=0.35,
    verbosity=0
)
xgb_model.fit(xtrain, ytrain)

trainxg = xgb_model.predict(xtrain)
testxg = xgb_model.predict(xtest)

# Store predictions
all_predictions['XGBoost'] = {'train': trainxg, 'test': testxg}

# Evaluate model
xgb_metrics = evaluate_model('XGBoost', ytrain, trainxg, ytest, testxg)
all_metrics['XGBoost'] = xgb_metrics

# Save outputs
pd.DataFrame(testxg).to_csv('test_xgb.csv', index=False)
pd.DataFrame(trainxg).to_csv('train_xgb.csv', index=False)

# Actual vs Predicted scatter plot for XGBoost
plt.figure(figsize=(10, 6))
plt.scatter(ytest, testxg)
plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], 'k--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('XGBoost: Actual vs Predicted Values for Test Set')
plt.text(0.05, 0.95, f'R² = {xgb_metrics["Test R²"]:.4f}\nRMSE = {xgb_metrics["Test RMSE"]:.4f}',
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('xgb_prediction_scatter.png', dpi=300)
plt.show()

# Feature importance in XGBoost model
xgb_importance = pd.DataFrame({
    'Feature': xtrain.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature importance in XGBoost model:")
print(xgb_importance)

# Feature importance plot for XGBoost
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=xgb_importance, palette='plasma')
plt.title('Feature Importance in XGBoost Model', fontsize=16)
plt.xlabel('Relative Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('xgb_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# ---- 4. SVM Feature Importance with permutation importance ----
r = permutation_importance(svm_model, xtest, ytest, n_repeats=30, random_state=42)
svm_importance = pd.DataFrame({
    'Feature': xtrain.columns,
    'Importance': r.importances_mean
}).sort_values(by='Importance', ascending=False)

print("\nFeature importance in SVM model:")
print(svm_importance)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=svm_importance, palette='magma')
plt.title('Feature Importance in SVM Model', fontsize=16)
plt.xlabel('Relative Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('svm_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# ---- 5. Comparing Feature Importance Across Models ----
all_importance = pd.DataFrame({
    'Feature': xtrain.columns,
    'SVM': svm_importance['Importance'].values,
    'Random Forest': rf_importance['Importance'].values if 'rf_importance' in locals() else feature_importance[
        'Importance'].values,
    'XGBoost': xgb_importance['Importance'].values
})

all_importance_relative = all_importance.copy()
for col in ['SVM', 'Random Forest', 'XGBoost']:
    # Normalize importance to percentage
    all_importance_relative[col] = (all_importance_relative[col] / all_importance_relative[col].sum()) * 100

print("\n=== Feature Importance Across All Models (Percentage) ===")
print(tabulate(all_importance_relative, headers='keys', tablefmt='pretty', floatfmt='.2f'))
all_importance_relative.to_csv('feature_importance.csv')

# Comparative bar chart of feature importance
plt.figure(figsize=(14, 8))
x_pos = np.arange(len(xtrain.columns))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 8))
ax.bar(x_pos - width, all_importance_relative['SVM'], width, label='SVM', color='#1f77b4')
ax.bar(x_pos, all_importance_relative['Random Forest'], width, label='Random Forest', color='#ff7f0e')
ax.bar(x_pos + width, all_importance_relative['XGBoost'], width, label='XGBoost', color='#2ca02c')

ax.set_xticks(x_pos)
ax.set_xticklabels(xtrain.columns)
ax.set_ylabel('Relative Importance (%)', fontsize=14)
ax.set_title('Comparison of Feature Importance Across Different Models', fontsize=18)
ax.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature importance heatmap
importance_heatmap = all_importance_relative.set_index('Feature')
plt.figure(figsize=(10, 6))
sns.heatmap(importance_heatmap, annot=True, cmap='YlGnBu', fmt='.2f')
plt.title('Feature Importance Heatmap Across Different Models', fontsize=18)
plt.tight_layout()
plt.savefig('feature_importance_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# ---- 6. Evaluation Metrics Table for All Models ----
# Convert results to DataFrame for better display
metrics_df = pd.DataFrame(all_metrics).T

# Arrange columns for better display
metrics_columns = ['Train RMSE', 'Test RMSE', 'Train R²', 'Test R²', 'Train MAE', 'Test MAE',
                   'Train MAPE', 'Test MAPE', 'Train d', 'Test d', 'Train NSE', 'Test NSE',
                   'Train KGE', 'Test KGE']
metrics_df = metrics_df[metrics_columns]

# Display table in console
print("\n=== Evaluation Metrics Table for All Models ===")
print(tabulate(metrics_df, headers='keys', tablefmt='pretty', floatfmt='.4f'))

# Save metrics table as CSV file
metrics_df.to_csv('model_evaluation_metrics.csv')

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
plt.title('Evaluation Metrics Table for All Models', fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('evaluation_metrics_table.png', dpi=300, bbox_inches='tight')
plt.show()

# ---- 6. Model Comparison Plot ----
plt.figure(figsize=(14, 10))
plt.plot(data.loc[test_mask].index, ytest, '-o', label='Actual Values', linewidth=3, markersize=8, alpha=0.7)
plt.plot(data.loc[test_mask].index, testsvm, '--', label='SVM', linewidth=2.5)
plt.plot(data.loc[test_mask].index, testrf, '--', label='Random Forest', linewidth=2.5)
plt.plot(data.loc[test_mask].index, testxg, '--', label='XGBoost', linewidth=2.5)
plt.title('Comparison of Different Models in Predicting Water Decline', fontsize=18, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Water Decline (meters)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ---- 7. Scatter Plots of Actual vs Predicted Values ----
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
models = ['SVM', 'Random Forest', 'XGBoost']
predictions = [testsvm, testrf, testxg]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, (model, pred, color) in enumerate(zip(models, predictions, colors)):
    axes[i].scatter(ytest, pred, alpha=0.7, s=50, color=color)
    axes[i].plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], 'k--', lw=2)
    axes[i].set_title(f'{model} Model', fontsize=16)
    axes[i].set_xlabel('Actual Values', fontsize=12)
    axes[i].set_ylabel('Predicted Values', fontsize=12)
    axes[i].grid(True, linestyle='--', alpha=0.7)

    r2 = r2_score(ytest, pred)
    rmse = np.sqrt(mean_squared_error(ytest, pred))
    axes[i].text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}',
                 transform=axes[i].transAxes,
                 fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('prediction_scatter_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# ---- 9. Feature Importance for Random Forest Model ----
rf_importance = pd.DataFrame({
    'Feature': x.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=rf_importance, palette='viridis')
plt.title('Feature Importance in Random Forest Model', fontsize=16)
plt.xlabel('Relative Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('rf_feature_importance_final.png', dpi=300, bbox_inches='tight')
plt.show()

# ---- 10. Feature Importance for XGBoost Model ----
xgb_importance = pd.DataFrame({
    'Feature': x.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=xgb_importance, palette='plasma')
plt.title('Feature Importance in XGBoost Model', fontsize=16)
plt.xlabel('Relative Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('xgb_feature_importance_final.png', dpi=300, bbox_inches='tight')
plt.show()

# ---- 11. Feature Importance for SVM Model ----
from sklearn.inspection import permutation_importance

r = permutation_importance(svm_model, xtest, ytest, n_repeats=30, random_state=42)
svm_importance = pd.DataFrame({
    'Feature': x.columns,
    'Importance': r.importances_mean
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))


