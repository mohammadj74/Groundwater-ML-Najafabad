import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import warnings
from matplotlib import rcParams
from tabulate import tabulate
import matplotlib.gridspec as gridspec

# Set Times New Roman font for all plots
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'

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

# --- 1. Read Data ---
data = pd.read_csv('vorodi2-jonob.txt', delimiter='\t', index_col='dateMiladi', parse_dates=True)

# --- 2. Preprocessing ---
data.index = pd.to_datetime(data.index)
data['year1'] = data.index.year
data['month1'] = data.index.month
data['day1'] = data.index.day
data['date_formatted'] = data.index.strftime('%Y-%m')  # For use in plots

# --- 3. Improved Visualizations ---

# 1. Groundwater decline over time with better coloring and more details
plt.figure(figsize=(14, 8))
sns.lineplot(x=data.index, y=data['oft'], linewidth=2.5, color='#1f77b4')
plt.fill_between(data.index, data['oft'], alpha=0.3, color='#1f77b4')
plt.title('Groundwater Decline Changes Over Time', fontsize=18, fontweight='bold', fontname='Times New Roman')
plt.xlabel('Date', fontsize=14,fontweight='bold', fontname='Times New Roman')
plt.ylabel('Groundwater Level Decline (m)', fontsize=14, fontweight='bold', fontname='Times New Roman')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('water_level_decline_over_time.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Groundwater level trend with confidence interval
plt.figure(figsize=(14, 8))
sns.regplot(x=data['year1'], y=data['taraz'],
            scatter_kws={'alpha': 0.5, 's': 30, 'color': '#2ca02c'},
            line_kws={'color': 'red', 'linewidth': 2})
plt.title('Groundwater Level Trend', fontsize=18, fontweight='bold', fontname='Times New Roman')
plt.xlabel('Year', fontsize=14, fontweight='bold', fontname='Times New Roman')
plt.ylabel('Groundwater Level (m)', fontsize=14, fontweight='bold', fontname='Times New Roman')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('groundwater_level_trend.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Improved boxplot for water decline by month
plt.figure(figsize=(14, 8))
ax = sns.violinplot(x='month1', y='oft', data=data, palette='viridis', inner='box')
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticklabels(months)
plt.title('Groundwater Decline Changes in Different Months', fontsize=18, fontweight='bold', fontname='Times New Roman')
plt.xlabel('Month', fontsize=14, fontweight='bold',fontname='Times New Roman')
plt.ylabel('Groundwater Level Decline (m)', fontsize=14, fontweight='bold', fontname='Times New Roman')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('monthly_water_decline_violinplot.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Correlation heatmap between variables
# تغییر نام ستون‌ها
renamed_data = data.rename(columns={
    'taraz': 'GwL',
    'oft': 'GwDL'  # اگر oft متغیر دیگری باشد، این خط را حذف یا اصلاح کن
})

# انتخاب ستون‌ها برای ماتریس همبستگی
correlation_columns = ['GwDL', 'GwL', 'T', 'P']
correlation_matrix = renamed_data[correlation_columns].corr()

# ایجاد ماسک برای نیمه بالایی ماتریس
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

# ترسیم نقشه حرارتی
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


# 5. Seasonal comparison of groundwater level
data['season'] = pd.cut(data['month1'],
                        bins=[0, 3, 6, 9, 12],
                        labels=['Winter', 'Spring', 'Summer', 'Fall'])
plt.figure(figsize=(12, 8))
sns.violinplot(x='season', y='taraz', data=data, palette='Set2', inner='box')
plt.title('Seasonal Comparison of Groundwater Level', fontsize=18, fontweight='bold', fontname='Times New Roman')
plt.xlabel('Season', fontsize=14, fontweight='bold', fontname='Times New Roman')
plt.ylabel('Groundwater Level (m)', fontsize=14, fontweight='bold', fontname='Times New Roman')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('seasonal_water_level_violinplot.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 4. Modeling and Evaluation ---
x = data[['T', 'P']]  # Input variables
y = data['oft']  # Target variable

# Split data into training and testing
train_mask = data.index <= '2018-07-20'
test_mask = data.index >= '2018-08-20'

xtrain, xtest = x[train_mask], x[test_mask]
ytrain, ytest = y[train_mask], y[test_mask]


# --- Write evaluation function for reuse ---
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


# Define dictionary to store results for all models
all_metrics = {}
all_predictions = {}

# --- 1. SVM Model ---
print("\n--- Support Vector Machine (SVM) Model ---")
svm_reg = SVR(kernel='rbf', C=22, epsilon=0.28, gamma=0.4)
svm_reg.fit(xtrain, ytrain)
trainsvm = svm_reg.predict(xtrain)
testsvm = svm_reg.predict(xtest)

# Store predictions
all_predictions['SVM'] = {'train': trainsvm, 'test': testsvm}

# Evaluate model
svm_metrics = evaluate_model('SVM', ytrain, trainsvm, ytest, testsvm)
all_metrics['SVM'] = svm_metrics

# --- 2. Random Forest Model ---
print("\n--- Random Forest Model ---")
rf_reg = RandomForestRegressor(max_leaf_nodes=20, n_estimators=4, random_state=42)
rf_reg.fit(xtrain, ytrain)
trainrf = rf_reg.predict(xtrain)
testrf = rf_reg.predict(xtest)

# Store predictions
all_predictions['Random Forest'] = {'train': trainrf, 'test': testrf}

# Evaluate model
rf_metrics = evaluate_model('Random Forest', ytrain, trainrf, ytest, testrf)
all_metrics['Random Forest'] = rf_metrics

# --- 3. XGBoost Model ---
print("\n--- XGBoost Model ---")
xg_reg = XGBRegressor(n_estimators=10, max_depth=20, eta=0.35, n_jobs=5)
xg_reg.fit(xtrain, ytrain)
trainxg = xg_reg.predict(xtrain)
testxg = xg_reg.predict(xtest)

# Store predictions
all_predictions['XGBoost'] = {'train': trainxg, 'test': testxg}

# Evaluate model
xg_metrics = evaluate_model('XGBoost', ytrain, trainxg, ytest, testxg)
all_metrics['XGBoost'] = xg_metrics

# --- 5. Display evaluation metrics table for each model ---

# Convert results to dataframe for better display
metrics_df = pd.DataFrame(all_metrics).T

# Set columns for better display
metrics_columns = ['Train RMSE', 'Test RMSE', 'Train R²', 'Test R²', 'Train MAE', 'Test MAE',
                   'Train MAPE', 'Test MAPE', 'Train d', 'Test d']
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
plt.title('Evaluation Metrics Table for All Models', fontsize=18, fontweight='bold',
          fontname='Times New Roman', pad=20)
plt.tight_layout()
plt.savefig('evaluation_metrics_table.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 6. Models comparison chart ---
plt.figure(figsize=(14, 10))
plt.plot(data.loc[test_mask].index, ytest, '-o', label='Actual Values', linewidth=3, markersize=8, alpha=0.7)
plt.plot(data.loc[test_mask].index, testsvm, '--', label='SVM', linewidth=2.5)
plt.plot(data.loc[test_mask].index, testrf, '--', label='Random Forest', linewidth=2.5)
plt.plot(data.loc[test_mask].index, testxg, '--', label='XGBoost', linewidth=2.5)
plt.title('Performance Comparison of Different Models in Water Decline Prediction',
          fontsize=18, fontweight='bold', fontname='Times New Roman')
plt.xlabel('Date', fontsize=14, fontname='Times New Roman')
plt.ylabel('Water Decline (meters)', fontsize=14, fontname='Times New Roman')
plt.legend(fontsize=12, prop={'family': 'Times New Roman'})
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 7. Feature importance for all three models ---

# 1. Feature importance for Random Forest model
rf_importance = pd.DataFrame({
    'Feature': x.columns,
    'Importance': rf_reg.feature_importances_
}).sort_values(by='Importance', ascending=False)

# 2. Feature importance for XGBoost model
xgb_importance = pd.DataFrame({
    'Feature': x.columns,
    'Importance': xg_reg.feature_importances_
}).sort_values(by='Importance', ascending=False)

# 3. For SVM we need coefficient calculation
# This is an approximate method, as SVM with RBF kernel doesn't directly provide feature importances
# We use relative coefficient importance
from sklearn.inspection import permutation_importance

# Calculate feature importance for SVM
r = permutation_importance(svm_reg, xtest, ytest, n_repeats=30, random_state=42)
svm_importance = pd.DataFrame({
    'Feature': x.columns,
    'Importance': r.importances_mean
}).sort_values(by='Importance', ascending=False)

# Dataframe to store feature importance for all models
all_importance = pd.DataFrame({
    'Feature': x.columns,
    'SVM': svm_importance['Importance'].values,
    'Random Forest': rf_importance['Importance'].values,
    'XGBoost': xgb_importance['Importance'].values
})

# Display feature importance table in console
print("\n=== Feature Importance for All Models ===")
all_importance_relative = all_importance.copy()
for col in ['SVM', 'Random Forest', 'XGBoost']:
    # Normalize importances to percentage
    all_importance_relative[col] = (all_importance_relative[col] / all_importance_relative[col].sum()) * 100

print(tabulate(all_importance_relative, headers='keys', tablefmt='pretty', floatfmt='.2f'))
all_importance_relative.to_csv('feature_importance.csv')

# Display feature importance chart for all models
# 1. Bar chart
# تغییر ترتیب متغیرها: اول P، بعد T، سپس بقیه
desired_order = ['P', 'T']  # ترتیب دلخواه
all_importance_relative = all_importance_relative.set_index('Feature').loc[desired_order].reset_index()

# حذف fig اضافی
# حذف figure اضافی
fig, ax = plt.subplots(figsize=(14, 8))

x_pos = np.arange(len(all_importance_relative['Feature']))
width = 0.009  # کاهش عرض ستون‌ها

# رسم بارها با رنگ‌های مشخص شده در کد قبلی
ax.bar(x_pos - width, all_importance_relative['SVM'], width, label='SVM', color='#800080')        # بنفش
ax.bar(x_pos, all_importance_relative['Random Forest'], width, label='Random Forest', color='#B76E79')  # رز گلد
ax.bar(x_pos + width, all_importance_relative['XGBoost'], width, label='XGBoost', color='#6B8E23')     # سبز کله‌غازی

# تنظیمات محور X
ax.set_xticks(x_pos)
ax.set_xticklabels(all_importance_relative['Feature'], fontname='Times New Roman')

# عنوان‌ها و برچسب‌ها
ax.set_ylabel('Relative Importance (%)', fontsize=14, fontname='Times New Roman')
ax.set_title('Feature Importance Comparison Across Different Models', fontsize=18, fontname='Times New Roman')

# تنظیم راهنما
ax.legend(prop={'family': 'Times New Roman'})

# افزودن خطوط راهنما
plt.grid(False)

# حذف نمایش مقادیر روی ستون‌ها و تنظیم محور Y

plt.tight_layout()
plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Feature importance heatmap
importance_heatmap = all_importance_relative.set_index('Feature')
plt.figure(figsize=(10, 6))
sns.heatmap(importance_heatmap, annot=True, cmap='YlGnBu', fmt='.2f')
plt.title('Feature Importance Heatmap Across Different Models',
          fontsize=18, fontname='Times New Roman')
plt.tight_layout()
plt.savefig('feature_importance_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 8. Evaluation metrics comparison charts ---

# Select important metrics to display
key_metrics = ['RMSE', 'MAE', 'R²', 'MAPE', 'd']

# Create comparison charts for training and testing
fig = plt.figure(figsize=(20, 15))
gs = gridspec.GridSpec(3, 2)

# Convert dataframe to suitable format for plotting
train_metrics = pd.DataFrame({
    'Model': [],
    'Metric': [],
    'Value': []
})

test_metrics = pd.DataFrame({
    'Model': [],
    'Metric': [],
    'Value': []
})

for model in all_metrics:
    for metric in key_metrics:
        # Add training values
        train_metrics = pd.concat([train_metrics, pd.DataFrame({
            'Model': [model],
            'Metric': [metric],
            'Value': [all_metrics[model][f'Train {metric}']]
        })])

        # Add testing values
        test_metrics = pd.concat([test_metrics, pd.DataFrame({
            'Model': [model],
            'Metric': [metric],
            'Value': [all_metrics[model][f'Test {metric}']]
        })])

# Training metrics chart
ax1 = plt.subplot(gs[0, :])
sns.barplot(x='Metric', y='Value', hue='Model', data=train_metrics, palette='Set1', ax=ax1)
ax1.set_title('Evaluation Metrics Comparison in Training Set',
              fontsize=16, fontname='Times New Roman')
ax1.set_xlabel('Evaluation Metric', fontsize=14, fontname='Times New Roman')
ax1.set_ylabel('Value', fontsize=14, fontname='Times New Roman')
ax1.legend(title='Model', prop={'family': 'Times New Roman'})
ax1.grid(True, linestyle='--', alpha=0.7)

# Testing metrics chart
ax2 = plt.subplot(gs[1, :])
sns.barplot(x='Metric', y='Value', hue='Model', data=test_metrics, palette='Set1', ax=ax2)
ax2.set_title('Evaluation Metrics Comparison in Test Set',
              fontsize=16, fontname='Times New Roman')
ax2.set_xlabel('Evaluation Metric', fontsize=14, fontname='Times New Roman')
ax2.set_ylabel('Value', fontsize=14, fontname='Times New Roman')
ax2.legend(title='Model', prop={'family': 'Times New Roman'})
ax2.grid(True, linestyle='--', alpha=0.7)

# Training and testing comparison chart for R² metric
r2_data = pd.DataFrame({
    'Model': [],
    'Dataset': [],
    'R²': []
})

for model in all_metrics:
    r2_data = pd.concat([r2_data, pd.DataFrame({
        'Model': [model, model],
        'Dataset': ['Train', 'Test'],
        'R²': [all_metrics[model]['Train R²'], all_metrics[model]['Test R²']]
    })])

ax3 = plt.subplot(gs[2, 0])
sns.barplot(x='Model', y='R²', hue='Dataset', data=r2_data, palette=['#4CAF50', '#F44336'], ax=ax3)
ax3.set_title('R² Comparison Between Training and Test Sets',
              fontsize=16, fontname='Times New Roman')
ax3.set_xlabel('Model', fontsize=14, fontname='Times New Roman')
ax3.set_ylabel('R²', fontsize=14, fontname='Times New Roman')
ax3.legend(title='Dataset', prop={'family': 'Times New Roman'})
ax3.grid(True, linestyle='--', alpha=0.7)

# Training and testing comparison chart for RMSE metric
rmse_data = pd.DataFrame({
    'Model': [],
    'Dataset': [],
    'RMSE': []
})

for model in all_metrics:
    rmse_data = pd.concat([rmse_data, pd.DataFrame({
        'Model': [model, model],
        'Dataset': ['Train', 'Test'],
        'RMSE': [all_metrics[model]['Train RMSE'], all_metrics[model]['Test RMSE']]
    })])

ax4 = plt.subplot(gs[2, 1])
sns.barplot(x='Model', y='RMSE', hue='Dataset', data=rmse_data, palette=['#4CAF50', '#F44336'], ax=ax4)
ax4.set_title('RMSE Comparison Between Training and Test Sets',
              fontsize=16, fontname='Times New Roman')
ax4.set_xlabel('Model', fontsize=14, fontname='Times New Roman')
ax4.set_ylabel('RMSE', fontsize=14, fontname='Times New Roman')
ax4.legend(title='Dataset', prop={'family': 'Times New Roman'})
ax4.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('evaluation_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 9. Results Summary ---
# Display best model based on different metrics
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

print("\n=== Results Summary: Best Models Based on Different Metrics ===")
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
plt.title('Results Summary: Best Models Based on Different Metrics',
          fontsize=18, fontweight='bold', fontname='Times New Roman', pad=20)
plt.tight_layout()
plt.savefig('results_summary.png', dpi=300, bbox_inches='tight')
plt.show()