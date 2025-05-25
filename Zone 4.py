# ---- کتابخانه ها ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import os
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
data = pd.read_csv('khamiran-heydar.txt', delimiter='\t', index_col='dateMiladi', parse_dates=True)
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
plt.savefig('water_level_decline_over_time_khamiran.png', dpi=300, bbox_inches='tight')
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
plt.savefig('groundwater_level_trend_khamiran.png', dpi=300, bbox_inches='tight')
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
plt.savefig('monthly_water_decline_violin_khamiran.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Seasonal comparison of groundwater levels with violin plot
plt.figure(figsize=(12, 8))
sns.violinplot(x='season', y='taraz', data=data, palette='Set2',
              inner='box', scale='width', bw=0.2)
plt.title('Seasonal Comparison of Groundwater Levels', fontsize=18, fontweight='bold', fontfamily='Times New Roman')
plt.xlabel('Season', fontsize=14, fontweight='bold', fontfamily='Times New Roman')
plt.ylabel('Groundwater Level (meter)', fontsize=14, fontweight='bold', fontfamily='Times New Roman')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('seasonal_water_level_violin_khamiran.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Correlation heatmap between variables
# Renaming columns for better readability
renamed_data = data.rename(columns={
    'taraz': 'GwL',
    'abyari': 'IV',
    'v': 'GwA',
    'oft': 'GwDL'
})

correlation_columns = ['GwDL', 'GwL', 'T', 'P', 'IV', 'GwA']
plt.figure(figsize=(10, 8))
correlation_matrix = renamed_data[correlation_columns].corr()

# Creating mask for upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

# Enhanced heatmap design
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
plt.savefig('correlation_heatmap_khamiran.png', dpi=300, bbox_inches='tight')
plt.show()

# ---- ساخت سری زمانی ----
seryzamani = data[['T', 'P', 'abyari', 'oft', 'v']]

# ---- تقسیم داده به آموزش و تست ----
train_end = '2018-07-20'
test_start = '2018-08-20'

x = seryzamani[['T', 'P', 'abyari', 'v']]
y = seryzamani['oft']

train_mask = data.index <= train_end
test_mask = data.index >= test_start

xtrain = x[train_mask]
ytrain = y[train_mask]
xtest = x[test_mask]
ytest = y[test_mask]

print(f"داده‌های آموزش: {len(xtrain)} نمونه")
print(f"داده‌های آزمون: {len(xtest)} نمونه")

# نمودار توزیع داده‌های آموزش و آزمون
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.hist(ytrain, bins=15, alpha=0.7, color='blue', label='آموزش')
plt.hist(ytest, bins=15, alpha=0.7, color='red', label='آزمون')
plt.title('توزیع مقادیر افت در مجموعه آموزش و آزمون', fontsize=14)
plt.xlabel('افت (متر)', fontsize=12)
plt.ylabel('فراوانی', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
sns.kdeplot(ytrain, label='آموزش', color='blue')
sns.kdeplot(ytest, label='آزمون', color='red')
plt.title('توزیع چگالی مقادیر افت', fontsize=14)
plt.xlabel('افت (متر)', fontsize=12)
plt.ylabel('چگالی', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('train_test_distribution_khamiran.png', dpi=300, bbox_inches='tight')
plt.show()


# ---- تابع ارزیابی مدل ----
def evaluate_model(model_name, y_true_train, y_pred_train, y_true_test, y_pred_test):
    """محاسبه و بازگرداندن شاخص‌های ارزیابی برای مدل"""
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


# تعریف دیکشنری برای ذخیره نتایج همه مدل‌ها
all_metrics = {}
all_predictions = {}

# ---- مدل SVM ----
print("\n--- مدل ماشین بردار پشتیبان (SVM) ---")
svm_model = SVR(kernel='rbf', epsilon=0.07, C=22, gamma=0.4)
svm_model.fit(xtrain, ytrain)

trainsvm = svm_model.predict(xtrain)
testsvm = svm_model.predict(xtest)

# ذخیره پیش‌بینی‌ها
all_predictions['SVM'] = {'train': trainsvm, 'test': testsvm}

# ارزیابی مدل
svm_metrics = evaluate_model('SVM', ytrain, trainsvm, ytest, testsvm)
all_metrics['SVM'] = svm_metrics

# ذخیره خروجی‌ها
pd.DataFrame(testsvm).to_csv('test2-svm-khamiran.csv', index=False)
pd.DataFrame(trainsvm).to_csv('train3-svm-khamiran.csv', index=False)

# ---- مدل Random Forest ----
print("\n--- مدل جنگل تصادفی (Random Forest) ---")
rf_model = RandomForestRegressor(max_leaf_nodes=40, n_estimators=4, random_state=42)
rf_model.fit(xtrain, ytrain)

trainrf = rf_model.predict(xtrain)
testrf = rf_model.predict(xtest)

# ذخیره پیش‌بینی‌ها
all_predictions['Random Forest'] = {'train': trainrf, 'test': testrf}

# ارزیابی مدل
rf_metrics = evaluate_model('Random Forest', ytrain, trainrf, ytest, testrf)
all_metrics['Random Forest'] = rf_metrics

# ذخیره خروجی‌ها
pd.DataFrame(testrf).to_csv('test2-rf-khamiran.csv', index=False)
pd.DataFrame(trainrf).to_csv('train2-rf-khamiran.csv', index=False)

# ---- مدل XGBoost ----
print("\n--- مدل XGBoost ---")
xgb_model = xgb.XGBRegressor(
    n_estimators=10,
    max_depth=40,
    learning_rate=0.35,
    n_jobs=5,
    verbosity=0
)
xgb_model.fit(xtrain, ytrain)

trainxg = xgb_model.predict(xtrain)
testxg = xgb_model.predict(xtest)

# ذخیره پیش‌بینی‌ها
all_predictions['XGBoost'] = {'train': trainxg, 'test': testxg}

# ارزیابی مدل
xgb_metrics = evaluate_model('XGBoost', ytrain, trainxg, ytest, testxg)
all_metrics['XGBoost'] = xgb_metrics

# ذخیره خروجی‌ها
pd.DataFrame(testxg).to_csv('test2-xg-khamiran.csv', index=False)
pd.DataFrame(trainxg).to_csv('train3-xg-khamiran.csv', index=False)

# ---- 5. نمایش جدول شاخص‌های ارزیابی برای هر مدل ----
# تبدیل نتایج به دیتافریم برای نمایش بهتر
metrics_df = pd.DataFrame(all_metrics).T

# تنظیم ستون‌ها برای نمایش بهتر
metrics_columns = ['Train RMSE', 'Test RMSE', 'Train R²', 'Test R²', 'Train MAE', 'Test MAE',
                   'Train MAPE', 'Test MAPE', 'Train d', 'Test d']
metrics_df = metrics_df[metrics_columns]

# نمایش جدول در کنسول
print("\n=== جدول شاخص‌های ارزیابی برای همه مدل‌ها ===")
print(tabulate(metrics_df, headers='keys', tablefmt='pretty', floatfmt='.4f'))

# ذخیره جدول شاخص‌ها به عنوان فایل CSV
metrics_df.to_csv('model_evaluation_metrics_khamiran.csv')

# نمایش جدول به صورت بصری
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
plt.title('جدول شاخص‌های ارزیابی برای همه مدل‌ها', fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('evaluation_metrics_table_khamiran.png', dpi=300, bbox_inches='tight')
plt.show()

# ---- 6. نمودار مقایسه مدل‌ها ----
plt.figure(figsize=(14, 10))
plt.plot(data.loc[test_mask].index, ytest, '-o', label='مقادیر واقعی', linewidth=3, markersize=8, alpha=0.7)
plt.plot(data.loc[test_mask].index, testsvm, '--', label='SVM', linewidth=2.5)
plt.plot(data.loc[test_mask].index, testrf, '--', label='Random Forest', linewidth=2.5)
plt.plot(data.loc[test_mask].index, testxg, '--', label='XGBoost', linewidth=2.5)
plt.title('مقایسه عملکرد مدل‌های مختلف در پیش‌بینی افت آب', fontsize=18, fontweight='bold')
plt.xlabel('تاریخ', fontsize=14)
plt.ylabel('افت آب (متر)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('model_comparison_khamiran.png', dpi=300, bbox_inches='tight')
plt.show()

# ---- 7. نمودار پراکندگی مقادیر واقعی و پیش‌بینی شده ----
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
models = ['SVM', 'Random Forest', 'XGBoost']
predictions = [testsvm, testrf, testxg]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, (model, pred, color) in enumerate(zip(models, predictions, colors)):
    axes[i].scatter(ytest, pred, alpha=0.7, s=50, color=color)
    axes[i].plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], 'k--', lw=2)
    axes[i].set_title(f'مدل {model}', fontsize=16)
    axes[i].set_xlabel('مقادیر واقعی', fontsize=12)
    axes[i].set_ylabel('مقادیر پیش‌بینی شده', fontsize=12)
    axes[i].grid(True, linestyle='--', alpha=0.7)

    r2 = r2_score(ytest, pred)
    rmse = np.sqrt(mean_squared_error(ytest, pred))
    axes[i].text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}',
                 transform=axes[i].transAxes,
                 fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('prediction_scatter_plots_khamiran.png', dpi=300, bbox_inches='tight')
plt.show()

# ---- 9. اهمیت ویژگی‌ها برای مدل جنگل تصادفی ----
rf_importance = pd.DataFrame({
    'Feature': x.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=rf_importance, palette='viridis')
plt.title('اهمیت ویژگی‌ها در مدل جنگل تصادفی', fontsize=16)
plt.xlabel('اهمیت نسبی', fontsize=14)
plt.ylabel('ویژگی', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('rf_feature_importance_khamiran.png', dpi=300, bbox_inches='tight')
plt.show()

# ---- 10. اهمیت ویژگی‌ها برای مدل XGBoost ----
xgb_importance = pd.DataFrame({
    'Feature': x.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=xgb_importance, palette='plasma')
plt.title('اهمیت ویژگی‌ها در مدل XGBoost', fontsize=16)
plt.xlabel('اهمیت نسبی', fontsize=14)
plt.ylabel('ویژگی', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('xgb_feature_importance_khamiran.png', dpi=300, bbox_inches='tight')
plt.show()

# ---- 11. اهمیت ویژگی‌ها برای مدل SVM ----
from sklearn.inspection import permutation_importance

r = permutation_importance(svm_model, xtest, ytest, n_repeats=30, random_state=42)
svm_importance = pd.DataFrame({
    'Feature': x.columns,
    'Importance': r.importances_mean
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=svm_importance, palette='magma')
plt.title('اهمیت ویژگی‌ها در مدل SVM', fontsize=16)
plt.xlabel('اهمیت نسبی', fontsize=14)
plt.ylabel('ویژگی', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('svm_feature_importance_khamiran.png', dpi=300, bbox_inches='tight')
plt.show()

# ---- 12. مقایسه اهمیت ویژگی‌ها بین مدل‌ها ----
all_importance = pd.DataFrame({
    'Feature': x.columns,
    'SVM': svm_importance['Importance'].values,
    'Random Forest': rf_importance['Importance'].values,
    'XGBoost': xgb_importance['Importance'].values
})

all_importance_relative = all_importance.copy()
for col in ['SVM', 'Random Forest', 'XGBoost']:
    # نرمال‌سازی اهمیت‌ها به درصد
    all_importance_relative[col] = (all_importance_relative[col] / all_importance_relative[col].sum()) * 100

print("\n=== اهمیت ویژگی‌ها برای همه مدل‌ها (درصد) ===")
print(tabulate(all_importance_relative, headers='keys', tablefmt='pretty', floatfmt='.2f'))
all_importance_relative.to_csv('feature_importance_khamiran.csv')

# نمودار میله‌ای مقایسه‌ای اهمیت ویژگی‌ها
plt.figure(figsize=(14, 8))
x_pos = np.arange(len(x.columns))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 8))
ax.bar(x_pos - width, all_importance_relative['SVM'], width, label='SVM', color='#1f77b4')
ax.bar(x_pos, all_importance_relative['Random Forest'], width, label='Random Forest', color='#ff7f0e')
ax.bar(x_pos + width, all_importance_relative['XGBoost'], width, label='XGBoost', color='#2ca02c')

ax.set_xticks(x_pos)
ax.set_xticklabels(x.columns)
ax.set_ylabel('اهمیت نسبی (%)', fontsize=14)
ax.set_title('مقایسه اهمیت ویژگی‌ها در مدل‌های مختلف', fontsize=18)
ax.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('feature_importance_comparison_khamiran.png', dpi=300, bbox_inches='tight')
plt.show()

# نمودار حرارتی اهمیت ویژگی‌ها
importance_heatmap = all_importance_relative.set_index('Feature')
plt.figure(figsize=(10, 6))
sns.heatmap(importance_heatmap, annot=True, cmap='YlGnBu', fmt='.2f')
plt.title('نمودار حرارتی اهمیت ویژگی‌ها در مدل‌های مختلف', fontsize=18)
plt.tight_layout()
plt.savefig('feature_importance_heatmap_khamiran.png', dpi=300, bbox_inches='tight')
plt.show()

# ---- 13. بهترین مدل‌ها بر اساس معیارهای مختلف ----
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

print("\n=== خلاصه نتایج: بهترین مدل‌ها بر اساس معیارهای مختلف ===")
print(tabulate(best_models, headers='keys', tablefmt='pretty', floatfmt='.4f'))

# نمایش بصری نتایج خلاصه
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
plt.title('خلاصه نتایج: بهترین مدل‌ها بر اساس معیارهای مختلف', fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('results_summary_khamiran.png', dpi=300, bbox_inches='tight')
plt.show()

# --- اصلاح شده: حذف بخش‌های 8 و 14 که مربوط به مقایسه پیش‌بینی آینده بود ---
# بخش 8 به طور کامل حذف شده (نمودار مقایسه پیش‌بینی‌های مدل‌های مختلف برای داده‌های جدید)

# بخش اصلاح شده برای نمودار مقایسه شاخص‌های ارزیابی (بخش 14 در کد اصلی)
# نیاز به بازنویسی کامل بخش دارد چون در کد اصلی دارای خطا بود

# مقایسه RMSE بین مدل‌ها
rmse_data = pd.DataFrame({
    'Model': ['SVM', 'Random Forest', 'XGBoost'],
    'Train RMSE': [all_metrics['SVM']['Train RMSE'],
                   all_metrics['Random Forest']['Train RMSE'],
                   all_metrics['XGBoost']['Train RMSE']],
    'Test RMSE': [all_metrics['SVM']['Test RMSE'],
                  all_metrics['Random Forest']['Test RMSE'],
                  all_metrics['XGBoost']['Test RMSE']]
})

# تبدیل به فرمت مناسب

print("\n=== خلاصه نتایج: بهترین مدل‌ها بر اساس معیارهای مختلف ===")
print(tabulate(best_models, headers='keys', tablefmt='pretty', floatfmt='.4f'))

# نمایش بصری نتایج خلاصه
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
plt.title('خلاصه نتایج: بهترین مدل‌ها بر اساس معیارهای مختلف', fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('results_summary.png', dpi=300, bbox_inches='tight')
plt.show()