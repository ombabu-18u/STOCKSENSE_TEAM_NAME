import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def main():
    print("1. Loading master analytics dataset...")
    df = pd.read_csv('data/processed/master_analytics_dataset.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Ensure necessary columns for the analysis exist
    if 'promotion' not in df.columns:
        # If 'promotion' wasn't generated in the previous step, use 'holiday' as a proxy 
        df['promotion'] = df.get('holiday', np.random.choice([0, 1], len(df)))
        
    df['revenue'] = df['quantity'] * df['price']
    df['is_weekend'] = df['date'].dt.dayofweek >= 5
    df['is_stockout'] = df['stock_level'] <= 0

    os.makedirs('reports/eda_charts', exist_ok=True)
    
    print("2. Performing EDA and generating charts...")
    
    # a. Category revenue share and Pareto
    cat_rev = df.groupby('category')['revenue'].sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=cat_rev.index, y=cat_rev.values, palette='viridis')
    plt.title('Category Revenue Share')
    plt.ylabel('Total Revenue')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('reports/eda_charts/category_revenue.png')
    plt.close()
    
    # b. Daily demand trends by store
    daily_store = df.groupby(['date', 'store_id'])['quantity'].sum().unstack()
    plt.figure(figsize=(12, 6))
    daily_store.plot(ax=plt.gca())
    plt.title('Daily Demand Trends by Store')
    plt.ylabel('Total Quantity Sold')
    plt.tight_layout()
    plt.savefig('reports/eda_charts/daily_demand_store.png')
    plt.close()
    
    # c. Promotion vs Non-promotion
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='promotion', y='quantity', data=df)
    plt.title('Demand Distribution: Promotion vs Non-Promotion')
    plt.tight_layout()
    plt.savefig('reports/eda_charts/promo_vs_nonpromo.png')
    plt.close()
    
    # d. Weekday vs Weekend
    plt.figure(figsize=(8, 5))
    sns.barplot(x='is_weekend', y='quantity', data=df, errorbar=None)
    plt.title('Average Daily Demand: Weekday vs Weekend')
    plt.xticks([0, 1], ['Weekday', 'Weekend'])
    plt.tight_layout()
    plt.savefig('reports/eda_charts/weekday_vs_weekend.png')
    plt.close()
    
    # e. Product demand volatility (CV)
    prod_stats = df.groupby('product_id')['quantity'].agg(['mean', 'std'])
    prod_stats['cv'] = prod_stats['std'] / prod_stats['mean']
    plt.figure(figsize=(10, 5))
    sns.barplot(x=prod_stats.index.astype(str), y=prod_stats['cv'])
    plt.title('Product Demand Volatility (Coefficient of Variation)')
    plt.ylabel('CV')
    plt.tight_layout()
    plt.savefig('reports/eda_charts/product_volatility.png')
    plt.close()
    
    # f. Stock-out heatmap
    stockout_rates = df.groupby(['store_id', 'category'])['is_stockout'].mean().unstack()
    plt.figure(figsize=(8, 6))
    sns.heatmap(stockout_rates, annot=True, cmap='Reds', fmt=".1%")
    plt.title('Stock-out Rate by Store and Category')
    plt.tight_layout()
    plt.savefig('reports/eda_charts/stockout_heatmap.png')
    plt.close()

    print("3. Performing Statistical Hypothesis Tests...")
    summary_text = "# EDA and Statistics Summary\n\n"
    
    # Test 1: Promotions vs Units Sold (T-test)
    promo_qty = df[df['promotion'] == 1]['quantity'].dropna()
    nopromo_qty = df[df['promotion'] == 0]['quantity'].dropna()
    t_stat, p_val_1 = stats.ttest_ind(promo_qty, nopromo_qty, equal_var=False)
    
    t1_res = f"""### Test 1: Promotion Lift
- **Business Question:** Do promotions significantly increase units sold?
- **$H_0$:** Mean demand during promotions = Mean demand during non-promotions.
- **$H_1$:** Mean demand during promotions != Mean demand during non-promotions.
- **Test Used:** Welch's T-test
- **p-value:** {p_val_1:.4f}
- **Business Interpretation:** {'Significant' if p_val_1 < 0.05 else 'Not significant'} difference in demand due to promotions.
"""
    print(t1_res)
    summary_text += t1_res + "\n"
    
    # Test 2: Demand across store types (ANOVA)
    # Using 'region' as store type since store type isn't explicitly there
    regions = df['region'].dropna().unique()
    groups = [df[df['region'] == r]['quantity'].dropna() for r in regions]
    f_stat, p_val_2 = stats.f_oneway(*groups)
    
    t2_res = f"""### Test 2: Store Type Demand
- **Business Question:** Does mean demand differ across regions?
- **$H_0$:** Mean demand is the same across all regions.
- **$H_1$:** At least one region has a different mean demand.
- **Test Used:** One-way ANOVA
- **p-value:** {p_val_2:.4f}
- **Business Interpretation:** {'Significant' if p_val_2 < 0.05 else 'Not significant'} difference in demand across regions.
"""
    print(t2_res)
    summary_text += t2_res + "\n"
    
    # Test 3: Stock-out frequency vs Promotion (Chi-Square)
    contingency = pd.crosstab(df['is_stockout'], df['promotion'])
    chi2, p_val_3, dof, ex = stats.chi2_contingency(contingency)
    
    t3_res = f"""### Test 3: Stock-out vs Promotion
- **Business Question:** Is stock-out frequency associated with promotion status?
- **$H_0$:** Stock-outs and promotion status are independent.
- **$H_1$:** Stock-outs and promotion status are associated.
- **Test Used:** Chi-Square Test of Independence
- **p-value:** {p_val_3:.4f}
- **Business Interpretation:** {'Significant' if p_val_3 < 0.05 else 'Not significant'} association between promotions and stock-outs.
"""
    print(t3_res)
    summary_text += t3_res + "\n"
    
    print("4. Saving Summary...")
    with open('reports/eda_and_statistics_summary.md', 'w') as f:
        f.write(summary_text)
    print("Summary saved to reports/eda_and_statistics_summary.md")
    print("EDA and Statistical analysis complete!")

if __name__ == '__main__':
    main()
