# EDA and Statistics Summary

### Test 1: Promotion Lift
- **Business Question:** Do promotions significantly increase units sold?
- **$H_0$:** Mean demand during promotions = Mean demand during non-promotions.
- **$H_1$:** Mean demand during promotions != Mean demand during non-promotions.
- **Test Used:** Welch's T-test
- **p-value:** 0.3312
- **Business Interpretation:** Not significant difference in demand due to promotions.

### Test 2: Store Type Demand
- **Business Question:** Does mean demand differ across regions?
- **$H_0$:** Mean demand is the same across all regions.
- **$H_1$:** At least one region has a different mean demand.
- **Test Used:** One-way ANOVA
- **p-value:** 0.6033
- **Business Interpretation:** Not significant difference in demand across regions.

### Test 3: Stock-out vs Promotion
- **Business Question:** Is stock-out frequency associated with promotion status?
- **$H_0$:** Stock-outs and promotion status are independent.
- **$H_1$:** Stock-outs and promotion status are associated.
- **Test Used:** Chi-Square Test of Independence
- **p-value:** 1.0000
- **Business Interpretation:** Not significant association between promotions and stock-outs.

