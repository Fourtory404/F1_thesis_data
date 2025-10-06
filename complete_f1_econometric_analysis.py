"""
Complete F1 Budget Cap Econometric Analysis
Implements full methodology with actual budget data
Author: F1 Thesis Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import warnings
import os
warnings.filterwarnings('ignore')

class F1EconometricAnalysis:
    """Complete econometric analysis implementation"""
    
    def __init__(self, df, output_dir='output'):
        self.df = df.copy()
        self.results = {}
        self.budget_var = None
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
    def construct_research_variables(self):
        """Construct all research variables"""
        print("\n" + "="*70)
        print("VARIABLE CONSTRUCTION")
        print("="*70)
        
        df = self.df.copy()
        
        # 1. Relative Performance (RP)
        print("\n1. Constructing Relative Performance (RP)...")
        
        races_per_season = {
            2016: 21, 2017: 20, 2018: 21, 2019: 21, 2020: 17,
            2021: 22, 2022: 22, 2023: 22, 2024: 24, 2025: 24
        }
        
        df['max_points_season'] = df['year'].map(lambda y: races_per_season.get(y, 22) * 44)
        df['RP'] = df['points'] / df['max_points_season']
        
        print(f"   RP range: {df['RP'].min():.3f} to {df['RP'].max():.3f}")
        
        # 2. Budget data
        print("\n2. Checking Budget Data...")
        
        if 'RB' in df.columns and 'estimated_budget_millions' in df.columns:
            print("   Real budget data detected")
            if 'RB_normalized' not in df.columns:
                df['RB_normalized'] = df['RB'] / df['RB'].max()
            self.budget_var = 'RB_normalized'
            print("   Using: Real budget data (RB_normalized)")
        else:
            print("   WARNING: Using performance proxy")
            df = df.sort_values(['constructor_id', 'year'])
            df['points_rolling_3yr'] = (df.groupby('constructor_id')['points']
                                         .transform(lambda x: x.rolling(3, min_periods=1).mean()))
            min_budget_proxy = df.groupby('year')['points_rolling_3yr'].transform('min')
            df['RB_normalized'] = df['points_rolling_3yr'] - min_budget_proxy
            df['RB_normalized'] = df['RB_normalized'] / df['RB_normalized'].max()
            self.budget_var = 'RB_normalized'
        
        # 3. Driver Effects proxy
        print("\n3. Constructing Driver Effects (DR_proxy)...")
        df['DR_proxy'] = df['points'] / df['max_points_season']
        
        # 4. Budget Cap dummy and interactions
        df['BC'] = (df['year'] >= 2021).astype(int)
        df['RB_BC_interaction'] = df[self.budget_var] * df['BC']
        
        # 5. Controls
        df['tech_reg_change'] = ((df['year'] == 2017) | (df['year'] == 2022)).astype(int)
        df['covid_year'] = (df['year'] == 2020).astype(int)
        
        # 6. Budget categories
        print("\n4. Classifying teams by budget category...")
        avg_points_pre2021 = (df[df['year'] <= 2020]
                               .groupby('constructor_id')['points'].mean())
        
        high_budget_teams = avg_points_pre2021.nlargest(3).index.tolist()
        medium_budget_teams = avg_points_pre2021.nlargest(7).index.tolist()[3:]
        
        df['budget_category'] = 'low'
        df.loc[df['constructor_id'].isin(medium_budget_teams), 'budget_category'] = 'medium'
        df.loc[df['constructor_id'].isin(high_budget_teams), 'budget_category'] = 'high'
        
        print(f"   High-budget: {df[df['budget_category']=='high']['constructor_name'].unique()}")
        print(f"   Medium-budget: {df[df['budget_category']=='medium']['constructor_name'].unique()}")
        print(f"   Low-budget: {df[df['budget_category']=='low']['constructor_name'].unique()}")
        
        self.df = df
        
        # Save variable construction summary
        summary_dict = {
            'Variable': ['RP', 'RB_normalized', 'DR_proxy', 'BC', 'RB_BC_interaction'],
            'Description': [
                'Relative Performance',
                'Normalized Budget',
                'Driver Effects Proxy',
                'Budget Cap Dummy (1 if year>=2021)',
                'Budget × Budget Cap Interaction'
            ],
            'Min': [df['RP'].min(), df[self.budget_var].min(), df['DR_proxy'].min(), 
                    df['BC'].min(), df['RB_BC_interaction'].min()],
            'Max': [df['RP'].max(), df[self.budget_var].max(), df['DR_proxy'].max(),
                    df['BC'].max(), df['RB_BC_interaction'].max()],
            'Mean': [df['RP'].mean(), df[self.budget_var].mean(), df['DR_proxy'].mean(),
                     df['BC'].mean(), df['RB_BC_interaction'].mean()]
        }
        summary_df = pd.DataFrame(summary_dict)
        summary_df.to_csv(f'{self.output_dir}/01_variable_summary.csv', index=False)
        print(f"\n✓ Saved: {self.output_dir}/01_variable_summary.csv")
        
        print("\nVariable construction complete")
        return df
    
    def create_team_tiers(self, df):
        """
        Create team performance tiers based on historical performance
        Reduces dimensionality while capturing key heterogeneity
        """
        # Calculate average performance by team across all years
        team_avg_performance = df.groupby('constructor_id')['RP'].mean().sort_values(ascending=False)
        
        print("\nTeam Performance Ranking (by average RP):")
        print(team_avg_performance)
        
        # Define tiers based on performance terciles or manual classification
        # Option A: Automatic terciles
        n_teams = len(team_avg_performance)
        top_cutoff = n_teams // 3
        mid_cutoff = 2 * n_teams // 3
        
        top_tier_teams = team_avg_performance.index[:top_cutoff].tolist()
        mid_tier_teams = team_avg_performance.index[top_cutoff:mid_cutoff].tolist()
        back_tier_teams = team_avg_performance.index[mid_cutoff:].tolist()
        
        # Option B: Manual classification (uncomment if preferred)
        # top_tier_teams = ['Mercedes', 'Red Bull', 'Ferrari']  # Adjust based on your data
        # mid_tier_teams = ['McLaren', 'Alpine', 'Aston Martin', 'AlphaTauri']
        # back_tier_teams = ['Williams', 'Alfa Romeo', 'Haas']
        
        print(f"\nTier Classification:")
        print(f"  Top tier ({len(top_tier_teams)}): {top_tier_teams}")
        print(f"  Mid tier ({len(mid_tier_teams)}): {mid_tier_teams}")
        print(f"  Back tier ({len(back_tier_teams)}): {back_tier_teams}")
        
        # Create tier variable
        df['team_tier'] = 'back'  # default
        df.loc[df['constructor_id'].isin(mid_tier_teams), 'team_tier'] = 'mid'
        df.loc[df['constructor_id'].isin(top_tier_teams), 'team_tier'] = 'top'
        
        return df
    
    
    def estimate_core_model(self):
        """Estimate core panel regression with simplified tier-based fixed effects"""
        print("\n" + "="*70)
        print("CORE PANEL REGRESSION MODEL (TIER-BASED FIXED EFFECTS)")
        print("="*70)
        
        df = self.df.copy()
        
        # Data validation
        print("\nData validation:")
        print(f"  Total rows: {len(df)}")
        print(f"  RP missing: {df['RP'].isna().sum()}")
        print(f"  {self.budget_var} missing: {df[self.budget_var].isna().sum()}")
        
        # Check for infinite values
        numeric_cols = [self.budget_var, 'DR_proxy', 'BC', 'RB_BC_interaction', 'tech_reg_change', 'covid_year']
        for col in numeric_cols:
            if col in df.columns:
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    print(f"  WARNING: {col} has {inf_count} infinite values")
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # Drop rows with missing key variables
        initial_len = len(df)
        df = df.dropna(subset=['RP', self.budget_var, 'DR_proxy'])
        dropped = initial_len - len(df)
        if dropped > 0:
            print(f"  Dropped {dropped} rows with missing values")
        
        print(f"  Final sample size: {len(df)}")
        
        if len(df) < 30:
            print("\nERROR: Sample size too small for regression")
            return None
        
        # Create team tiers
        df = self.create_team_tiers(df)
        
        # Prepare variables
        y = df['RP'].astype(float)
        X_vars = [self.budget_var, 'BC', 'RB_BC_interaction', 
                  'tech_reg_change', 'covid_year']
        
        X = df[X_vars].copy()
        
        # CRITICAL: Convert all columns to numeric, force errors to NaN
        print("\nConverting variables to numeric:")
        for col in X_vars:
            original_dtype = X[col].dtype
            X[col] = pd.to_numeric(X[col], errors='coerce')
            converted_dtype = X[col].dtype
            print(f"  {col}: {original_dtype} -> {converted_dtype}")
            
            # Check if conversion created NaN values
            nan_count = X[col].isna().sum()
            if nan_count > 0:
                print(f"    WARNING: {nan_count} values converted to NaN")
        
        # Drop any rows that have NaN after conversion
        before_drop = len(X)
        X = X.dropna()
        y = y.loc[X.index]
        after_drop = len(X)
        if before_drop != after_drop:
            print(f"  Dropped {before_drop - after_drop} rows with NaN after numeric conversion")
        
        # Check for zero variance
        zero_var_cols = []
        for col in X_vars:
            if col in X.columns and X[col].std() == 0:
                print(f"  WARNING: {col} has zero variance, removing from model")
                zero_var_cols.append(col)
        
        for col in zero_var_cols:
            X_vars.remove(col)
            X = X.drop(columns=[col])
        
        # Add tier fixed effects (drop first tier as reference category)
        df_aligned = df.loc[X.index]  # Align df with X after dropping NaN
        tier_dummies = pd.get_dummies(df_aligned['team_tier'], prefix='tier', drop_first=True)
        
        # Ensure tier dummies are numeric (0/1)
        for col in tier_dummies.columns:
            tier_dummies[col] = tier_dummies[col].astype(int)
        
        X = pd.concat([X, tier_dummies], axis=1)
        X = add_constant(X)
        
        # Final data type check
        print("\nFinal data types before regression:")
        print(f"  y (RP): {y.dtype}")
        for col in X.columns:
            print(f"  {col}: {X[col].dtype}")
        
        # Verify all are numeric
        non_numeric = [col for col in X.columns if not np.issubdtype(X[col].dtype, np.number)]
        if non_numeric:
            print(f"\n  ERROR: Non-numeric columns detected: {non_numeric}")
            print("  Converting to numeric...")
            for col in non_numeric:
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Final NaN check
        if X.isna().any().any():
            print("\n  WARNING: NaN values detected after all conversions")
            print(X.isna().sum())
            X = X.dropna()
            y = y.loc[X.index]
            print(f"  Final sample size: {len(X)}")
        
        # Check degrees of freedom
        print(f"\n  Independent variables: {len(X.columns)}")
        print(f"  Observations: {len(X)}")
        print(f"  Degrees of freedom: {len(X) - len(X.columns)}")
        
        if len(X.columns) >= len(X):
            print("\n  ERROR: Still not enough degrees of freedom")
            print("  Try one of these solutions:")
            print("  1. Remove some control variables")
            print("  2. Increase sample size")
            print("  3. Use pooled OLS without fixed effects")
            return None
        
        try:
            model = OLS(y, X, missing='drop')
            results = model.fit(cov_type='HC1')  # Robust standard errors
            
            print("\n" + "="*70)
            print("REGRESSION RESULTS")
            print("="*70)
            print(results.summary())
            
            # Save full regression output
            with open(f'{self.output_dir}/02_full_regression_results.txt', 'w') as f:
                f.write("TIER-BASED FIXED EFFECTS MODEL\n")
                f.write("="*70 + "\n\n")
                f.write(results.summary().as_text())
            print(f"\n✓ Saved: {self.output_dir}/02_full_regression_results.txt")
            
            print("\n" + "="*70)
            print("KEY COEFFICIENTS")
            print("="*70)
            
            coef_names = {
                self.budget_var: 'Budget effect',
                'DR_proxy': 'Driver effect',
                'BC': 'Budget cap main effect',
                'RB_BC_interaction': 'Budget cap × Budget interaction',
                'tech_reg_change': 'Technical regulation change',
                'covid_year': 'COVID year effect'
            }
            
            # Create coefficients table
            coef_data = []
            for var, name in coef_names.items():
                if var in results.params:
                    coef = results.params[var]
                    pval = results.pvalues[var]
                    stderr = results.bse[var]
                    sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
                    print(f"\n{name}: {coef:.4f} {sig}")
                    print(f"  Std Error: {stderr:.4f}")
                    print(f"  P-value: {pval:.4f}")
                    
                    coef_data.append({
                        'Variable': name,
                        'Coefficient': coef,
                        'Std Error': stderr,
                        't-statistic': coef/stderr if stderr != 0 else np.nan,
                        'P-value': pval,
                        'Significance': sig
                    })
            
            coef_df = pd.DataFrame(coef_data)
            coef_df.to_csv(f'{self.output_dir}/02_key_coefficients.csv', index=False)
            print(f"\n✓ Saved: {self.output_dir}/02_key_coefficients.csv")
            
            # Calculate efficiency (residuals)
            self.df['EF'] = np.nan
            self.df.loc[results.model.data.row_labels, 'EF'] = results.resid
            
            print(f"\nModel fit:")
            print(f"  R-squared: {results.rsquared:.4f}")
            print(f"  Adj R-squared: {results.rsquared_adj:.4f}")
            print(f"  F-statistic: {results.fvalue:.2f} (p={results.f_pvalue:.4f})")
            
            # Save model fit statistics
            fit_stats = pd.DataFrame({
                'Statistic': ['R-squared', 'Adj R-squared', 'F-statistic', 'F p-value', 'N observations'],
                'Value': [results.rsquared, results.rsquared_adj, results.fvalue, 
                         results.f_pvalue, results.nobs]
            })
            fit_stats.to_csv(f'{self.output_dir}/02_model_fit_statistics.csv', index=False)
            print(f"✓ Saved: {self.output_dir}/02_model_fit_statistics.csv")
            
            self.results['core_model'] = results
            return results
            
        except Exception as e:
            print(f"\nERROR in regression: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def chow_test(self):
        """Structural break test"""
        print("\n" + "="*70)
        print("CHOW TEST FOR STRUCTURAL BREAK")
        print("="*70)
        
        df = self.df.copy()
        
        pre_cap = df[df['year'] <= 2020].copy()
        post_cap = df[df['year'] >= 2021].copy()
        
        print(f"\nPre-cap: {len(pre_cap)} obs")
        print(f"Post-cap: {len(post_cap)} obs")
        
        X_vars = [self.budget_var, 'DR_proxy']
        
        y_pre = pre_cap['RP']
        X_pre = add_constant(pre_cap[X_vars])
        model_pre = OLS(y_pre, X_pre, missing='drop').fit()
        
        y_post = post_cap['RP']
        X_post = add_constant(post_cap[X_vars])
        model_post = OLS(y_post, X_post, missing='drop').fit()
        
        y_pooled = df['RP']
        X_pooled = add_constant(df[X_vars])
        model_pooled = OLS(y_pooled, X_pooled, missing='drop').fit()
        
        RSS_pre = model_pre.ssr
        RSS_post = model_post.ssr
        RSS_pooled = model_pooled.ssr
        
        k = len(X_vars) + 1
        n1 = len(y_pre.dropna())
        n2 = len(y_post.dropna())
        
        chow_numerator = (RSS_pooled - (RSS_pre + RSS_post)) / k
        chow_denominator = (RSS_pre + RSS_post) / (n1 + n2 - 2*k)
        chow_stat = chow_numerator / chow_denominator
        p_value = 1 - stats.f.cdf(chow_stat, k, n1 + n2 - 2*k)
        
        print(f"\nChow F-statistic: {chow_stat:.4f}")
        print(f"P-value: {p_value:.6f}")
        
        if p_value < 0.05:
            print("\nREJECT null: Significant structural break detected")
            conclusion = "REJECT null: Significant structural break detected"
        else:
            print("\nFAIL TO REJECT: No significant structural break")
            conclusion = "FAIL TO REJECT: No significant structural break"
        
        # Save Chow test results
        with open(f'{self.output_dir}/03_chow_test_results.txt', 'w') as f:
            f.write("="*70 + "\n")
            f.write("CHOW TEST FOR STRUCTURAL BREAK\n")
            f.write("="*70 + "\n")
            f.write(f"Pre-cap: {len(pre_cap)} obs\n")
            f.write(f"Post-cap: {len(post_cap)} obs\n")
            f.write(f"Chow F-statistic: {chow_stat:.4f}\n")
            f.write(f"P-value: {p_value:.6f}\n")
            f.write(f"{conclusion}\n")
        print(f"\n✓ Saved: {self.output_dir}/03_chow_test_results.txt")
        
        # Save detailed results to CSV
        chow_results_df = pd.DataFrame({
            'Metric': ['Pre-cap observations', 'Post-cap observations', 
                      'Chow F-statistic', 'P-value', 'Conclusion'],
            'Value': [len(pre_cap), len(post_cap), chow_stat, p_value, conclusion]
        })
        chow_results_df.to_csv(f'{self.output_dir}/03_chow_test_results.csv', index=False)
        print(f"✓ Saved: {self.output_dir}/03_chow_test_results.csv")
        
        self.results['chow_test'] = {
            'statistic': chow_stat,
            'p_value': p_value,
            'model_pre': model_pre,
            'model_post': model_post
        }
        
        return chow_stat, p_value
    
    def markov_chain_analysis(self):
        """Competitive dynamics analysis"""
        print("\n" + "="*70)
        print("MARKOV CHAIN ANALYSIS")
        print("="*70)
        
        df = self.df.copy()
        
        def categorize_position(pos):
            if pos <= 3:
                return 'Top 3'
            elif pos <= 6:
                return 'Midfield'
            else:
                return 'Bottom'
        
        df['position_category'] = df['position'].apply(categorize_position)
        
        def calculate_transition_matrix(data):
            data = data.sort_values(['constructor_id', 'year'])
            data['next_position'] = data.groupby('constructor_id')['position_category'].shift(-1)
            transitions = data[data['next_position'].notna()].copy()
            
            transition_counts = pd.crosstab(
                transitions['position_category'],
                transitions['next_position'],
                normalize='index'
            )
            
            categories = ['Top 3', 'Midfield', 'Bottom']
            for cat in categories:
                if cat not in transition_counts.index:
                    transition_counts.loc[cat] = 0
                if cat not in transition_counts.columns:
                    transition_counts[cat] = 0
            
            return transition_counts.loc[categories, categories].fillna(0)
        
        pre_transitions = calculate_transition_matrix(df[df['year'] <= 2020])
        post_transitions = calculate_transition_matrix(df[df['year'] >= 2021])
        
        print("\nPre-Budget Cap Transitions:")
        print(pre_transitions.round(3))
        
        print("\nPost-Budget Cap Transitions:")
        print(post_transitions.round(3))
        
        # Save transition matrices
        pre_transitions.to_csv(f'{self.output_dir}/04_markov_pre_cap.csv')
        post_transitions.to_csv(f'{self.output_dir}/04_markov_post_cap.csv')
        print(f"\n✓ Saved: {self.output_dir}/04_markov_pre_cap.csv")
        print(f"✓ Saved: {self.output_dir}/04_markov_post_cap.csv")
        
        mobility_pre = 1 - np.mean(np.diag(pre_transitions))
        mobility_post = 1 - np.mean(np.diag(post_transitions))
        
        print(f"\nMobility Index:")
        print(f"  Pre-cap:  {mobility_pre:.3f}")
        print(f"  Post-cap: {mobility_post:.3f}")
        print(f"  Change: {((mobility_post - mobility_pre) / mobility_pre * 100):+.1f}%")
        
        # Save mobility metrics
        mobility_df = pd.DataFrame({
            'Period': ['Pre-cap', 'Post-cap', 'Change (%)'],
            'Mobility Index': [mobility_pre, mobility_post, 
                              (mobility_post - mobility_pre) / mobility_pre * 100]
        })
        mobility_df.to_csv(f'{self.output_dir}/04_mobility_index.csv', index=False)
        print(f"✓ Saved: {self.output_dir}/04_mobility_index.csv")
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        sns.heatmap(pre_transitions, annot=True, fmt='.3f', cmap='Blues',
                    ax=axes[0], cbar_kws={'label': 'Probability'})
        axes[0].set_title('Pre-Budget Cap Transitions')
        axes[0].set_xlabel('Next Year Position')
        axes[0].set_ylabel('Current Position')
        
        sns.heatmap(post_transitions, annot=True, fmt='.3f', cmap='Oranges',
                    ax=axes[1], cbar_kws={'label': 'Probability'})
        axes[1].set_title('Post-Budget Cap Transitions')
        axes[1].set_xlabel('Next Year Position')
        axes[1].set_ylabel('Current Position')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/04_markov_transitions.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.output_dir}/04_markov_transitions.png")
        plt.show()
        
        self.results['markov'] = {
            'pre_transitions': pre_transitions,
            'post_transitions': post_transitions,
            'mobility_pre': mobility_pre,
            'mobility_post': mobility_post
        }
        
        return pre_transitions, post_transitions
    
    def convergence_analysis(self):
        """Sigma and beta convergence tests"""
        print("\n" + "="*70)
        print("CONVERGENCE ANALYSIS")
        print("="*70)
        
        df = self.df.copy()
        
        # Sigma-convergence
        print("\n1. SIGMA-CONVERGENCE")
        print("="*70)
        
        yearly_std = df.groupby('year')['RP'].std()
        
        pre_std = yearly_std[yearly_std.index <= 2020].mean()
        post_std = yearly_std[yearly_std.index >= 2021].mean()
        
        print(f"\nStd Dev of RP:")
        print(f"  Pre-cap:  {pre_std:.4f}")
        print(f"  Post-cap: {post_std:.4f}")
        print(f"  Change: {((post_std - pre_std) / pre_std * 100):+.1f}%")
        
        if post_std < pre_std:
            print("\nSIGMA-CONVERGENCE detected")
            sigma_result = "SIGMA-CONVERGENCE detected"
        else:
            print("\nSIGMA-DIVERGENCE")
            sigma_result = "SIGMA-DIVERGENCE"
        
        # Save sigma convergence results
        sigma_df = pd.DataFrame({
            'Year': yearly_std.index,
            'Std_Dev_RP': yearly_std.values
        })
        sigma_df.to_csv(f'{self.output_dir}/05_sigma_convergence_yearly.csv', index=False)
        print(f"\n✓ Saved: {self.output_dir}/05_sigma_convergence_yearly.csv")
        
        sigma_summary = pd.DataFrame({
            'Period': ['Pre-cap', 'Post-cap', 'Change (%)'],
            'Std Dev': [pre_std, post_std, (post_std - pre_std) / pre_std * 100],
            'Result': [sigma_result, '', '']
        })
        sigma_summary.to_csv(f'{self.output_dir}/05_sigma_convergence_summary.csv', index=False)
        print(f"✓ Saved: {self.output_dir}/05_sigma_convergence_summary.csv")
        
        # Beta-convergence
        print("\n2. BETA-CONVERGENCE")
        print("="*70)
        
        df_sorted = df.sort_values(['constructor_id', 'year'])
        df_sorted['RP_growth'] = df_sorted.groupby('constructor_id')['RP'].pct_change()
        
        pre_data = df_sorted[(df_sorted['year'] >= 2017) & (df_sorted['year'] <= 2020)].copy()
        pre_data['initial_RP'] = pre_data.groupby('constructor_id')['RP'].transform('first')
        
        post_data = df_sorted[df_sorted['year'] >= 2021].copy()
        post_data['initial_RP'] = post_data.groupby('constructor_id')['RP'].transform('first')
        
        y_pre = pre_data['RP_growth'].dropna()
        X_pre = add_constant(pre_data.loc[y_pre.index, 'initial_RP'])
        beta_pre_model = OLS(y_pre, X_pre).fit()
        beta_pre = beta_pre_model.params['initial_RP']
        
        y_post = post_data['RP_growth'].dropna()
        X_post = add_constant(post_data.loc[y_post.index, 'initial_RP'])
        beta_post_model = OLS(y_post, X_post).fit()
        beta_post = beta_post_model.params['initial_RP']
        
        print(f"\nBeta Coefficient:")
        print(f"  Pre-cap:  {beta_pre:.4f} (p={beta_pre_model.pvalues['initial_RP']:.4f})")
        print(f"  Post-cap: {beta_post:.4f} (p={beta_post_model.pvalues['initial_RP']:.4f})")
        
        # Save beta convergence results
        beta_summary = pd.DataFrame({
            'Period': ['Pre-cap', 'Post-cap'],
            'Beta Coefficient': [beta_pre, beta_post],
            'P-value': [beta_pre_model.pvalues['initial_RP'], 
                       beta_post_model.pvalues['initial_RP']],
            'Significant': [beta_pre_model.pvalues['initial_RP'] < 0.05,
                           beta_post_model.pvalues['initial_RP'] < 0.05]
        })
        beta_summary.to_csv(f'{self.output_dir}/05_beta_convergence_summary.csv', index=False)
        print(f"✓ Saved: {self.output_dir}/05_beta_convergence_summary.csv")
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(yearly_std.index, yearly_std.values, 'o-', linewidth=2, markersize=8)
        axes[0].axvline(x=2020.5, color='red', linestyle='--', linewidth=2)
        axes[0].axhline(y=pre_std, color='blue', linestyle=':', alpha=0.5, label=f'Pre: {pre_std:.3f}')
        axes[0].axhline(y=post_std, color='orange', linestyle=':', alpha=0.5, label=f'Post: {post_std:.3f}')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Std Dev of RP')
        axes[0].set_title('Sigma-Convergence')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        pre_scatter = pre_data.dropna(subset=['RP_growth', 'initial_RP'])
        post_scatter = post_data.dropna(subset=['RP_growth', 'initial_RP'])
        
        axes[1].scatter(pre_scatter['initial_RP'], pre_scatter['RP_growth'], 
                       alpha=0.6, s=50, label='Pre-Cap', color='blue')
        axes[1].scatter(post_scatter['initial_RP'], post_scatter['RP_growth'],
                       alpha=0.6, s=50, label='Post-Cap', color='orange')
        
        x_range = np.linspace(0, 0.5, 100)
        axes[1].plot(x_range, beta_pre_model.params['const'] + beta_pre * x_range,
                    'b--', linewidth=2, label=f'Pre: β={beta_pre:.3f}')
        axes[1].plot(x_range, beta_post_model.params['const'] + beta_post * x_range,
                    '--', color='orange', linewidth=2, label=f'Post: β={beta_post:.3f}')
        
        axes[1].set_xlabel('Initial Performance')
        axes[1].set_ylabel('Growth Rate')
        axes[1].set_title('Beta-Convergence')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/05_convergence_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.output_dir}/05_convergence_analysis.png")
        plt.show()
        
        self.results['convergence'] = {
            'sigma_pre': pre_std,
            'sigma_post': post_std,
            'beta_pre': beta_pre,
            'beta_post': beta_post
        }
        
        return {'sigma_pre': pre_std, 'sigma_post': post_std}
    
    def generate_thesis_report(self):
        """Generate final summary"""
        print("\n" + "="*70)
        print("THESIS SUMMARY REPORT")
        print("="*70)
        
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("THESIS SUMMARY REPORT")
        report_lines.append("="*70)
        report_lines.append("")
        
        print("\nRESEARCH QUESTIONS:")
        report_lines.append("RESEARCH QUESTIONS:")
        report_lines.append("")
        
        print("\nRQ1: Budget cap effect on performance")
        report_lines.append("RQ1: Budget cap effect on performance")
        if 'core_model' in self.results:
            bc_effect = self.results['core_model'].params.get('BC', np.nan)
            bc_pval = self.results['core_model'].pvalues.get('BC', np.nan)
            print(f"  Effect: {bc_effect:.4f} (p={bc_pval:.4f})")
            print(f"  {'Significant' if bc_pval < 0.05 else 'Not significant'}")
            report_lines.append(f"  Effect: {bc_effect:.4f} (p={bc_pval:.4f})")
            report_lines.append(f"  {'Significant' if bc_pval < 0.05 else 'Not significant'}")
        report_lines.append("")
        
        print("\nRQ2: Competitive balance improvement")
        report_lines.append("RQ2: Competitive balance improvement")
        if 'convergence' in self.results:
            sigma_change = ((self.results['convergence']['sigma_post'] - 
                           self.results['convergence']['sigma_pre']) / 
                          self.results['convergence']['sigma_pre'] * 100)
            print(f"  Variance change: {sigma_change:+.1f}%")
            print(f"  Balance {'improved' if sigma_change < 0 else 'worsened'}")
            report_lines.append(f"  Variance change: {sigma_change:+.1f}%")
            report_lines.append(f"  Balance {'improved' if sigma_change < 0 else 'worsened'}")
        report_lines.append("")
        
        print("\nRQ3: Structural break at 2021")
        report_lines.append("RQ3: Structural break at 2021")
        if 'chow_test' in self.results:
            print(f"  Chow test p-value: {self.results['chow_test']['p_value']:.6f}")
            print(f"  {'Significant break' if self.results['chow_test']['p_value'] < 0.05 else 'No significant break'}")
            report_lines.append(f"  Chow test p-value: {self.results['chow_test']['p_value']:.6f}")
            report_lines.append(f"  {'Significant break' if self.results['chow_test']['p_value'] < 0.05 else 'No significant break'}")
        report_lines.append("")
        
        print("\nMETHODS COMPLETED:")
        report_lines.append("METHODS COMPLETED:")
        print("  - Panel regression with fixed effects")
        print("  - Structural break testing (Chow)")
        print("  - Markov chain analysis")
        print("  - Convergence tests (sigma & beta)")
        report_lines.append("  - Panel regression with fixed effects")
        report_lines.append("  - Structural break testing (Chow)")
        report_lines.append("  - Markov chain analysis")
        report_lines.append("  - Convergence tests (sigma & beta)")
        report_lines.append("")
        
        print("\nDATA SUMMARY:")
        report_lines.append("DATA SUMMARY:")
        print(f"  Period: {self.df['year'].min()}-{self.df['year'].max()}")
        print(f"  Observations: {len(self.df)}")
        print(f"  Teams: {self.df['constructor_name'].nunique()}")
        report_lines.append(f"  Period: {self.df['year'].min()}-{self.df['year'].max()}")
        report_lines.append(f"  Observations: {len(self.df)}")
        report_lines.append(f"  Teams: {self.df['constructor_name'].nunique()}")
        
        # Save summary report
        with open(f'{self.output_dir}/06_thesis_summary_report.txt', 'w') as f:
            f.write('\n'.join(report_lines))
        print(f"\n✓ Saved: {self.output_dir}/06_thesis_summary_report.txt")

def run_complete_analysis(df, output_dir='output'):
    """Execute complete analysis pipeline"""
    print("\n" + "="*70)
    print("F1 BUDGET CAP - COMPLETE ECONOMETRIC ANALYSIS")
    print("="*70)
    print(f"Output directory: {output_dir}/")
    
    analyzer = F1EconometricAnalysis(df, output_dir=output_dir)
    
    print("\nSTEP 1: Variable Construction")
    analyzer.construct_research_variables()
    
    print("\nSTEP 2: Core Regression")
    analyzer.estimate_core_model()
    
    print("\nSTEP 3: Structural Break Testing")
    analyzer.chow_test()
    
    print("\nSTEP 4: Competitive Dynamics")
    analyzer.markov_chain_analysis()
    
    print("\nSTEP 5: Convergence Testing")
    analyzer.convergence_analysis()
    
    print("\nSTEP 6: Final Report")
    analyzer.generate_thesis_report()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nAll outputs saved to: {output_dir}/")
    
    return analyzer