"""Compute per-length-class statistics and paired significance tests for the sweep results."""
import json
import numpy as np
from scipy.stats import wilcoxon, ttest_rel
from collections import defaultdict

def bootstrap_ci(data, n_resamples=5000, ci=0.95):
    """Bootstrap 95% CI for the mean."""
    if len(data) < 2:
        return (data[0], data[0]) if data else (0, 0)
    means = np.zeros(n_resamples)
    for i in range(n_resamples):
        sample = np.random.choice(data, size=len(data), replace=True)
        means[i] = np.mean(sample)
    alpha = (1 - ci) / 2
    return (float(np.percentile(means, alpha * 100)),
            float(np.percentile(means, (1 - alpha) * 100)))

d = json.load(open('results_sweep.json'))
cases = d['case_metrics']

STRATEGIES = ['full_context', 'truncation', 'running_summary', 'hierarchical', 'slot_memory']
STRAT_LABELS = {
    'full_context': 'Full Context',
    'truncation': 'Truncation',
    'running_summary': 'Running Summary',
    'hierarchical': 'Append-only Notes',
    'slot_memory': 'Fact-Slot Memory',
}

# ---- 1. Per-length-class aggregate stats ----
print("=" * 80)
print("PER-LENGTH-CLASS STATISTICS")
print("=" * 80)

lengths = ['short', 'medium', 'long', 'xlong', 'xxlong']
length_labels = {'short': 'Short (14)', 'medium': 'Medium (32)', 'long': 'Long (60)',
                 'xlong': 'XLong (120)', 'xxlong': 'XXLong (200)'}

for strat in STRATEGIES:
    print(f"\n--- {STRAT_LABELS[strat]} ---")
    print(f"{'Length':<15} {'n':<5} {'Retention%':<20} {'Cost%':<20} {'TokRed%':<20}")
    print("-" * 80)
    for ll in lengths:
        vals = [c['strategies'][strat] for c in cases if c['length_class'] == ll]
        rets = np.array([v['retention_pct'] for v in vals])
        costs = np.array([v['cost_pct_of_baseline'] for v in vals])
        toks = np.array([v['token_reduction_real_pct'] for v in vals if v['token_reduction_real_pct'] is not None])
        
        ret_ci = bootstrap_ci(rets)
        cost_ci = bootstrap_ci(costs)
        tok_ci = bootstrap_ci(toks) if len(toks) else (0, 0)
        
        print(f"{length_labels[ll]:<15} {len(rets):<5} "
              f"{np.mean(rets):.1f} ± {np.std(rets, ddof=1):.1f} [{ret_ci[0]:.1f}-{ret_ci[1]:.1f}]   "
              f"{np.mean(costs):.1f} ± {np.std(costs, ddof=1):.1f} [{cost_ci[0]:.1f}-{cost_ci[1]:.1f}]   "
              f"{np.mean(toks):.1f} ± {np.std(toks, ddof=1):.1f} [{tok_ci[0]:.1f}-{tok_ci[1]:.1f}]")

# ---- 2. Paired significance tests ----
print("\n" + "=" * 80)
print("PAIRED SIGNIFICANCE TESTS (Wilcoxon signed-rank)")
print("=" * 80)

METHODS_PAIRS = [
    ('slot_memory', 'hierarchical', 'Fact-Slot vs Notes'),
    ('slot_memory', 'running_summary', 'Fact-Slot vs Summary'),
    ('slot_memory', 'truncation', 'Fact-Slot vs Truncation'),
    ('slot_memory', 'full_context', 'Fact-Slot vs Full'),
    ('running_summary', 'full_context', 'Summary vs Full'),
]

for a, b, label in METHODS_PAIRS:
    print(f"\n--- {label} ---")
    for metric_name, metric_key in [('Retention', 'retention_pct'), 
                                     ('Cost% of baseline', 'cost_pct_of_baseline'),
                                     ('Token Reduction%', 'token_reduction_real_pct')]:
        pairs = []
        for c in cases:
            va = c['strategies'][a].get(metric_key)
            vb = c['strategies'][b].get(metric_key)
            if va is not None and vb is not None:
                pairs.append((va, vb))
        pairs = np.array(pairs)
        if len(pairs) < 6:
            print(f"  {metric_name}: insufficient pairs ({len(pairs)})")
            continue
        diff = pairs[:, 0] - pairs[:, 1]
        mean_a = np.mean(pairs[:, 0])
        mean_b = np.mean(pairs[:, 1])
        mean_diff = np.mean(diff)
        
        # Wilcoxon
        try:
            w_stat, w_p = wilcoxon(pairs[:, 0], pairs[:, 1])
        except:
            w_stat, w_p = None, 1.0
        
        # Paired t-test
        t_stat, t_p = ttest_rel(pairs[:, 0], pairs[:, 1])
        
        print(f"  {metric_name}: {mean_a:.1f} vs {mean_b:.1f}, "
              f"diff={mean_diff:+.1f} (n={len(pairs)}), "
              f"Wilcoxon p={w_p:.4f}, "
              f"paired-t p={t_p:.4f}")

# ---- 3. Aggregate table with CIs ----
print("\n" + "=" * 80)
print("AGGREGATE TABLE WITH CIs (LaTeX-ready)")
print("=" * 80)

for strat in STRATEGIES:
    vals = [c['strategies'][strat] for c in cases]
    rets = np.array([v['retention_pct'] for v in vals])
    costs = np.array([v['cost_pct_of_baseline'] for v in vals])
    toks = np.array([v['token_reduction_real_pct'] for v in vals if v['token_reduction_real_pct'] is not None])
    
    ret_ci = bootstrap_ci(rets)
    cost_ci = bootstrap_ci(costs)
    tok_ci = bootstrap_ci(toks) if len(toks) else (0, 0)
    
    print(f"{STRAT_LABELS[strat]:<22} & "
          f"${np.mean(toks):.1f}\\%\\pm{np.std(toks, ddof=1):.1f}$ & "
          f"${np.mean(rets):.1f}\\%\\pm{np.std(rets, ddof=1):.1f}$ & "
          f"${np.mean(costs):.1f}\\%\\pm{np.std(costs, ddof=1):.1f}$ \\\\")

# ---- 4. Per-length-class LaTeX table ----
print("\n" + "=" * 80)
print("PER-LENGTH-CLASS LaTeX TABLE")
print("=" * 80)

for strat_name, strat_key in [('Fact-Slot Memory', 'slot_memory'), ('Running Summary', 'running_summary')]:
    print(f"\n% {strat_name}")
    for ll in lengths:
        vals = [c['strategies'][strat_key] for c in cases if c['length_class'] == ll]
        rets = np.array([v['retention_pct'] for v in vals])
        costs = np.array([v['cost_pct_of_baseline'] for v in vals])
        print(f"  {length_labels[ll]:<15} & "
              f"${np.mean(rets):.1f}\\%\\pm{np.std(rets, ddof=1):.1f}$ & "
              f"${np.mean(costs):.1f}\\%\\pm{np.std(costs, ddof=1):.1f}$ \\\\")
