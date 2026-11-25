"""Write the findings into the README after the real experiment runs."""
from src.data import ROOT


def update_readme(df, comparison, metrics, importance):
    top = importance.sort_values('Permutation RMSE increase', ascending=False).iloc[0]
    baseline = comparison.set_index('Model').loc['Linear Regression']
    findings = (f'I worked with {len(df):,} records. Average output was {df.PE.mean():.2f} MW, '
                f'and the temperature-output correlation was {df.corr().loc["AT", "PE"]:.3f}. '
                f'The linear baseline had a test RMSE of {baseline.RMSE:.2f} MW. '
                f'After selecting and tuning {metrics["model"]} using training cross-validation, '
                f'the final test RMSE was {metrics["RMSE"]:.2f} MW, MAE was {metrics["MAE"]:.2f} MW '
                f'and R² was {metrics["R2"]:.4f}. '
                f'The largest permutation importance belonged to {top.Feature}, with a '
                f'{top["Permutation RMSE increase"]:.2f} MW increase in error when shuffled. '
                'That is a measure of model dependence, not a causal effect.')
    path = ROOT / 'README.md'
    text = path.read_text(encoding='utf-8')
    start, end = '<!-- RESULTS START -->', '<!-- RESULTS END -->'
    path.write_text(text.split(start)[0]+start+'\n\n'+findings+'\n\n'+end+text.split(end)[1], encoding='utf-8')
    return findings
