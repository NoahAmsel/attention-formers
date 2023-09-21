import pandas as pd
import seaborn as sns

df = pd.read_csv("results_qk.csv")
wide = pd.pivot(df, index="dim", columns="ntokens", values="error")
plot = sns.heatmap(wide, cbar_kws={'label': 'avg angle btwn prediction & label'})
fig = plot.get_figure()
fig.savefig("out_qk.png") 