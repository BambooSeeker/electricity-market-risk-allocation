from pathlib import Path
import runpy

root = Path(__file__).resolve().parent
# Figure 2 is the author-provided diagram distributed with the manuscript.
runpy.run_path(str(root/'plot_results_final.py'), run_name='__main__')
