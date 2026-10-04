# Figure provenance and QA

Figures 1--2 retain the author artwork. Figure 3 rebuilds the same four profile comparisons using archived monthly statistics and all confirmation event labels. The original four displayed hourly month profiles are retained. Figure 4 uses all 7,873 records and 624 events. The full-history input remains access-controlled.

Figure 5 evaluates 136 complete trailing 30-calendar-day windows daily; every denominator is positive. Windows overlap and are descriptive, not independent replicates. Figure 6 sums score-sorted addition-minus-removal priorities at every feasible exchange count across the actual 21,22,22 pairs. Its x-axis is changed memberships 2*l; y-axis is F_m(l). Triangles indicate nonzero realized excess differences for the incremental pair, not cumulative realized value. No fitted trend, jitter or selected sample is used.

Figure 7 includes all 66 zone-months for both score origins. Each cell is 100*(local captured excess - base captured excess)/total excess. Zero denominators are gray. Signed-log normalization is explicit, linear within +/-1 percentage point and identical for both blocks. All values are unchanged. Calendar-month uncertainty remains in the results table.

Yellow #D5A62D is primary, blue #277DA8 secondary, green #7E9854 auxiliary. Negative outcome markers and negative heatmap effects use blue. The heatmap zero is olive, never pale white. Styles and shapes additionally distinguish curves and effect directions. PDF/SVG have editable text; PNG is 600 dpi. Static validator warnings about absent TIFF are accepted because line artwork is supplied as vector PDF.

Rebuild Figure 7 with plot_paper_results.py, Figures 5--6 with plot_zhejiang_temporal.py --input an_authorized_audit.csv, and Figure 3 with plot_profile_palette.py --audit an_authorized_audit.csv --monthly data/summary/zhejiang_operating_profile.csv --out results/figures. Full Zhejiang inputs are not supplied publicly.
