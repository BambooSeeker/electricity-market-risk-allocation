# Data dictionary

Zhejiang candidate: `time` is local delivery timestamp, half-hourly;
`da_price_mean` is the mean of archived nodal DA price columns, CNY/MWh;
`load_day_ahead_pred` is forecast system load, MW;
`net_load_pred` is forecast load minus forecast solar and wind, MW;
`da_cong_node_abs_mean` is mean absolute archived DA nodal congestion, CNY/MWh;
`spread_rt_minus_da` is the target generation-unit RT minus DA price, CNY/MWh.
`week_saved_score_audit.csv` supplies original priorities and monthly-cohort alert
flags for the same week. The full January--February source records are excluded.

NYISO processed: local `timestamp`, `zone`, and `occurrence` identify a delivered
hour; `issue_date` is the load-forecast archive publication date;
`day_ahead_price`, `real_time_price`, `day_ahead_congestion`, `da_rt_spread`,
`tail_threshold`, and `negative_excess` are USD/MWh; `load_forecast` is MW.
`lower_tail_event` is a binary threshold label. `zj_source_score` is an optional
archived prediction on NYISO inputs, with no restricted training records.
The reconstruction adds shifted historical statistics and calibration scores.
