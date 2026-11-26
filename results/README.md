# results/

The rows of every experiment the paper reports, as the training / evaluation scripts wrote them
(CSV: one row per run unless noted; `run` = the checkpoint name). Every figure and table of the
paper is drawn from these files by the scripts listed (see the README for the commands);
regenerated, they match the paper's figures pixel for pixel and its tables byte for byte.

| file | what | read by |
|---|---|---|
| `biomassters/crop_spread/mistr_biomassters_S8Tsea+S4Tmon_q50_woracle10_ep20_seed0_3107fc40.pt` | per-sample crop counts of one run (exp.bench.mistr_crop_spread) | per_sample_compute |
| `biomassters/inject_fft.csv` | earlier-code runs (the S2 and T_acq points of Fig 10 only) | space_time_ablation |
| `biomassters/inject_flops.csv` | GMACs of those earlier-code runs | space_time_ablation |
| `biomassters/mech_attn.csv` | attention mass per key group (exp.mistr.mech_probe) | oneway_probe_tables |
| `biomassters/mech_decoder.csv` | decoder crop weight and pooled predictions (exp.mistr.mech_probe) | oneway_probe_tables |
| `biomassters/mistr.csv` | one row per training run (exp.mistr.train): config, val/test metrics, GMACs | per_sample_compute, round1_main, round1_main_std, space_time_ablation, terramind_tradeoff |
| `biomassters/test_cost.csv` | GMACs per sample over the whole test split (exp.mistr.measure_test_cost) | round1_main, round1_main_std, space_time_ablation, terramind_tradeoff |
| `biomassters_year/mistr.csv` | one row per training run (exp.mistr.train): config, val/test metrics, GMACs | space_time_ablation |
| `geoidchange/tiletrack_N_prefix_fixed_flops.csv` | GEOID causal-prefix GMACs, fixed resolution | geoid_trackN |
| `geoidchange/tiletrack_N_prefix_flops.csv` | GEOID causal-prefix GMACs, MiSTR (exp.geoidchange.tiletrack_flops) | geoid_trackN |
| `geoidchange/tiletrack_O_prefix_flops.csv` | GEOID causal-prefix GMACs, MiSTR at the paper's budgets | geoid_trackN |
| `geoidchange/trackN_rescore.jsonl` | GEOID tracking test scores per event position (exp.geoidchange.trackN_rescore) | geoid_trackN |
| `pastis/crop_spread/mistr_pastis_S8Tsea+S4Tmon_q50_woracle50_ep100_seed0_9a7509fd.pt` | per-sample crop counts of one run (exp.bench.mistr_crop_spread) | per_sample_compute |
| `pastis/inject_fft.csv` | earlier-code runs (the S2 and T_acq points of Fig 10 only) | space_time_ablation |
| `pastis/inject_flops.csv` | GMACs of those earlier-code runs | space_time_ablation |
| `pastis/mech_attn.csv` | attention mass per key group (exp.mistr.mech_probe) | oneway_probe_tables |
| `pastis/mech_decoder.csv` | decoder crop weight and pooled predictions (exp.mistr.mech_probe) | oneway_probe_tables |
| `pastis/mistr.csv` | one row per training run (exp.mistr.train): config, val/test metrics, GMACs | baselines_r1, oneway_attn_trained, per_sample_compute, round1_main, round1_main_std, round1_throughput, season_range_tradeoff, selector_tables_r1, space_time_ablation, terramind_tradeoff |
| `pastis/oneway_cost.csv` | GMACs of the second pass per attention mode (exp.mistr.oneway_cost) | oneway_cost_table |
| `pastis/test_cost.csv` | GMACs per sample over the whole test split (exp.mistr.measure_test_cost) | baselines_r1, oneway_attn_trained, round1_main, round1_main_std, round1_throughput, season_range_tradeoff, selector_tables_r1, space_time_ablation, terramind_tradeoff |
| `pastis/throughput.csv` | measured GPU ms per sample (exp.bench.mistr_throughput) | round1_throughput |
| `pastis_year/mistr.csv` | one row per training run (exp.mistr.train): config, val/test metrics, GMACs | space_time_ablation |
| `planteur/inject_fft.csv` | earlier-code runs (the S2 and T_acq points of Fig 10 only) | space_time_ablation |
| `planteur/inject_flops.csv` | GMACs of those earlier-code runs | space_time_ablation |
| `planteur/mech_attn.csv` | attention mass per key group (exp.mistr.mech_probe) | oneway_probe_tables |
| `planteur/mech_decoder.csv` | decoder crop weight and pooled predictions (exp.mistr.mech_probe) | oneway_probe_tables |
| `planteur/mistr.csv` | one row per training run (exp.mistr.train): config, val/test metrics, GMACs | round1_main, round1_main_std, space_time_ablation, terramind_tradeoff |
| `planteur/test_cost.csv` | GMACs per sample over the whole test split (exp.mistr.measure_test_cost) | round1_main, round1_main_std, space_time_ablation, terramind_tradeoff |
| `planteur_year/mistr.csv` | one row per training run (exp.mistr.train): config, val/test metrics, GMACs | space_time_ablation |
| `tessera/pastis_frozen.csv` | frozen Tessera v2 heads (exp.tessera.head) | tessera_frozen_tradeoff |
| `yieldsat_raw/inject_fft.csv` | earlier-code runs (the S2 and T_acq points of Fig 10 only) | space_time_ablation |
| `yieldsat_raw/inject_flops.csv` | GMACs of those earlier-code runs | space_time_ablation |
| `yieldsat_raw/mech_attn.csv` | attention mass per key group (exp.mistr.mech_probe) | oneway_probe_tables |
| `yieldsat_raw/mech_decoder.csv` | decoder crop weight and pooled predictions (exp.mistr.mech_probe) | oneway_probe_tables |
| `yieldsat_raw/mistr.csv` | one row per training run (exp.mistr.train): config, val/test metrics, GMACs | baselines_r1, round1_main, round1_main_std, round1_throughput, space_time_ablation, terramind_tradeoff |
| `yieldsat_raw/test_cost.csv` | GMACs per sample over the whole test split (exp.mistr.measure_test_cost) | baselines_r1, round1_main, round1_main_std, round1_throughput, space_time_ablation, terramind_tradeoff |
| `yieldsat_raw/throughput.csv` | measured GPU ms per sample (exp.bench.mistr_throughput) | round1_throughput |
| `yieldsat_raw_year/mistr.csv` | one row per training run (exp.mistr.train): config, val/test metrics, GMACs | space_time_ablation |
