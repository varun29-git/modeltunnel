# Dry-run campaign plan (no jobs launched)

```
Campaign dry run — cap 60.0 H100-hours
Rates: {'1500m': 'measured synthetic (evidence/muon_25l_readme.md)', '200m': 'ASSUMED (unmeasured)', '700m': 'ASSUMED (unmeasured)'}; conservative factor 0.6

stage0   stage0/preflight_dense                                            0M tok      0 upd  exp   0.75h  cons   0.75h
stage0   stage0/preflight_ternary                                          0M tok      0 upd  exp   0.75h  cons   0.75h
stage1   stage1/200m/sqrt_fan_in/lr=0.001                                600M tok    287 upd  exp   1.18h  cons   1.92h
stage1   stage1/200m/sqrt_fan_in/lr=0.003                                600M tok    287 upd  exp   1.18h  cons   1.92h
stage1   stage1/200m/sqrt_fan_in/lr=0.009                                600M tok    287 upd  exp   1.18h  cons   1.92h
stage1   stage1/200m/sqrt_fan_in_ratio/lr=0.001                          600M tok    287 upd  exp   1.18h  cons   1.92h
stage1   stage1/200m/sqrt_fan_in_ratio/lr=0.003                          600M tok    287 upd  exp   1.18h  cons   1.92h
stage1   stage1/200m/sqrt_fan_in_ratio/lr=0.009                          600M tok    287 upd  exp   1.18h  cons   1.92h
stage2   stage2/700m/winner_1                                           1000M tok    477 upd  exp   4.11h  cons   6.75h
stage2   stage2/700m/winner_2                                           1000M tok    477 upd  exp   4.11h  cons   6.75h
stage2   stage2/1500m/transfer_check                                     500M tok    239 upd  exp   3.26h  cons   5.26h
stage3   stage3/200m/mix=baseline                                        600M tok    287 upd  exp   1.18h  cons   1.92h
stage3   stage3/200m/mix=indic_heavy                                     600M tok    287 upd  exp   1.18h  cons   1.92h
stage3   stage3/200m/mix=stem_heavy                                      600M tok    287 upd  exp   1.18h  cons   1.92h
stage3   stage3/200m/mix=en_heavy                                        600M tok    287 upd  exp   1.18h  cons   1.92h
stage4   stage4/200m/parent_dense                                        600M tok    287 upd  exp   1.18h  cons   1.92h
stage4   stage4/200m/qat/switch=0.5/dur=120000000                        120M tok     58 upd  exp   0.33h  cons   0.50h
stage4   stage4/200m/qat/switch=0.5/dur=240000000                        240M tok    115 upd  exp   0.58h  cons   0.92h
stage4   stage4/200m/qat/switch=0.8/dur=120000000                        120M tok     58 upd  exp   0.33h  cons   0.50h
stage4   stage4/200m/qat/switch=0.8/dur=240000000                        240M tok    115 upd  exp   0.58h  cons   0.92h
stage4   stage4/200m/control_dense_continuation                          120M tok     58 upd  exp   0.44h  cons   0.59h
stage4   stage4/200m/control_ternary_from_start                          720M tok    344 upd  exp   1.76h  cons   2.78h
stage5   stage5/200m/second_seed_1                                       600M tok    287 upd  exp   1.18h  cons   1.92h
stage5   stage5/200m/second_seed_2                                       600M tok    287 upd  exp   1.18h  cons   1.92h
         TOTAL                                                                             exp  32.35h  cons  51.47h
Cap check: conservative total 51.47h <= 60.0h
```
