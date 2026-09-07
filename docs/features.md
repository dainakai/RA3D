# Feature catalog

The following table lists the exact **92-column full-model order**. Screening columns form a 24-element subset, static columns form a 77-element subset, and 15 columns depend on reconstructed images. Select features using the exported lists in `ra3d.config`; do not infer the input set from every numeric table column.

For diameter summaries, the expected merged diameter is `(Dv**3 + Ds**3)**(1/3)`. The pre-window uses up to ten **observations** at or before f−2, which can extend farther back than ten frames in a gapped track. The post-window uses up to twenty observations after f. Kinematic histories instead use simultaneous observations in the ten **frames** f−10 through f−1. Image windows use f−10 through f−2. Slopes use actual integer frame spacing.

| Full index | Column | Stage | Definition |
| ---: | --- | --- | --- |
| 1 | `time_offset_frame` | Screening + static + full | Candidate frame minus vanished-track endpoint frame. |
| 2 | `xy_dist_um` | Screening + static + full | XY separation of the candidate particle positions. |
| 3 | `z_dist_um` | Screening + static + full | Absolute Z separation. |
| 4 | `closest_distance_um` | Screening + static + full | Euclidean three-dimensional separation. |
| 5 | `effective_distance_um` | Screening + static + full | sqrt(XY separation squared + (0.15 * Z separation) squared). |
| 6 | `radius_sum_um` | Screening + static + full | Sum of the two particle radii. |
| 7 | `distance_margin_um` | Screening + static + full | Euclidean separation minus the radius sum. |
| 8 | `effective_margin_um` | Screening + static + full | Effective separation minus the radius sum. |
| 9 | `vanished_diameter_um` | Screening + static + full | Vanishing-particle diameter used at candidate generation. |
| 10 | `survivor_diameter_um` | Screening + static + full | Surviving-particle diameter used at candidate generation. |
| 11 | `closing_speed_um_frame` | Screening + static + full | Relative velocity projected toward the interparticle displacement. |
| 12 | `post_survival_frames` | Screening + static + full | Survivor duration after the vanished endpoint. |
| 13 | `pre_overlap_frames` | Screening + static + full | Duration for which the two tracks overlap before the endpoint. |
| 14 | `boundary_exit_score` | Screening + static + full | Clipped proximity to XY/Z boundaries or recording end. |
| 15 | `survivor_change_score` | Screening + static + full | Capped combination of pre/post diameter and velocity change. |
| 16 | `survivor_delta_diameter_px` | Screening + static + full | Post minus pre survivor diameter in the short change window. |
| 17 | `survivor_delta_speed_px_frame` | Screening + static + full | Norm of the componentwise pre/post velocity change in legacy coordinate units. |
| 18 | `vertical_y_gap_um` | Screening + static + full | Signed vertical separation used by the catching diagnostic. |
| 19 | `vertical_closing_um_frame` | Screening + static + full | Relative vertical velocity toward catching. |
| 20 | `vertical_time_to_catch_frame` | Screening + static + full | Estimated vertical catching time. |
| 21 | `vertical_penalty` | Screening + static + full | Penalty from unfavorable vertical geometry or catching time. |
| 22 | `xy_gate_ratio_400` | Screening + static + full | XY separation / 400 µm. |
| 23 | `z_gate_ratio_6000` | Screening + static + full | Z separation / 6,000 µm. |
| 24 | `xy_z_gate_value_400_6000` | Screening + static + full | Squared XY/400 and Z/6,000 ratios, summed. |
| 25 | `pre_diameter_vanished_um` | Static + full | Median of up to ten valid vanishing-track observations at or before f−2. |
| 26 | `pre_diameter_survivor_um` | Static + full | Median of up to ten valid survivor observations at or before f−2. |
| 27 | `pre_diameter_large_um` | Static + full | Larger of the two available pre-event median diameters. |
| 28 | `pre_diameter_small_um` | Static + full | Smaller of the two available pre-event median diameters. |
| 29 | `post_diameter_um` | Static + full | Median of the first twenty valid survivor observations after f. |
| 30 | `pre_diameter_vanished_n` | Static + full | Number of valid observations in the vanished pre-event diameter estimate. |
| 31 | `pre_diameter_survivor_n` | Static + full | Number of valid observations in the survivor pre-event diameter estimate. |
| 32 | `post_diameter_n` | Static + full | Number of valid observations in the post-event diameter estimate. |
| 33 | `post_over_large` | Static + full | Post-event diameter / larger pre-event diameter. |
| 34 | `post_over_expected` | Static + full | Post-event diameter / expected merged diameter. |
| 35 | `post_minus_large` | Static + full | Post-event diameter minus larger pre-event diameter (µm). |
| 36 | `post_minus_expected` | Static + full | Post-event diameter minus expected merged diameter (µm). |
| 37 | `volume_resid_frac` | Static + full | Absolute fractional difference between expected and observed post-event volume. |
| 38 | `signed_volume_resid_frac` | Static + full | (Expected diameter cubed − post diameter cubed) / expected diameter cubed. |
| 39 | `small_large_ratio` | Static + full | Smaller / larger pre-event median diameter. |
| 40 | `expected_growth_um` | Static + full | Expected merged diameter minus larger pre-event diameter. |
| 41 | `growth_fraction_of_expected` | Static + full | Observed diameter growth / expected growth; NaN when the denominator is near zero. |
| 42 | `pre_n_min` | Static + full | Minimum of the two pre-event diameter observation counts. |
| 43 | `pre_kin_both_count` | Static + full | Number of simultaneously observed trajectory points in f−10 through f−1. |
| 44 | `pre_kin_span_frames` | Static + full | Last minus first simultaneously observed pre-event frame. |
| 45 | `pre_eff_first_um` | Static + full | First effective separation in the simultaneous pre-event history. |
| 46 | `pre_eff_last_um` | Static + full | Last effective separation in the simultaneous pre-event history. |
| 47 | `pre_eff_min_um` | Static + full | Minimum effective separation in the simultaneous pre-event history. |
| 48 | `pre_eff_max_um` | Static + full | Maximum effective separation in the simultaneous pre-event history. |
| 49 | `pre_eff_last_minus_first_um` | Static + full | Last minus first effective separation in the simultaneous pre-event history. |
| 50 | `pre_eff_slope_um_frame` | Static + full | Least-squares slope of effective separation in the simultaneous pre-event history. |
| 51 | `pre_eff_tail_slope_um_frame` | Static + full | Least-squares slope over the last four points of effective separation in the simultaneous pre-event history. |
| 52 | `pre_eff_increase_fraction` | Static + full | Fraction of positive consecutive differences in effective separation in the simultaneous pre-event history. |
| 53 | `pre_eff_tail_increase_fraction` | Static + full | Fraction of positive consecutive differences over the last four points in effective separation in the simultaneous pre-event history. |
| 54 | `pre_eff_event_minus_last_um` | Static + full | Candidate value minus last value of effective separation in the simultaneous pre-event history. |
| 55 | `pre_eff_event_minus_min_um` | Static + full | Candidate value minus minimum of effective separation in the simultaneous pre-event history. |
| 56 | `pre_xy_first_um` | Static + full | First XY separation in the simultaneous pre-event history. |
| 57 | `pre_xy_last_um` | Static + full | Last XY separation in the simultaneous pre-event history. |
| 58 | `pre_xy_min_um` | Static + full | Minimum XY separation in the simultaneous pre-event history. |
| 59 | `pre_xy_max_um` | Static + full | Maximum XY separation in the simultaneous pre-event history. |
| 60 | `pre_xy_last_minus_first_um` | Static + full | Last minus first XY separation in the simultaneous pre-event history. |
| 61 | `pre_xy_slope_um_frame` | Static + full | Least-squares slope of XY separation in the simultaneous pre-event history. |
| 62 | `pre_xy_tail_slope_um_frame` | Static + full | Least-squares slope over the last four points of XY separation in the simultaneous pre-event history. |
| 63 | `pre_xy_increase_fraction` | Static + full | Fraction of positive consecutive differences in XY separation in the simultaneous pre-event history. |
| 64 | `pre_xy_tail_increase_fraction` | Static + full | Fraction of positive consecutive differences over the last four points in XY separation in the simultaneous pre-event history. |
| 65 | `pre_xy_event_minus_last_um` | Static + full | Candidate value minus last value of XY separation in the simultaneous pre-event history. |
| 66 | `pre_xy_event_minus_min_um` | Static + full | Candidate value minus minimum of XY separation in the simultaneous pre-event history. |
| 67 | `pre_z_first_um` | Static + full | First absolute Z separation in the simultaneous pre-event history. |
| 68 | `pre_z_last_um` | Static + full | Last absolute Z separation in the simultaneous pre-event history. |
| 69 | `pre_z_last_minus_first_um` | Static + full | Last minus first absolute Z separation in the simultaneous pre-event history. |
| 70 | `pre_z_slope_um_frame` | Static + full | Least-squares slope of absolute Z separation in the simultaneous pre-event history. |
| 71 | `pre_closing_eff_last_um_frame` | Static + full | Last observed effective-distance closing speed. |
| 72 | `pre_closing_xy_last_um_frame` | Static + full | Last observed XY closing speed. |
| 73 | `pre_closing_eff_median_um_frame` | Static + full | Median observed effective-distance closing speed. |
| 74 | `pre_closing_xy_median_um_frame` | Static + full | Median observed XY closing speed. |
| 75 | `rendered_pre_frames` | Image + full | Number of reconstructed pre-event crops in f−10 through f−2. |
| 76 | `skipped_pre_frames` | Image + full | Number of requested pre-event frames unavailable to the crop construction. |
| 77 | `pre_same_interp_count` | Image + full | Count of frames with both interpolated track positions associated with one component. |
| 78 | `pre_same_interp_fraction` | Image + full | Interpolated common-component count / rendered frame count, or zero when none rendered. |
| 79 | `pre_same_observed_count` | Image + full | Common-component count using actual observed points. |
| 80 | `pre_alternating_same_count` | Image + full | Common-component count with interpolated positions when exactly one of the two tracks has an observed point. |
| 81 | `same_component_area_median_px` | Image + full | Median pixel area of associated common components, or zero. |
| 82 | `same_component_score` | Image + full | Same value as pre_same_interp_fraction; retained for the frozen schema. |
| 83 | `fragment_pre_full_v1` | Image + full | Interpolated common-component count ≥9, duplicated by same_ge9. |
| 84 | `fragment_pre8_v1` | Image + full | Interpolated common-component count ≥8, duplicated by same_ge8. |
| 85 | `fragment_pre7_v1` | Image + full | Interpolated common-component count ≥7, duplicated by same_ge7. |
| 86 | `same_ge7` | Image + full | Interpolated common-component count ≥7. |
| 87 | `same_ge8` | Image + full | Interpolated common-component count ≥8. |
| 88 | `same_ge9` | Image + full | Interpolated common-component count ≥9. |
| 89 | `diameter_under_0p725` | Static + full | Post/expected <0.725, at least eight post observations, and small/large ≥0.4. |
| 90 | `diameter_under_0p90` | Static + full | Post/expected <0.90, at least three post observations, and small/large ≥0.4. |
| 91 | `same7_and_diameter_under_0p90` | Image + full | Both same_ge7 and diameter_under_0p90. |
| 92 | `moving_apart_xy_rule` | Static + full | At least five paired pre-points, XY slope ≥5 µm/frame, XY growth ≥20 µm and increase fraction ≥0.75. |

## Missingness and auxiliary columns

Missing measurements remain NaN. Raw image aggregates are NaN when the image-acquisition policy did not select that candidate; the four image-derived indicators then evaluate to zero. For an acquired candidate, zero rendered frames are recorded explicitly rather than fabricated as a positive measurement. The original aggregation formulas are in [features.py](../src/ra3d/features.py), [feature_primitives.py](../src/ra3d/_core/feature_primitives.py) and [track_images.py](../src/ra3d/_core/track_images.py).

Keys (`vanished_track_id, survivor_track_id, candidate_frame`), geometry (`x_um, y_um, z_um`), ranks, scores, sample weights, truth labels, and `image_enriched` are auxiliary table fields. They are used for grouping, policy, supervision or audits, not blindly appended to the HGB feature matrix. `vanished_end_frame` is required by output provenance and label association.

## Relation features

Relation inputs are derived from the full-feature table and scene OOF base scores. The derived context includes candidate ranks, margins, distances, geometry and base logits; relative inputs compare a candidate with its set. The 3D input has 36 context and ten relative components, defined in [context.py](../src/ra3d/_core/context.py). Training-set means and standard deviations are saved in the model. Context is recomputed from the published score arrays and table order; the nested arrays encode the required scene exclusions.

The 2D input has 33 context and eight relative components, defined in [two_d_relation.py](../src/ra3d/_core/two_d_relation.py). Its 49 HGB columns are listed in [two_d_common.py](../src/ra3d/_core/two_d_common.py). The two schemas and their normalization statistics are not interchangeable.
