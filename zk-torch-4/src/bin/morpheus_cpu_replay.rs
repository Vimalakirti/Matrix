//! Genuine host masked-PIOP replay and direct emitted-row evaluation.
//! This is not an authenticated PCS proof or GPU test.
fn main() {
    assert!(!cfg!(feature="cuda"), "use --no-default-features --features cpu-replay");
    assert!(!zk_torch_4::morpheus::piop::gpu::enabled());
    println!("RUN batched_conv_cpu_semantics_and_explicit_shape"); zk_torch_4::morpheus::piop::cv_batch_tests::batched_conv_cpu_semantics_and_explicit_shape(); println!("PASS batched_conv_cpu_semantics_and_explicit_shape");
    println!("RUN batched_conv_piop_masks_padding_and_rejects_false_outputs"); zk_torch_4::morpheus::piop::cv_batch_tests::batched_conv_piop_masks_padding_and_rejects_false_outputs(); println!("PASS batched_conv_piop_masks_padding_and_rejects_false_outputs");
    println!("RUN batched_conv_piop_rejects_padded_output_and_full_advice"); zk_torch_4::morpheus::piop::cv_batch_tests::batched_conv_piop_rejects_padded_output_and_full_advice(); println!("PASS batched_conv_piop_rejects_padded_output_and_full_advice");
    println!("RUN batched_conv_piop_stride_dilation_and_shared_kernel"); zk_torch_4::morpheus::piop::cv_batch_tests::batched_conv_piop_stride_dilation_and_shared_kernel(); println!("PASS batched_conv_piop_stride_dilation_and_shared_kernel");
    println!("RUN batched_conv_actual_field_tables_match_forward"); zk_torch_4::morpheus::piop::cv_batch_tests::batched_conv_actual_field_tables_match_forward(); println!("PASS batched_conv_actual_field_tables_match_forward");
    println!("RUN conv_kernel_source_cpu_and_collision_guards"); zk_torch_4::morpheus::piop::cv_batch_tests::conv_kernel_source_cpu_and_collision_guards(); println!("PASS conv_kernel_source_cpu_and_collision_guards");
    println!("RUN conv_kernel_source_piop_preserves_nonidentity_producers"); zk_torch_4::morpheus::piop::cv_batch_tests::conv_kernel_source_piop_preserves_nonidentity_producers(); println!("PASS conv_kernel_source_piop_preserves_nonidentity_producers");
    println!("RUN exact_maxpool_piop_binds_advice_output_selector_and_terminal_public_root"); zk_torch_4::morpheus::piop::maxpool_tests::exact_maxpool_piop_binds_advice_output_selector_and_terminal_public_root(); println!("PASS exact_maxpool_piop_binds_advice_output_selector_and_terminal_public_root");
    println!("RUN exact_maxpool_piop_rejects_changed_registered_support"); zk_torch_4::morpheus::piop::maxpool_tests::exact_maxpool_piop_rejects_changed_registered_support(); println!("PASS exact_maxpool_piop_rejects_changed_registered_support");
    println!("RUN pillar_maxpool_piop_attainment_source_width_and_public_root"); zk_torch_4::morpheus::piop::maxpool_tests::pillar_maxpool_piop_attainment_source_width_and_public_root(); println!("PASS pillar_maxpool_piop_attainment_source_width_and_public_root");
    println!("RUN family_sources_and_field_maps"); zk_torch_4::morpheus::piop::cv::family_tests::family_sources_and_field_maps(); println!("PASS family_sources_and_field_maps");
    println!("RUN family_geometry_and_collision_guards"); zk_torch_4::morpheus::piop::cv::family_tests::family_geometry_and_collision_guards(); println!("PASS family_geometry_and_collision_guards");
    println!("RUN family_truncate_regrow_counterexamples"); zk_torch_4::morpheus::piop::cv::family_tests::family_truncate_regrow_counterexamples(); println!("PASS family_truncate_regrow_counterexamples");
    println!("RUN family_masked_piop_source_replay"); zk_torch_4::morpheus::piop::cv::family_tests::family_masked_piop_source_replay(); println!("PASS family_masked_piop_source_replay");
    println!("RUN cpu_replay_rejects_tampered_transcript_and_zero_skip_is_exact");
    zk_torch_4::morpheus::piop::cv_batch_tests::cpu_replay_rejects_tampered_transcript_and_zero_skip_is_exact();
    println!("PASS cpu_replay_rejects_tampered_transcript_and_zero_skip_is_exact");
    println!("RUN honest_prover_refuses_recomputed_false_max");
    zk_torch_4::morpheus::piop::maxpool_tests::honest_prover_refuses_recomputed_false_max();
    println!("PASS honest_prover_refuses_recomputed_false_max (prover refusal only)");
}
