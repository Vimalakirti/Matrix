fn main(){
 if std::env::var_os("COMPACT_RESTRICTED_ONLY").is_some(){zk_torch_4::morpheus::auth_compact::functional_tests::restricted_proof_zero_extension_verifies();println!("PASS restricted proof real dense verification and suffix rejection");return;}
    zk_torch_4::morpheus::link_fast::bounded_cpu_tests::streaming_services_equal_dense_and_public_preflight();
    println!("PASS bounded link streaming services");
    zk_torch_4::morpheus::link_fast::bounded_cpu_tests::bounded_link_preserves_zero_query_free_binary_chi_and_transcript();
    println!("PASS bounded link exact transcript + query-free chi");
    if std::env::var_os("COMPACT_LINK_ONLY").is_some(){return}
    zk_torch_4::morpheus::public_support::functional_tests::all_allocated_pieces_and_suffix_are_included();
    println!("PASS public support admission");
    zk_torch_4::morpheus::wire::compact_tests::compact_framing_codes_and_virtual_zero_extension();
    println!("PASS compact transport and zero expansion");
    zk_torch_4::morpheus::auth_compact::functional_tests::original_index_algebra_and_sampler_equivalence();
    println!("PASS original_index_algebra_and_sampler_equivalence");
    zk_torch_4::morpheus::auth_compact::functional_tests::full_support_proof_matches_dense_and_real_negatives();
    println!("PASS full_support_proof_matches_dense_and_real_negatives");
    zk_torch_4::morpheus::auth_compact::functional_tests::restricted_proof_zero_extension_verifies();
    println!("PASS restricted proof real dense verification and suffix rejection");
}
