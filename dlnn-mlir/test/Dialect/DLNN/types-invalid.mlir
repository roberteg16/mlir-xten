// RUN: dlnn-opt %s -split-input-file -verify-diagnostics

// expected-error@+1 {{invalid size}}
func.func private @fm_domain_invalid_size() -> !dlnn.fm<1xf32[-2]>

// -----

#semi_affine = affine_map<(d0)[N] -> (d0 * N)>

// expected-error@+1 {{not pure-affine}}
func.func private @fm_org_semi_affine() -> !dlnn.fm<1xf32, #semi_affine>

// -----

#sym_mismatch = affine_map<(d0) -> (d0)>

// expected-error@+1 {{!= type symbols (1)}}
func.func private @fm_org_sym_mismatch() -> !dlnn.fm<1xf32, #sym_mismatch>

// -----

#dim_mismatch = affine_map<(d0, d1)[N] -> (d0, d1)>

// expected-error@+1 {{!= type dims (1)}}
func.func private @fm_org_dim_mismatch() -> !dlnn.fm<1xf32, #dim_mismatch>

// -----

#org = affine_map<(C, W, H)[N] -> (H, C floordiv 8, W, N, C mod 8)>

// expected-error@+1 {{not surjective}}
func.func private @fm_org_not_surjective() -> !dlnn.fm<3xf32[108,108], #org>
