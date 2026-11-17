//! Exact bounded InstanceNorm3D, preserving the quantized affine layout.
//! See docs/morpheus_instance_norm_profile.md for domains, lifting, and the
//! deliberate difference from the old unchecked floating statistics helper.
use super::{DagBuilder, EdgeId};
use crate::basicblock::llama::{floor_sqrt, rms_bounds, RMS_MEAN_MAX};

/// Exactly round the nonnegative IEEE-754 input times 2^sf, ties upward.
/// No floating multiplication, ambient rounding mode, or minimum-one epsilon.
/// A positive epsilon may quantize to zero; that selects the documented zero
/// statistic extension rather than silently changing the requested epsilon.
pub fn epsilon_units(eps: f64, sf: usize) -> i128 {
    assert!(sf <= 20, "InstanceNorm supports activation scales 0..20");
    assert!(eps.is_finite() && eps >= 0.0, "InstanceNorm epsilon must be finite and nonnegative");
    let bits = eps.to_bits();
    let exponent = ((bits >> 52) & 2047) as i32;
    let fraction = bits & ((1u64 << 52) - 1);
    let mantissa = if exponent == 0 { fraction } else { fraction | (1u64 << 52) } as i128;
    let power = if exponent == 0 { -1074 } else { exponent - 1023 - 52 } + sf as i32;
    let units = if mantissa == 0 { 0 } else if power >= 0 {
        // Reject before shifting: checked_shl only checks shift width, not
        // significant bits lost off the top of a signed machine integer.
        assert!(power <= 27 && mantissa <= (RMS_MEAN_MAX >> power), "epsilon outside public statistic domain");
        mantissa << power
    } else {
        let shift = (-power) as usize;
        if shift >= 55 { 0 } else { (mantissa + (1i128 << (shift - 1))) >> shift }
    };
    assert!((0..=RMS_MEAN_MAX).contains(&units), "epsilon outside public statistic domain");
    units
}

/// Public bounds; none is inferred from observed witness values.
#[derive(Clone, Copy, Debug)]
pub struct InstanceNormBounds {
    pub input: i128,
    pub parameter: i128,
    pub scale: i128,
    pub scaled_input: i128,
}

pub fn instance_norm_bounds(sf: usize, epsilon: i128) -> InstanceNormBounds {
    let (rmax, _, _) = rms_bounds(sf);
    assert!((0..=RMS_MEAN_MAX).contains(&epsilon), "epsilon outside public statistic domain");
    let q = 1i128 << sf;
    let input = floor_sqrt((RMS_MEAN_MAX - epsilon) * q) / 2;
    let parameter = (1i128 << 24) - 1;
    let scale = (parameter * rmax + q - 1) / q;
    let scaled_input = (scale * input + q - 1) / q;
    // Each ScaleDown identity has independent bounds on the raw product,
    // quotient, and selector. The entire integer residual is < p, not only
    // the honest product. These conservative limits cover every s in 0..20.
    assert!(4*input*input + q*(RMS_MEAN_MAX-epsilon) + 2*q < (1i128 << 50));
    assert!(parameter * rmax + q * scale + 2*q < (1i128 << 57));
    assert!(scale * input + q * scaled_input + 2*q < (1i128 << 59));
    assert!(2 * (parameter + 2 * scaled_input) < (1i128 << 62));
    InstanceNormBounds { input, parameter, scale, scaled_input }
}

impl DagBuilder {
    /// X[C,D,H,W], gamma[C], beta[C] -> Y[C,D,H,W], CNN row-major.
    /// eps is rounded EXACTLY from its binary f64 value into activation units.
    /// For explicit integer epsilon selection use `instancenorm3d_units`.
    pub fn instancenorm3d(&mut self, x: EdgeId, gamma: EdgeId, beta: EdgeId, eps: f64) -> EdgeId {
        let sf = self.init_values[x].as_ref().unwrap().sf;
        self.instancenorm3d_units(x, gamma, beta, epsilon_units(eps, sf))
    }

    /// Exact affine profile: mu=RD(sum x,n), z_i=RD((x_i-mu)^2,Q),
    /// v=RD(sum z,n), r=RN(sqrt(Q^3/(v+epsilon))) (zero ->0),
    /// a=RD(gamma*r,Q), o=beta-RD(a*mu,Q), y=RD(a*x,Q)+o.
    /// All arithmetic uses logical coordinates; output padding is zero.
    pub fn instancenorm3d_units(&mut self, x: EdgeId, gamma: EdgeId, beta: EdgeId, epsilon: i128) -> EdgeId {
        let shape = self.init_values[x].as_ref().unwrap().shape.clone();
        assert_eq!(shape.len(), 4, "InstanceNorm requires [C,D,H,W]");
        assert!(shape.iter().all(|&d| d > 0), "InstanceNorm dimensions must be positive");
        let sf = self.init_values[x].as_ref().unwrap().sf;
        let channels = shape[0];
        let count = shape[1..].iter().try_fold(1usize, |n, &d| n.checked_mul(d)).expect("InstanceNorm spatial size overflow");
        assert!((1..=(1 << 24)).contains(&count), "InstanceNorm spatial count outside 1..2^24");
        for e in [gamma, beta] {
            let w = self.init_values[e].as_ref().unwrap();
            assert_eq!(w.shape, vec![channels], "InstanceNorm affine parameter shape mismatch");
            assert_eq!(w.sf, sf, "InstanceNorm affine parameter scale mismatch");
        }
        let b = instance_norm_bounds(sf, epsilon);
        // Einsum/Add/mask use column-major dimensions. Reverse only the shape
        // metadata so their indexing agrees with CNN row-major storage; this
        // is NOT a transpose. C remains the slowest-varying dimension.
        let view = vec![shape[3], shape[2], shape[1], channels];
        let xv = self.change_shape(x, view.clone());
        let xv = self.mask(xv, view.clone());
        self.integer_interval(xv, -b.input, b.input);
        let gamma = self.mask(gamma, vec![channels]);
        let beta = self.mask(beta, vec![channels]);
        self.integer_interval(gamma, -b.parameter, b.parameter);
        self.integer_interval(beta, -b.parameter, b.parameter);

        let sum = self.einsum("whdc->c".into(), vec![xv], false)[0];
        let mean = self.div_const(sum, count)[0];
        self.integer_interval(mean, -b.input, b.input);
        let mean4 = self.change_shape(mean, vec![1, 1, 1, channels]);
        let centered = self.sub(xv, mean4)[0];
        let centered = self.mask(centered, view.clone());
        self.integer_interval(centered, -2 * b.input, 2 * b.input);
        let sq = self.einsum("whdc,whdc->whdc".into(), vec![centered, centered], true)[0];
        self.integer_interval(sq, 0, RMS_MEAN_MAX - epsilon);
        let square_sum = self.einsum("whdc->c".into(), vec![sq], false)[0];
        let variance = self.div_const(square_sum, count)[0];
        self.integer_interval(variance, 0, RMS_MEAN_MAX - epsilon);
        let eps = self.integer_literal(vec![channels], epsilon, sf);
        let statistic = self.add(variance, eps)[0];
        let statistic = self.mask(statistic, vec![channels]);
        let reciprocal = self.rms_reciprocal_from_mean(statistic);

        // These consumers bind gamma/beta to the actual proved affine map;
        // neither is pruned as an advice-only input or detached model weight.
        let scale = self.einsum("c,c->c".into(), vec![gamma, reciprocal], true)[0];
        self.integer_interval(scale, -b.scale, b.scale);
        let scaled_mean = self.einsum("c,c->c".into(), vec![scale, mean], true)[0];
        self.integer_interval(scaled_mean, -b.scaled_input, b.scaled_input);
        let offset = self.sub(beta, scaled_mean)[0];
        self.integer_interval(offset, -b.parameter - b.scaled_input, b.parameter + b.scaled_input);
        let scaled = self.einsum("whdc,c->whdc".into(), vec![xv, scale], true)[0];
        self.integer_interval(scaled, -b.scaled_input, b.scaled_input);
        let offset4 = self.change_shape(offset, vec![1, 1, 1, channels]);
        let y = self.add(scaled, offset4)[0];
        self.integer_interval(y, -b.parameter - 2*b.scaled_input, b.parameter + 2*b.scaled_input);
        let y = self.mask(y, view);
        self.change_shape(y, shape)
    }
}
