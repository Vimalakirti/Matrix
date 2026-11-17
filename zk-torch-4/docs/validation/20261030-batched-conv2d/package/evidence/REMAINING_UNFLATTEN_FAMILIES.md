# Actionable inherited issue outside the Conv2D reviewed scope

Final combined `zk-torch-4/src/morpheus/piop/cv.rs` anchors:
- Conv3D calls old `unflatten`: line 382.
- DepthwiseConv2D calls old `unflatten`: line 390.
- ConvTranspose2D calls old `unflatten`: line 397.
- ConvTranspose3D calls old `unflatten`: line 404.
- `unflatten` lines 549–556 blindly bypasses every FlattenKernel,
  FlattenKernel3D and ChangeShape producer.

These families were **not** covered by the new `conv2d_kernel` identity guard.
The independent Conv2D verdict must not be applied to them.

## Small source-level counterexample shared by all four

Take one input/output channel and a width-two kernel `[2,3]`, every other kernel
axis of length one. Perform the normal matching flatten; then use a registered
ChangeShape to truncate its two coefficients to one, followed by a ChangeShape
to grow back to two. The actual kernel reaching the conv is `[2,0]`. The old
unflatten walks through both width-changing views and restores `[2,3]`.

For Conv3D, use input `[D=1,H=1,W=3]` values `[1,2,3]`, unit stride, kernel
`[KD=1,KH=1,KW=2]`. Forward with actual flat weights gives `[2,4]`; the normal
reduction using the incorrectly bypassed raw weights proves `[8,13]`.
DepthwiseConv2D with `[H=1,W=3]` is the same counterexample.

For ConvTranspose2D, input `[H=1,W=2]` values `[1,2]`, kernel `[KH=1,KW=2]`,
unit stride gives actual logical output `[2,4,0]`, but bypassed raw weights give
`[2,7,6]`. ConvTranspose3D with depth-one input/kernel is identical. These are
small integer source/mapping counterexamples, not executed PIOP claims.

## Required matching-geometry checks / repair shape

At minimum, never skip a ChangeShape whose input/output padded arities differ.
Do not stop at a matching final arity: truncate then regrow is not identity.
Before bypassing a flatten producer, verify all tap counts, axis ordering,
channel counts and spatial strides, and reject zero/overflow/noninjective
scatter geometry. Otherwise retain the actual flat edge and prove a public tap
map (as the new Conv2D path does).

- Conv3D: `FlattenKernel3D.stride_w=Wp`, `stride_h=Hp*Wp`, tap counts match
  `(KD,KH,KW)`, fields `(c_out,c_in)=(D,C)`.
- DepthwiseConv2D: `FlattenKernel.s_w=Wp`, tap counts match, dilation=(1,1),
  fields `(c_out,c_in)=(channels,1)`.
- ConvTranspose2D: `FlattenKernel.s_w=ConvTranspose2D.flat_stride=Woutp`,
  tap counts match, dilation=(1,1), fields `(c_out,c_in)=(C_in,C_out)` because
  the actual transposed weight table is `[C_in,C_out,...]`.
- ConvTranspose3D: output-grid flat strides `(Houtp*Woutp,Woutp)`, tap counts
  match, fields `(c_out,c_in)=(C_in,C_out)`.

The new optional flat-kernel relation in `conv` currently asserts
`WKind::Normal`. Supporting the transpose/depthwise fallback needs deliberate
matching of its channel offsets/axes, not just removing that assertion.
