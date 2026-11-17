//! Wire/setup identity of the incompatible SHAKE256 candidate profile.
pub const PROTOCOL_VERSION: u64 = 4;
pub const MAGIC: &[u8; 8] = b"MRPHSv4\0";
pub fn accepts_wire_magic(bytes: &[u8]) -> bool { bytes == MAGIC }
#[cfg(test)] mod tests {
 use super::*;
 #[test] fn incompatible_wire_profiles_are_rejected() {
  assert!(accepts_wire_magic(b"MRPHSv4\0"));
  for b in [b"MRPHSv1\0",b"MRPHSv2\0",b"MRPHSv3\0",b"MRPHSv5\0"] { assert!(!accepts_wire_magic(b)); }
  assert!(!accepts_wire_magic(b"MRPHSv4"));
  assert!(!accepts_wire_magic(b"MRPHSv4\0extra"));
 }
}
