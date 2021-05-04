
use az_game::igo::*;

#[cfg(test)]
mod stones {
  use super::*;

  #[test]
  fn add_stone64() {
    let b = Stones::new64(0b0000000_1111111_1111111_1111111_1111111_1111111_1111111);
    assert_eq!(
      format!("{}", b + 1),
      "-------\n\
      -------\n\
      -------\n\
      -------\n\
      -------\n\
      -------\n\
      o------\n"
    );
  }
  #[test]
  fn add_stone128() {
    let b = Stones::new128(
      0b000000000_111111111_111111111_111111111_111111111_111111111_111111111_111111111_111111111,
    );
    assert_eq!(
      format!("{}", b + 1),
      "---------\n\
      ---------\n\
      ---------\n\
      ---------\n\
      ---------\n\
      ---------\n\
      ---------\n\
      ---------\n\
      o--------\n"
    );
  }
  #[test]
  fn flip_diag32_0() {
    let b = Stones::new32(
      0b11100_00101_01011_01110_01000
    );
    assert_eq!(
      format!("{}", b.flip_diag()),
       "o----\n\
        o-ooo\n\
        oo-o-\n\
        --oo-\n\
        -oo--\n"
    );
  }
  #[test]
  fn flip_diag32_1() {
    let b = Stones::new32(
      0b11100_11100_11100_11100_11100
    );
    assert_eq!(
      format!("{}", b.flip_diag()),
       "ooooo\n\
        ooooo\n\
        ooooo\n\
        -----\n\
        -----\n"
    );
  }
  #[test]
  fn flip_diag32_2() {
    let b = Stones::new32(
      0b00000_01110_01010_01010_01110
    );
    assert_eq!(
      format!("{}", b.flip_diag()),
       "-----\n\
        -oooo\n\
        -o--o\n\
        -oooo\n\
        -----\n"
    );
  }
  #[test]
  fn flip_diag64_1() {
    let b = Stones::new64(
      0b1100000_1100000_1100000_1100000_1100000_1100000_1100000
    );
    assert_eq!(
      format!("{}", b.flip_diag()),
       "ooooooo\n\
        ooooooo\n\
        -------\n\
        -------\n\
        -------\n\
        -------\n\
        -------\n"
    );
  }
  #[test]
  fn flip_diag64_2() {
    let b = Stones::new64(
      0b0000000_0111110_0100010_0100010_0100010_0100010_0111110
    );
    assert_eq!(
      format!("{}", b.flip_diag()),
        "-------\n\
        -oooooo\n\
        -o----o\n\
        -o----o\n\
        -o----o\n\
        -oooooo\n\
        -------\n"
    );
  }
  #[test]
  fn flip_diag128_1() {
    let b = Stones::new128(
      0b110000000_110000000_110000000_110000000_110000000_110000000_110000000_110000000_110000000
    );
    assert_eq!(
      format!("{}", b.flip_diag()),
       "ooooooooo\n\
        ooooooooo\n\
        ---------\n\
        ---------\n\
        ---------\n\
        ---------\n\
        ---------\n\
        ---------\n\
        ---------\n"
    );
  }
  #[test]
  fn flip_diag128_2() {
    let b = Stones::new128(
      0b000000000_011111110_010000010_010000010_010000010_010000010_010000010_010000010_011111110
    );
    assert_eq!(
      format!("{}", b.flip_diag()),
       "---------\n\
        -oooooooo\n\
        -o------o\n\
        -o------o\n\
        -o------o\n\
        -o------o\n\
        -o------o\n\
        -oooooooo\n\
        ---------\n"
    );
  }
  #[test]
  fn flip_vert32() {
    let b = Stones::new32(
      0b10101_01010_11111_00001_00010
    );
    assert_eq!(
      format!("{}", b.flip_vert()),
       "o-o-o\n\
        -o-o-\n\
        ooooo\n\
        o----\n\
        -o---\n"
    );
  }
  #[test]
  fn flip_vert64() {
    let b = Stones::new64(
      0b1010101_0101010_1111111_0000001_0000010_0000001_0000010
    );
    assert_eq!(
      format!("{}", b.flip_vert()),
       "o-o-o-o\n\
        -o-o-o-\n\
        ooooooo\n\
        o------\n\
        -o-----\n\
        o------\n\
        -o-----\n"
    );
  }
  #[test]
  fn flip_vert128() {
    let b = Stones::new128(
      0b101010101_010101010_111111111_111111111_000000001_000000010_000000001_000000010_000000001
    );
    assert_eq!(
      format!("{}", b.flip_vert()),
       "o-o-o-o-o\n\
        -o-o-o-o-\n\
        ooooooooo\n\
        ooooooooo\n\
        o--------\n\
        -o-------\n\
        o--------\n\
        -o-------\n\
        o--------\n"
    );
  }
}