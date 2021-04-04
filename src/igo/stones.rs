use super::*;
use std::fmt;
use std::fmt::{Display, Formatter};
use std::ops::{Not};
use std::cmp::PartialEq;
use auto_ops::impl_op;

impl Stones {
  pub fn new(size: BoardSize) -> Stones {
    match size {
      BoardSize::S5 => Stones::A32(0),
      BoardSize::S7 => Stones::A64(0),
      BoardSize::S9 => Stones::A128(0),
    }
  }
  pub fn new32(val: u32) -> Stones {
    Stones::A32(val)
  }
  pub fn new64(val: u64) -> Stones {
    Stones::A64(val)
  }
  pub fn new128(val: u128) -> Stones {
    Stones::A128(val)
  }
  pub fn count_ones(&self) -> u32 {
    match self {
      Stones::A32(val) => val.count_ones(),
      Stones::A64(val) => val.count_ones(),
      Stones::A128(val) => val.count_ones(),
    }
  }
}

macro_rules! add_op {
  ($opr:tt, $left:ty) => {
    impl_op!($opr |a: $left, b: usize| -> $left {
      match a {
        Stones::A32(n) => Stones::A32(n $opr b as u32),
        Stones::A64(n) => Stones::A64(n $opr b as u64),
        Stones::A128(n) => Stones::A128(n $opr b as u128),
      }
    });
  };
}
macro_rules! add_op_stone {
  ($opr:tt, $left:ty, $right:ty) => {
    impl_op!($opr |a: $left, b: $right| -> $left { 
      match (a, b) {
        (Stones::A32(n1), Stones::A32(n2)) => Stones::A32(n1 $opr n2),
        (Stones::A64(n1), Stones::A64(n2)) => Stones::A64(n1 $opr n2),
        (Stones::A128(n1), Stones::A128(n2)) => Stones::A128(n1 $opr n2),
        _ => panic!("stones type not matched"),
      }
    });
  };
}
macro_rules! add_eq {
  ($($t:ty)*) => ($(
    impl PartialEq<$t> for Stones {
      fn eq(&self, rhs: &$t) -> bool {
        match self {
          Stones::A32(n1) => n1 == &(*rhs as u32),
          Stones::A64(n1) => n1 == &(*rhs as u64),
          Stones::A128(n1) => n1 == &(*rhs as u128),
        }
      }
    }
  )*)
}

add_op!(>>, Stones);
add_op!(<<, Stones);
add_op!(&, Stones);
add_op!(|, Stones);
add_op!(^, Stones);
add_op!(+, Stones);
add_op_stone!(&, Stones, Stones);
add_op_stone!(|, Stones, Stones);
add_op_stone!(^, Stones, Stones);
add_eq! {i32 u32}

impl Not for Stones {
  type Output = Stones;
  fn not(self) -> Stones {
    match self {
      Stones::A32(n1) => Stones::A32(!n1),
      Stones::A64(n1) => Stones::A64(!n1),
      Stones::A128(n1) => Stones::A128(!n1),
    }
  }
}

impl PartialEq for Stones {
  fn eq(&self, rhs: &Stones) -> bool {
    match (self, rhs) {
      (Stones::A32(n1), Stones::A32(n2)) => n1 == n2,
      (Stones::A64(n1), Stones::A64(n2)) => n1 == n2,
      (Stones::A128(n1), Stones::A128(n2)) => n1 == n2,
      _ => panic!("stones type not matched"),
    }
  }
}

impl Display for Stones {
  fn fmt(&self, f: &mut Formatter) -> fmt::Result {
    let size = match self {
      Stones::A32(_) => 5,
      Stones::A64(_) => 7,
      Stones::A128(_) => 9,
    };
    let mut bstr = "".to_string();
    for y in 0..size {
      for x in 0..size {
        let a = x + y * size;
        let b = *self >> a & 1;
        if b == 1 {
          bstr.push_str("o");
        } else {
          bstr.push_str("-");
        }
      }
      bstr.push('\n');
    }
    write!(f, "{}", bstr)
  }
}
