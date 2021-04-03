use super::*;
use std::fmt;
use std::fmt::{Display, Formatter};
use std::ops::{Not, Add};
use std::cmp::PartialEq;
use auto_ops::impl_op;

impl Stones {
  pub fn new(size: BoardSize) -> Stones {
    match size {
      BoardSize::S5 => Stones::A32(0),
      BoardSize::S7 => Stones::A64(0, 0),
    }
  }
  pub fn new32(val: u32) -> Stones {
    Stones::A32(val)
  }
  pub fn new64(v1: u32, v2: u32) -> Stones {
    Stones::A64(v1, v2)
  }
  pub fn new96(v1: u32, v2: u32, v3: u32) -> Stones {
    Stones::A96(v1, v2, v3)
  }
}

macro_rules! add_op {
  ($opr:tt, $left:ty) => {
    impl_op!($opr |a: $left, b: usize| -> $left {
      let rhs = b as u32;
      match a {
        Stones::A32(n) => Stones::A32(n $opr rhs),
        Stones::A64(_, n2) => Stones::A64(0, n2 $opr rhs),
        Stones::A96(_, _, n3) => Stones::A96(0, 0, n3 $opr rhs),
      }
    });
  };
}
macro_rules! add_op_stone {
  ($opr:tt, $left:ty, $right:ty) => {
    impl_op!($opr |a: $left, b: $right| -> $left { 
      match (a, b) {
        (Stones::A32(n1), Stones::A32(n2)) => Stones::A32(n1 $opr n2),
        (Stones::A64(a1, a2), Stones::A64(b1, b2)) => Stones::A64(a1 $opr b1, a2 $opr b2),
        (Stones::A96(a1, a2, a3), Stones::A96(b1, b2, b3)) => Stones::A96(a1 $opr b1, a2 $opr b2, a3 $opr b3),
        _ => panic!("stones type not matched"),
      }
    });
  };
}
macro_rules! add_eq {
  ($($t:ty)*) => ($(
    impl PartialEq<$t> for Stones {
      fn eq(&self, rhs: &$t) -> bool {
        let val = &(*rhs as u32);
        match self {
          Stones::A32(n1) => n1 == val,
          Stones::A64(n1, n2) => *n1 == 0 && n2 == val,
          Stones::A96(n1, n2, n3) => *n1 == 0 && *n2 == 0 && n3 == val,
        }
      }
    }
  )*)
}

add_op!(&, Stones);
add_op!(|, Stones);
add_op!(^, Stones);
add_op_stone!(&, Stones, Stones);
add_op_stone!(|, Stones, Stones);
add_op_stone!(^, Stones, Stones);
add_eq! {i32 u32}

impl_op!(>> |a: Stones, b: usize| -> Stones {
  let rhs = b as u32;
  match a {
    Stones::A32(n) => Stones::A32(n >> rhs),
    Stones::A64(n1, n2) => {
      let mut s1 = 0;
      let mut s2 = 0;
      if rhs == 0 {
        s2 = n2;
        s1 = n1;
      } else if rhs < 32 {
        s2 = n2 >> rhs | n1 << (32 - rhs);
        s1 = n1 >> rhs;
      } else if rhs < 64 {
        s2 = n1 >> (rhs - 32);
      }
      Stones::A64(s1, s2)
    },
    Stones::A96(n1, n2, n3) => {
      let mut s1 = 0;
      let mut s2 = 0;
      let mut s3 = 0;
      if rhs == 0 {
        s3 = n3;
        s2 = n2;
        s1 = n1;
      } else if rhs < 32 {
        s3 = n3 >> rhs | n2 << (32 - rhs);
        s2 = n2 >> rhs | n1 << (32 - rhs);
        s1 = n1 >> rhs;
      } else if rhs == 32 {
        s3 = n2;
        s2 = n1;
      } else if rhs < 64 {
        s3 = n2 >> (rhs - 32) | n1 << (64 - rhs);
        s2 = n1 >> (rhs - 32);
      } else if rhs < 96 {
        s3 = n1 >> (rhs - 64);
      }
      Stones::A96(s1, s2, s3)
    },
  }
});
impl_op!(<< |a: Stones, b: usize| -> Stones {
  let rhs = b as u32;
  match a {
    Stones::A32(n) => Stones::A32(n << rhs),
    Stones::A64(n1, n2) => {
      let mut s1 = 0;
      let mut s2 = 0;
      if rhs == 0 {
        s1 = n1;
        s2 = n2;
      } else if rhs < 32 {
        s1 = n1 << rhs | n2 >> (32 - rhs);
        s2 = n2 << rhs;
      } else if rhs < 64{
        s1 = n2 << (rhs - 32);
      }
      Stones::A64(s1, s2)
    },
    Stones::A96(n1, n2, n3) => {
      let mut s1 = 0;
      let mut s2 = 0;
      let mut s3 = 0;
      if rhs == 0 {
        s1 = n1;
        s2 = n2;
        s3 = n3;
      } else if rhs < 32 {
        s1 = n1 << rhs | n2 >> (32 - rhs);
        s2 = n2 << rhs | n3 >> (32 - rhs);
        s3 = n3 << rhs;
      } else if rhs == 32 {
        s1 = n2;
        s2 = n3;
      } else if rhs < 64 {
        s1 = n2 << (rhs - 32) | n3 >> (64 - rhs);
        s2 = n3 << (rhs - 32);
      } else if rhs < 96 {
        s1 = n3 << (rhs - 64);
      }
      Stones::A96(s1, s2, s3)
    },
  }
});

impl Not for Stones {
  type Output = Stones;
  fn not(self) -> Stones {
    match self {
      Stones::A32(n1) => Stones::A32(!n1),
      Stones::A64(n1, n2) => Stones::A64(!n1, !n2),
      Stones::A96(n1, n2, n3) => Stones::A96(!n1, !n2, !n3),
    }
  }
}
impl Add<usize> for Stones {
  type Output = Stones;
  fn add(self, rhs: usize) -> Stones {
    let val = rhs as u32;
    match self {
      Stones::A32(n1) => Stones::A32(n1 + val),
      Stones::A64(n1, n2) => {
        let s2 = n2.wrapping_add(val);
        let mut s1 = n1;
        if (n2 & 0x80000000 == 0x80000000) && (s2 & 0x80000000 == 0) {
          s1 = s1.wrapping_add(1);
        }
        Stones::A64(s1, s2)
      },
      Stones::A96(n1, n2, n3) => {
        let s3 = n3.wrapping_add(val);
        let mut s2 = n2;
        let mut s1 = n1;
        if (n3 & 0x80000000 == 0x80000000) && (s3 & 0x80000000 == 0) {
          s2 = s2.wrapping_add(1);
          if (n2 & 0x80000000 == 0x80000000) && (s2 & 0x80000000 == 0) {
            s1 = s1.wrapping_add(1);
          }
        }
        Stones::A96(s1, s2, s3)
      },
    }
  }
}
impl PartialEq for Stones {
  fn eq(&self, rhs: &Stones) -> bool {
    match (self, rhs) {
      (Stones::A32(n1), Stones::A32(n2)) => n1 == n2,
      (Stones::A64(a1, a2), Stones::A64(b1, b2)) => a1 == b1 && a2 == b2,
      (Stones::A96(a1, a2, a3), Stones::A96(b1, b2, b3)) => a1 == b1 && a2 == b2  && a3 == b3,
      _ => panic!("stones type not matched"),
    }
  }
}

impl Display for Stones {
  fn fmt(&self, f: &mut Formatter) -> fmt::Result {
    let size = match self {
      Stones::A32(_) => 5,
      Stones::A64(_, _) => 7,
      Stones::A96(_, _, _) => 9,
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
