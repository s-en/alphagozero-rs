use super::*;
use std::fmt;
use std::fmt::{Display, Formatter};
use std::ops::{Not};
use std::cmp::PartialEq;
use auto_ops::impl_op;
use auto_ops;


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
  pub fn size(&self) -> usize {
    match self {
      Stones::A32(_) => 5,
      Stones::A64(_) => 7,
      Stones::A128(_) => 9,
    }
  }
  pub fn count_ones(&self) -> u32 {
    match self {
      Stones::A32(val) => val.count_ones(),
      Stones::A64(val) => val.count_ones(),
      Stones::A128(val) => val.count_ones(),
    }
  }
  pub fn flip_diag(&self) -> Stones {
    match self {
      Stones::A32(val) => {
        let mut x: u32 = *val;
        let mut t: u32;
        const K1: u32 = 0b11000_11000_00000_00000_00000;
        const K2: u32 = 0b00100_00000_10000_00000_00000;
        const K3: u32 = 0b10010_00100_01000_10010_00000;
        t  = K1 & (x ^ (x << 18));
        x ^=       t ^ (t >> 18) ;
        t  = K2 & (x ^ (x << 12));
        x ^=       t ^ (t >> 12) ;
        t  = K3 & (x ^ (x <<  6));
        x ^=       t ^ (t >>  6) ;
        Stones::new32(x)
      },
      Stones::A64(val) => {
        let mut x: u64 = *val;
        let mut t: u64;
        const K4: u64 = 0b1110000_1110000_1110000_0000000_0000000_0000000_0000000;
        const K2: u64 = 0b1001100_0001100_0000000_1100000_1100100_0000000_0000000;
        const K1: u64 = 0b0101010_1000000_0001010_1010000_0000010_1010100_0000000;
        t  = K4 & (x ^ (x << 32));
        x ^=       t ^ (t >> 32) ;
        t  = K2 & (x ^ (x << 16));
        x ^=       t ^ (t >> 16) ;
        t  = K1 & (x ^ (x << 8));
        x ^=       t ^ (t >> 8) ;
        Stones::new64(x)
      },
      Stones::A128(val) => {
        let mut x: u128 = *val;
        let mut t: u128;
        const K5: u128 = 0b111100000_111100000_111100000_111100000_000000000_000000000_000000000_000000000_000000000;
        const K4: u128 = 0b000010000_000000000_000000000_000000000_100000000_000000000_000000000_000000000_000000000;
        const K3: u128 = 0b000000000_000010000_000000000_000000000_010000000_000000000_000000000_000000000_000000000;
        const K2: u128 = 0b110001100_110001100_000010000_000000000_001000000_110001100_110001100_000000000_000000000;
        const K1: u128 = 0b101001010_000000000_101001010_000010000_000100000_101001010_000000000_101001010_000000000;
        t  = K5 & (x ^ (x << 50));
        x ^=       t ^ (t >> 50) ;
        t  = K4 & (x ^ (x << 40));
        x ^=       t ^ (t >> 40) ;
        t  = K3 & (x ^ (x << 30));
        x ^=       t ^ (t >> 30) ;
        t  = K2 & (x ^ (x << 20));
        x ^=       t ^ (t >> 20) ;
        t  = K1 & (x ^ (x << 10));
        x ^=       t ^ (t >> 10) ;
        Stones::new128(x)
      }
    }
  }
  pub fn flip_vert(&self) -> Stones {
    match self {
      Stones::A32(val) => {
        let mut x: u32 = *val;
        let mut t: u32;
        const K1: u32 = 0b11111_11111_00000_00000_00000;
        const K2: u32 = 0b11111_00000_00000_11111_00000;
        t  = K1 & (x ^ (x << 15));
        x ^=       t ^ (t >> 15) ;
        t  = K2 & (x ^ (x << 5));
        x ^=       t ^ (t >> 5) ;
        Stones::new32(x)
      },
      Stones::A64(val) => {
        let mut x: u64 = *val;
        let mut t: u64;
        const K1: u64 = 0b1111111_1111111_1111111_0000000_0000000_0000000_0000000;
        const K2: u64 = 0b1111111_0000000_0000000_0000000_1111111_0000000_0000000;
        t  = K1 & (x ^ (x << 28));
        x ^=       t ^ (t >> 28) ;
        t  = K2 & (x ^ (x << 14));
        x ^=       t ^ (t >> 14) ;
        Stones::new64(x)
      },
      Stones::A128(val) => {
        let mut x: u128 = *val;
        let mut t: u128;
        const K1: u128 = 0b111111111_111111111_111111111_111111111_000000000_000000000_000000000_000000000_000000000;
        const K2: u128 = 0b111111111_111111111_000000000_000000000_000000000_111111111_111111111_000000000_000000000;
        const K3: u128 = 0b111111111_000000000_111111111_000000000_000000000_111111111_000000000_111111111_000000000;
        t  = K1 & (x ^ (x << 45));
        x ^=       t ^ (t >> 45) ;
        t  = K2 & (x ^ (x << 18));
        x ^=       t ^ (t >> 18) ;
        t  = K3 & (x ^ (x << 9));
        x ^=       t ^ (t >> 9) ;
        Stones::new128(x)
      }
    }
  }
  pub fn vec(&self) -> Vec<f32> {
    let s = self.size();
    let smax = s * s;
    let mut vec: Vec<f32> = vec![0.0; smax];
    let mut cnt = 0;
    for i in 0..smax {
      if *self >> i & 1 == 1 {
        vec[cnt] = 1.0;
      } else {
        vec[cnt] = 0.0;
      }
      cnt += 1;
    }
    vec
  }
  pub fn set_vec(&mut self, svec: Vec<f32>) {
    let s = self.size();
    let smax = s * s;
    *self = *self & 0;
    for i in 0..smax {
      if let Some(&action) = svec.get(i) {
        if action > 0.0 {
          *self = *self | 1 << i;
        }
      }
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
