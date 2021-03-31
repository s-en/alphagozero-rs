use std::fmt;
use std::fmt::{Display, Formatter};
use std::ops::{BitAnd, BitOr, BitXor, Not, Shr, Shl};
use std::cmp::PartialEq;

pub fn test() -> String {
  "Hello!".to_string()
}

#[derive(Copy, Clone)]
pub enum BoardSize {
  S5 = 5,
  S7 = 7
}

#[derive(Copy, Clone)]
pub enum Turn {
  Black = 1,
  White = -1
}

pub enum Dir {
  Left,
  Right,
  Down,
  Up,
}

#[derive(Copy, Clone)]
enum Stones {
  A32(u32),
  A64(u64),
  A128(u128)
}
impl Shr<usize> for Stones {
  type Output = Stones;
  fn shr(self, rhs: usize) -> Stones {
    match self {
      Stones::A32(n) => Stones::A32(n >> rhs),
      Stones::A64(n) => Stones::A64(n >> rhs),
      Stones::A128(n) => Stones::A128(n >> rhs),
    }
  }
}
impl Shl<usize> for Stones {
  type Output = Stones;
  fn shl(self, rhs: usize) -> Stones {
    match self {
      Stones::A32(n) => Stones::A32(n << rhs),
      Stones::A64(n) => Stones::A64(n << rhs),
      Stones::A128(n) => Stones::A128(n << rhs),
    }
  }
}
impl BitAnd for Stones {
  type Output = Stones;
  fn bitand(self, rhs: Stones) -> Stones {
    match (self, rhs) {
      (Stones::A32(n1), Stones::A32(n2)) => Stones::A32(n1 & n2),
      (Stones::A64(n1), Stones::A64(n2)) => Stones::A64(n1 & n2),
      (Stones::A128(n1), Stones::A128(n2)) => Stones::A128(n1 & n2),
      _ => panic!("stones type not matched"),
    }
  }
}
impl BitAnd<usize> for Stones {
  type Output = Stones;
  fn bitand(self, rhs: usize) -> Stones {
    match self {
      Stones::A32(n1) => Stones::A32(n1 & rhs as u32),
      Stones::A64(n1) => Stones::A64(n1 & rhs as u64),
      Stones::A128(n1) => Stones::A128(n1 & rhs as u128),
    }
  }
}
impl BitOr for Stones {
  type Output = Stones;
  fn bitor(self, rhs: Stones) -> Stones {
    match (self, rhs) {
      (Stones::A32(n1), Stones::A32(n2)) => Stones::A32(n1 | n2),
      (Stones::A64(n1), Stones::A64(n2)) => Stones::A64(n1 | n2),
      (Stones::A128(n1), Stones::A128(n2)) => Stones::A128(n1 | n2),
      _ => panic!("stones type not matched"),
    }
  }
}
impl BitOr<usize> for Stones {
  type Output = Stones;
  fn bitor(self, rhs: usize) -> Stones {
    match self {
      Stones::A32(n1) => Stones::A32(n1 | rhs as u32),
      Stones::A64(n1) => Stones::A64(n1 | rhs as u64),
      Stones::A128(n1) => Stones::A128(n1 | rhs as u128),
    }
  }
}
impl BitXor for Stones {
  type Output = Stones;
  fn bitxor(self, rhs: Stones) -> Stones {
    match (self, rhs) {
      (Stones::A32(n1), Stones::A32(n2)) => Stones::A32(n1 ^ n2),
      (Stones::A64(n1), Stones::A64(n2)) => Stones::A64(n1 ^ n2),
      (Stones::A128(n1), Stones::A128(n2)) => Stones::A128(n1 ^ n2),
      _ => panic!("stones type not matched"),
    }
  }
}
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
impl PartialEq<i32> for Stones {
  fn eq(&self, rhs: &i32) -> bool {
    let val: &u32 = &(*rhs as u32);
    match self {
      Stones::A32(n1) => n1 == val,
      _ => panic!("stones type not matched"),
    }
  }
}
impl PartialEq<u32> for Stones {
  fn eq(&self, rhs: &u32) -> bool {
    match self {
      Stones::A32(n1) => n1 == rhs,
      _ => panic!("stones type not matched"),
    }
  }
}
impl PartialEq<u64> for Stones {
  fn eq(&self, rhs: &u64) -> bool {
    match self {
      Stones::A64(n1) => n1 == rhs,
      _ => panic!("stones type not matched"),
    }
  }
}
impl PartialEq<u128> for Stones {
  fn eq(&self, rhs: &u128) -> bool {
    match self {
      Stones::A128(n1) => n1 == rhs,
      _ => panic!("stones type not matched"),
    }
  }
}
impl Stones {
  fn new(size: BoardSize) -> Stones {
    match size {
      BoardSize::S5 => Stones::A32(0),
      BoardSize::S7 => Stones::A64(0),
    }
  }
  fn new32(val: u32) -> Stones {
    Stones::A32(val)
  }
  fn new64(val: u64) -> Stones {
    Stones::A64(val)
  }
  fn new128(val: u128) -> Stones {
    Stones::A128(val)
  }
}

pub struct Board {
  size: BoardSize,
  turn: Turn,
  step: u32,
  black: Stones,
  white: Stones
}

impl Board {
  pub fn new(size: BoardSize) -> Board {
    Board {
      size,
      turn: Turn::Black,
      step: 0,
      black: Stones::new(size),
      white: Stones::new(size),
    }
  }
  fn stones(&self, color: Turn) -> Stones {
    match color {
      Turn::Black => self.black,
      Turn::White => self.white
    }
  }
  fn opp_stones(&self, color: Turn) -> Stones {
    match color {
      Turn::Black => self.white,
      Turn::White => self.black
    }
  }
  fn set_stones(&mut self, color: Turn, val: Stones) {
    match color {
      Turn::Black => self.black = val,
      Turn::White => self.white = val
    };
  }
  fn edge(&self, dir: Dir) -> Stones {
    match self.size {
      BoardSize::S5 => 
        match dir {
          Dir::Right => Stones::new32(0b10000_10000_10000_10000_10000),
          Dir::Left => Stones::new32(0b00001_00001_00001_00001_00001),
          Dir::Down => Stones::new32(0b11111_00000_00000_00000_00000),
          Dir::Up => Stones::new32(0b00000_00000_00000_00000_11111),
        },
      BoardSize::S7 =>
        match dir {
          Dir::Right => Stones::new64(0b1000000_1000000_1000000_1000000_1000000_1000000_1000000),
          Dir::Left => Stones::new64(0b0000001_0000001_0000001_0000001_0000001_0000001_0000001),
          Dir::Down => Stones::new64(0b1111111_1111111_1111111_1111111_1111111_1111111_0000000),
          Dir::Up => Stones::new64(0b0000000_1111111_1111111_1111111_1111111_1111111_1111111),
        }
    }
  }
  pub fn remove_death_stones(&mut self, color: Turn) {
    let s = self.size as usize;
    let bw = self.black | self.white;
    let ps = self.stones(color);
    let sc = bw & (
      (bw >> 1 | self.edge(Dir::Right)) &
      (bw << 1 | self.edge(Dir::Left)) &
      (bw >> s | self.edge(Dir::Down)) &
      (bw << s | self.edge(Dir::Up)));
    let mut sc = ps & sc;
    println!("{}", sc);
    // 周りに１つでも生石がいれば生き
    let mut death = bw;
    while death != sc {
      death = sc;
      let alive = sc ^ ps;
      sc = sc & !(
        sc & (alive >> 1 & !self.edge(Dir::Right)) |
        sc & (alive << 1 & !self.edge(Dir::Left)) |
        sc & (alive >> s & !self.edge(Dir::Down)) |
        sc & (alive << s & !self.edge(Dir::Up))
      );
      println!("{}", sc);
    }
    println!("{}", death);
    self.set_stones(color, ps & !death);
  }
  pub fn action_xy(&mut self, x: u32, y: u32, turn: Turn) {
    let mov = x + (y - 1) * self.size as u32;
    self.action(mov, turn);
  }
  pub fn action(&mut self, mov: u32, turn: Turn) {
    let stones = self.stones(turn);
    self.set_stones(turn, stones | 1 << (mov - 1));
  }
}

impl Display for Board {
  fn fmt(&self, f: &mut Formatter) -> fmt::Result {
    let size = self.size as usize;
    let mut bstr = "".to_string();
    for y in 0..size+1 {
      for x in 0..size+1 {
        if y == 0 {
          bstr.push_str(&x.to_string());
          continue;
        }
        if x == 0 {
          bstr.push_str(&y.to_string());
          continue;
        }
        let a = x - 1 + (y - 1)*size;
        let b = self.black >> a & 1;
        let w = self.white >> a & 1;
        if b == 1 {
          bstr.push_str("o");
        } else if w == 1 {
          bstr.push_str("x");
        } else {
          bstr.push_str("-");
        }
      }
      bstr.push('\n');
    }
    write!(f, "{}", bstr)
  }
}

impl Display for Stones {
  fn fmt(&self, f: &mut Formatter) -> fmt::Result {
    let size = 5;
    let mut bstr = "".to_string();
    for y in 0..size {
      for x in 0..size {
        let a = x + y * 5;
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

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn remove_death_stones_5x5() {
      assert_eq!(2 + 2, 4);
      let mut board = Board::new(BoardSize::S5);
      let b = Turn::Black;
      let w = Turn::White;
      board.action_xy(3, 3, b);
      board.action_xy(3, 4, b);
      board.action_xy(4, 4, b);
      board.action_xy(2, 3, w);
      board.action_xy(4, 3, w);
      board.action_xy(3, 2, w);
      board.action_xy(2, 4, w);
      board.action_xy(5, 4, w);
      board.action_xy(3, 5, w);
      board.action_xy(4, 5, w);

      board.action_xy(1, 1, b);
      board.action_xy(2, 1, w);
      board.action_xy(1, 2, w);

      board.action_xy(1, 3, b);
      board.action_xy(1, 4, w);

      board.action_xy(5, 5, b);
      board.action_xy(5, 1, b);
      board.remove_death_stones(b);
      println!("{}", board);
      assert_eq!(
        format!("{}", board),
        "012345\n\
        1-x--o\n\
        2x-x--\n\
        3-x-x-\n\
        4xx--x\n\
        5--xx-\n"
      );
  }
}