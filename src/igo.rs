pub mod board;
pub mod stones;

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
impl Turn {
  fn rev(&self) -> Turn {
    match self {
      Turn::Black => Turn::White,
      Turn::White => Turn::Black
    }
  }
}

#[derive(Copy, Clone)]
pub enum Stones {
  A32(u32),
  A64(u32, u32),
  A96(u32, u32, u32)
}

pub struct Board {
  size: BoardSize,
  turn: Turn,
  step: u32,
  black: Stones,
  white: Stones,
  history_black: [Stones; 3],
  history_white: [Stones; 3],
}
