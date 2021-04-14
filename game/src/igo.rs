pub mod board;
pub mod stones;
pub mod mcts;
use std::collections::HashMap;

#[derive(Copy, Clone, Hash, Eq, PartialEq)]
pub enum BoardSize {
  S5 = 5,
  S7 = 7,
  S9 = 9,
}

#[derive(Copy, Clone, Hash, Eq, PartialEq)]
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

#[derive(Copy, Clone, Hash, Eq)]
pub enum Stones {
  A32(u32),
  A64(u64),
  A128(u128)
}

#[derive(Hash, Eq, PartialEq, Clone)]
pub struct Board {
  size: BoardSize,
  turn: Turn,
  pass_cnt: u32,
  step: u32,
  black: Stones,
  white: Stones,
  history_black: [Stones; 3],
  history_white: [Stones; 3],
}

pub struct MCTS {
  sim_num: u32,
  cpuct: f32,
  board: Board,
  qsa: HashMap<(u64, usize), f32>, // Q values
  nsa: HashMap<(u64, usize), u32>, // edge visited times
  ns: HashMap<u64, u32>, // board visited times
  ps: HashMap<u64, Vec<f32>>, // initial policy (returned by neural net)
  es: HashMap<u64, i8>, // game ended
  vs: HashMap<u64, Vec<bool>>, // valid moves
  search_cnt: u32,
  cache_hit_cnt: u32,
}
