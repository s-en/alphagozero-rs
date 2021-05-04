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
  pub size: BoardSize,
  pub turn: Turn,
  pub pass_cnt: u32,
  pub step: u32,
  pub black: Stones,
  pub white: Stones,
  pub history_black: [Stones; 5],
  pub history_white: [Stones; 5],
}

pub struct MCTS {
  sim_num: u32,
  cpuct: f32,
  qsa: HashMap<(u64, usize), f32>, // Q values
  nsa: HashMap<(u64, usize), u32>, // edge visited times
  ns: HashMap<u64, u32>, // board visited times
  ps: HashMap<u64, Vec<f32>>, // initial policy (returned by neural net)
  es: HashMap<u64, i8>, // game ended
  vs: HashMap<u64, Vec<bool>>, // valid moves
  search_cnt: u32,
  cache_hit_cnt: u32,
}
