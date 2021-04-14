use super::*;
use std::fmt;
use std::fmt::{Display, Formatter};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const HIST_SIZE: usize = 3;

pub enum Dir {
  Left,
  Right,
  Down,
  Up,
}

impl Board {
  pub fn new(size: BoardSize) -> Board {
    Board {
      size,
      turn: Turn::Black,
      step: 0,
      pass_cnt: 0,
      black: Stones::new(size),
      white: Stones::new(size),
      history_black: [Stones::new(size); HIST_SIZE],
      history_white: [Stones::new(size); HIST_SIZE],
    }
  }
  pub fn calc_hash(&self) -> u64 {
    let mut s = DefaultHasher::new();
    self.hash(&mut s);
    s.finish()
  }
  pub fn calc_hash_action(&self, action: usize) -> u64 {
    let mut s = DefaultHasher::new();
    self.hash(&mut s);
    s.write_u32(action as u32);
    s.finish()
  }
  pub fn action_size(&self) -> usize {
    let s = self.size as usize;
    s * s + 1
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
  fn history(&self, color: Turn) -> [Stones; 3] {
    match color {
      Turn::Black => self.history_black,
      Turn::White => self.history_white
    }
  }
  fn opp_history(&self, color: Turn) -> [Stones; 3] {
    match color {
      Turn::Black => self.history_white,
      Turn::White => self.history_black
    }
  }
  pub fn set_stones(&mut self, color: Turn, val: Stones) {
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
          Dir::Left  => Stones::new32(0b00001_00001_00001_00001_00001),
          Dir::Down  => Stones::new32(0b11111_00000_00000_00000_00000),
          Dir::Up    => Stones::new32(0b00000_00000_00000_00000_11111),
        },
      BoardSize::S7 =>
        match dir {
          Dir::Right => Stones::new64(0b1000000_1000000_1000000_1000000_1000000_1000000_1000000),
          Dir::Left  => Stones::new64(0b0000001_0000001_0000001_0000001_0000001_0000001_0000001),
          Dir::Down  => Stones::new64(0b1111111_1111111_1111111_1111111_1111111_1111111_0000000),
          Dir::Up    => Stones::new64(0b0000000_1111111_1111111_1111111_1111111_1111111_1111111),
        }
      BoardSize::S9 =>
        match dir {
          Dir::Right => Stones::new128(0b100000000_100000000_100000000_100000000_100000000_100000000_100000000_100000000_100000000),
          Dir::Left  => Stones::new128(0b000000001_000000001_000000001_000000001_000000001_000000001_000000001_000000001_000000001),
          Dir::Down  => Stones::new128(0b111111111_111111111_111111111_111111111_111111111_111111111_111111111_111111111_000000000),
          Dir::Up    => Stones::new128(0b000000000_111111111_111111111_111111111_111111111_111111111_111111111_111111111_111111111),
        }
    }
  }
  fn mask(&self, stone: Stones) -> Stones {
    match self.size {
      BoardSize::S5 => stone & 0b11111_11111_11111_11111_11111,
      BoardSize::S7 => stone & Stones::new64(0b1111111_1111111_1111111_1111111_1111111_1111111_1111111),
      BoardSize::S9 => stone & Stones::new128(0b111111111_111111111_111111111_111111111_111111111_111111111_111111111_111111111_111111111),
    }
  }
  pub fn vec_valid_moves(&self, color: Turn) -> Vec<bool> {
    let stones = self.valid_moves(color);
    let amax = self.action_size() - 1;
    let mut res: Vec<bool> = Vec::new();
    for i in 0..amax {
      if stones >> i & 1 == 1 { res[i] = true; }
      else { res[i] = false; }
    }
    res[amax] = true; // pass
    res
  }
  pub fn valid_moves(&self, color: Turn) -> Stones {
    let s = self.size as usize;
    let bw = self.black | self.white;
    let st = self.stones(color);
    let op = self.opp_stones(color);
    let hist = self.history(color);
    let opp_hist = self.opp_history(color);
    let kuten = self.mask(!bw);
    let surround_kuten = kuten & (
      (op >> 1 | self.edge(Dir::Right)) &
      (op << 1 | self.edge(Dir::Left)) &
      (op >> s | self.edge(Dir::Down)) &
      (op << s | self.edge(Dir::Up)));
    // try adding surrounded stones
    let mut st_try = surround_kuten;
    let mut dstone = Stones::new(self.size);
    while st_try != 0 {
      let rbit = st_try & ((!st_try) + 1);
      st_try = st_try ^ rbit; // remove most right bit
      let new_try = st | rbit;
      let ds = self.death_stones(op, new_try);
      if hist.contains(&new_try) && opp_hist.contains(&(op ^ ds)) {
        // kou
        continue;
      }
      dstone = dstone | ds;
    }
    // valid if any of the beside stone died
    let dbeside = surround_kuten & (
      (dstone >> 1 & !self.edge(Dir::Right)) |
      (dstone << 1 & !self.edge(Dir::Left)) |
      (dstone >> s & !self.edge(Dir::Down)) |
      (dstone << s & !self.edge(Dir::Up)));
    let valids = (kuten & !surround_kuten) | dbeside;
    valids
  }
  pub fn count_diff(&self) -> i32 {
    let b: i32 = self.black.count_ones() as i32;
    let w: i32 = self.white.count_ones() as i32;
    b - w
  }
  pub fn game_ended(&self) -> i8 {
    if self.pass_cnt < 2 { return 0; }
    if self.count_diff() > 0 { return 1; }
    return -1;
  }
  pub fn action_xy(&mut self, x: u32, y: u32, turn: Turn) {
    let mov = x + (y - 1) * self.size as u32;
    self.action(mov, turn);
  }
  pub fn action(&mut self, mov: u32, turn: Turn) {
    self.step += 1;
    if mov == self.size as u32 { // pass
      self.turn = turn.rev();
      self.pass_cnt += 1;
      return;
    }
    self.pass_cnt = 0;
    for i in (1..3).rev() {
      self.history_black[i] = self.history_black[i-1];
      self.history_white[i] = self.history_white[i-1];
    }
    self.history_black[0] = self.black;
    self.history_white[0] = self.white;
    let stones = self.stones(turn);
    // hit stone
    self.set_stones(turn, stones | 1 << (mov - 1));
    // remove opp color
    self.remove_death_stones(turn.rev());
    // remove my color
    self.remove_death_stones(turn);
    self.turn = turn.rev();
  }
  pub fn remove_death_stones(&mut self, color: Turn) {
    let stones = self.stones(color);
    let dstones = self.death_stones(stones, self.opp_stones(color));
    self.set_stones(color, stones & !dstones);
  }
  pub fn death_stones(&self, stone: Stones, opp_stone: Stones) -> Stones {
    let s = self.size as usize;
    let bw = stone | opp_stone;
    let ps = stone;
    let sc = bw & (
      (bw >> 1 | self.edge(Dir::Right)) &
      (bw << 1 | self.edge(Dir::Left)) &
      (bw >> s | self.edge(Dir::Down)) &
      (bw << s | self.edge(Dir::Up)));
    let mut sc = ps & sc;
    // alive when any aliveStone beside
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
    }
    death
  }
  pub fn rev(&mut self) {
    self.turn = self.turn.rev();
    self.black = self.black ^ self.white;
    self.white = self.black ^ self.white;
    self.black = self.black ^ self.white;
    for i in 0..HIST_SIZE {
      self.history_black[i] = self.history_black[i] ^ self.history_white[i];
      self.history_white[i] = self.history_black[i] ^ self.history_white[i];
      self.history_black[i] = self.history_black[i] ^ self.history_white[i];
    }
  }
  pub fn next_state(&mut self, mov: u32) -> Board {
    self.action(mov, Turn::Black);
    let mut new_board = self.clone();
    new_board.rev();
    new_board
  }
  // input style for nnet
  pub fn input(&self, rot_t: i64, flip: bool) -> Vec<f32> {
    let mut vec: Vec<f32> = self.black.vec();
    vec.append(&mut self.white.vec());
    vec
  }
  pub fn canonical_form(&self, turn: Turn) -> Board {
    let mut board = self.clone();
    if turn == Turn::White { board.rev(); }
    board
  }
  // pub fn symmetries(&self, pi: Vec<f32>) -> (Vec<f32>, Vec<f32>) {
  //   let n = self.size as i64;
  //   let pi_board = Tensor::of_slice(&pi).reshape(&[n, n]);
  //   let b: Vec<Tensor> = vec![];
  //   let p: Vec<Tensor> = vec![];

  //   for i in 1..5 {
  //     for j in &[true, false] {
  //       let newB = board.rot90(i, &[0, 1]);
  //       let newPi = pi_board.rot90(i, &[0, 1]);
  //       if j:
  //           newB = np.fliplr(newB)
  //           newPi = np.fliplr(newPi)
  //       l += [(newB, list(newPi.ravel()) + [pi[-1]])]
  //     }
  //   }
  //   (b, p)
  // }
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
