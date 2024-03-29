use super::*;
use std::fmt;
use std::fmt::{Display, Formatter};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const HIST_SIZE: usize = 5;
const KIFU_TABLE: [char; 20] = [
  'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t'
];

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
      kifu: Vec::new()
    }
  }
  pub fn size(&self) -> u32 {
    self.size as u32
  }
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.black.hash(state);
    self.white.hash(state);
    self.history_black.hash(state);
    self.history_white.hash(state);
    self.pass_cnt.hash(state);
    self.turn.hash(state);
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
  pub fn to_vec(&self) -> Vec<f32> {
    let mut stones: Vec<f32> = Vec::new();
    stones.extend(self.black.vec());
    stones.extend(self.white.vec());
    for h in self.history_black {
      stones.extend(h.vec());
    }
    for h in self.history_white {
      stones.extend(h.vec());
    }
    stones
  }
  pub fn set_vec(&mut self, vec: Vec<f32>) {
    let s = self.size as usize;
    let max = s * s;
    if vec.len() < 100 {
      panic!("len {}", vec.len());
    }
    self.black.set_vec(vec[0..max].to_vec());
    self.white.set_vec(vec[max..max*2].to_vec());
    let hist_len = self.history_black.len();
    for i in 0..hist_len {
      self.history_black[i].set_vec(vec[max*(2+i)..max*(3+i)].to_vec());
      self.history_white[i].set_vec(vec[max*(2+i+hist_len)..max*(3+i+hist_len)].to_vec());
    }
  }
  fn history(&self, color: Turn) -> [Stones; HIST_SIZE] {
    match color {
      Turn::Black => self.history_black,
      Turn::White => self.history_white
    }
  }
  fn opp_history(&self, color: Turn) -> [Stones; HIST_SIZE] {
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
          Dir::Down  => Stones::new64(0b1111111_0000000_0000000_0000000_0000000_0000000_0000000),
          Dir::Up    => Stones::new64(0b0000000_0000000_0000000_0000000_0000000_0000000_1111111),
        }
      BoardSize::S9 =>
        match dir {
          Dir::Right => Stones::new128(0b100000000_100000000_100000000_100000000_100000000_100000000_100000000_100000000_100000000),
          Dir::Left  => Stones::new128(0b000000001_000000001_000000001_000000001_000000001_000000001_000000001_000000001_000000001),
          Dir::Down  => Stones::new128(0b111111111_000000000_000000000_000000000_000000000_000000000_000000000_000000000_000000000),
          Dir::Up    => Stones::new128(0b000000000_000000000_000000000_000000000_000000000_000000000_000000000_000000000_111111111),
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
  // 本来のルールの有効手
  pub fn vec_valid_moves(&self, color: Turn) -> Vec<bool> {
    let stones = self.valid_moves(color);
    let amax = self.action_size() - 1;
    let mut res: Vec<bool> = vec![false; self.action_size()];
    for i in 0..amax {
      let key = stones >> i as u128 & 1 == 1;
      res[i] = key;
    }
    // 常にパスできる
    res[amax] = true; // pass
    res
  }
  // 実際の有効手
  pub fn vec_valid_moves_for_search(&self, color: Turn) -> Vec<bool> {
    let stones = self.valid_moves_for_train(color, true);
    let amax = self.action_size() - 1;
    let mut res: Vec<bool> = vec![false; self.action_size()];
    for i in 0..amax {
      let key = stones >> i as u128 & 1 == 1;
      res[i] = key;
    }
    // 常にパスできる
    res[amax] = true; // pass
    res
  }
  // CPU探索用の有効手
  pub fn vec_valid_moves_for_cpu(&self, color: Turn) -> Vec<bool> {
    let stones = self.valid_moves_for_train(color, false);
    let amax = self.action_size() - 1;
    let mut res: Vec<bool> = vec![false; self.action_size()];
    for i in 0..amax {
      let key = stones >> i as u128 & 1 == 1;
      res[i] = key;
    }
    // 常にパスできる
    res[amax] = true; // pass
    res
  }
  // 学習で利用したい有効手
  pub fn vec_valid_moves_for_train(&self, color: Turn) -> Vec<bool> {
    let stones = self.valid_moves_for_train(color, true);
    let amax = self.action_size() - 1;
    let mut res: Vec<bool> = vec![false; self.action_size()];
    let mut true_cnt = 0;
    for i in 0..amax {
      let key = stones >> i as u128 & 1 == 1;
      res[i] = key;
      if key {
        true_cnt += 1;
      }
    }
    if true_cnt == 0 {
      // 手がもう無いときだけパスできる
      res[amax] = true; // pass
    }
    res
  }
  // 禁止はコウ
  pub fn valid_moves(&self, color: Turn) -> Stones {
    let s = self.size as usize;
    let bw = self.black | self.white;
    let st = self.stones(color);
    let op = self.opp_stones(color);
    let hist = self.history(color);
    let opp_hist = self.opp_history(color);
    let kuten = self.mask(!bw);
    // kuten where surrounded by same color stones
    let op_surround_kuten = kuten & (
      (op >> 1 | self.edge(Dir::Right)) &
      (op << 1 | self.edge(Dir::Left)) &
      (op >> s as u128 | self.edge(Dir::Down)) &
      (op << s as u128 | self.edge(Dir::Up)));

    // find kou
    let mut kou = Stones::new(self.size);
    let mut st_try = op_surround_kuten;
    while st_try != 0 {
      // try adding stone where surrounded by stones
      let rbit = st_try & ((!st_try) + 1);
      st_try = st_try ^ rbit; // remove right end bit
      // try st
      let new_try = st | rbit;
      let ds = self.death_stones(op, new_try); // kill oponent stones first
      let killed_op = op ^ ds;
      let ds = self.death_stones(new_try, killed_op); // kill self stone next
      let killed_st = new_try ^ ds;
      if hist.contains(&killed_st) && opp_hist.contains(&killed_op) {
        // ignore kou
        kou = kou | rbit;
      }
      if killed_st == st && killed_op == op {
        // １手前もコウ
        kou = kou | rbit;
      }
    }
    // exclude kou
    let valids = kuten ^ kou;
    valids
  }
  // 自殺手禁止、自分の目埋め禁止
  pub fn valid_moves_for_train(&self, color: Turn, ignore_meume: bool) -> Stones {
    let s = self.size as usize;
    let bw = self.black | self.white;
    let st = self.stones(color);
    let op = self.opp_stones(color);
    let hist = self.history(color);
    let opp_hist = self.opp_history(color);
    let kuten = self.mask(!bw);
    // kuten where surrounded by same color stones
    // let st_surround_kuten = kuten & (
    //   (st >> 1 | self.edge(Dir::Right)) &
    //   (st << 1 | self.edge(Dir::Left)) &
    //   (st >> s as u128 | self.edge(Dir::Down)) &
    //   (st << s as u128 | self.edge(Dir::Up)));
    // let op_surround_kuten = kuten & (
    //   (op >> 1 | self.edge(Dir::Right)) &
    //   (op << 1 | self.edge(Dir::Left)) &
    //   (op >> s as u128 | self.edge(Dir::Down)) &
    //   (op << s as u128 | self.edge(Dir::Up)));
    let surround_kuten = kuten & (
      (bw >> 1 | self.edge(Dir::Right)) &
      (bw << 1 | self.edge(Dir::Left)) &
      (bw >> s as u128 | self.edge(Dir::Down)) &
      (bw << s as u128 | self.edge(Dir::Up)));

    // ignore suicide
    let mut st_try = surround_kuten;
    let mut suicide = Stones::new(self.size);
    let mut kou = Stones::new(self.size);
    let mut dstone = Stones::new(self.size);
    while st_try != 0 {
      // try adding stone where surrounded by stones
      let rbit = st_try & ((!st_try) + 1);
      st_try = st_try ^ rbit; // remove right end bit
      // try st
      let new_try = st | rbit;
      // 相手の死に石
      let ds_op = self.death_stones(op, new_try);
      if hist.contains(&new_try) && opp_hist.contains(&(op ^ ds_op)) {
        // kou
        kou = kou | rbit;
        continue;
      }
      dstone = dstone | ds_op; // 相手の石が取れれば合法
      // 自分の死に石
      let ds_st = self.death_stones(new_try, op);
      // 自殺手禁止
      suicide = suicide | (ds_st & rbit);
    }
    // valid if any of the beside stone died
    let dbeside = surround_kuten & (
      (dstone >> 1 & !self.edge(Dir::Right)) |
      (dstone << 1 & !self.edge(Dir::Left)) |
      (dstone >> s as u128 & !self.edge(Dir::Down)) |
      (dstone << s as u128 & !self.edge(Dir::Up)));
    let valids;
    if ignore_meume {
      valids = (kuten & !surround_kuten) | dbeside;
    } else {
      valids = (kuten & !suicide & !kou) | dbeside;
    }
    valids
  }
  pub fn kill_point(&self, color: Turn) -> Stones {
    let s = self.size as usize;
    let bs = s * s;
    let st = self.stones(color);
    let op = self.opp_stones(color);
    let mut kpoint = Stones::new(self.size);
    for a in 0..bs {
      let new_try = st | 1 << a;
      let ds = self.death_stones(op, new_try);
      if ds != 0 {
        kpoint = kpoint | 1 << a;
      }
    }
    kpoint
  }
  // コウが何手前か返す
  pub fn kou_cnt(&self, mov: u32, color: Turn) -> i32 {
    let s = self.size as usize;
    let bw = self.black | self.white;
    let st = self.stones(color);
    let op = self.opp_stones(color);
    let hist = self.history(color);
    let opp_hist = self.opp_history(color);
    // 既に石のある所はコウではない
    if (st | op) & (1 << mov) != 0 {
      return -1;
    }
    // １手打ってみて試す
    let new_try = st | (1 << mov);
    let ds = self.death_stones(op, new_try); // kill oponent stones first
    let killed_op = op ^ ds;
    let ds = self.death_stones(new_try, killed_op); // kill self stone next
    let killed_st = new_try ^ ds;
    if killed_st == st && killed_op == op {
      // 現在と形が変わらないのもコウ
      return 0;
    }
    for i in 0..HIST_SIZE {
      if hist[i] == killed_st && opp_hist[i] == killed_op {
        return i as i32 + 1;
      }
    }
    return -1;
  }

  pub fn count_diff(&self) -> i32 {
    let b: i32 = self.black.count_ones() as i32;
    let w: i32 = self.white.count_ones() as i32;
    b - w
  }
  pub fn game_ended(&self, auto_resign: bool, komi: i32) -> i8 {
    let s = self.size().pow(2);
    let diff = self.count_diff();
    if self.pass_cnt >= 2 || (auto_resign && diff.abs() >= s as i32) || self.step >= s * 3 + 5{
      if diff > komi { return 1; }
      return -1;
    }
    return 0;
  }
  pub fn get_kifu_sgf(&self) -> String {
    let mut kifu = "(;GM[1]SZ[".to_string() + &(self.size as i32).to_string() + &"];".to_string();
    let s = self.size as usize;
    for &k in &self.kifu {
      let mut turn = 0;
      if k > 0 { turn = 1; }
      let n = k.abs() as usize - 1;
      let mut x = n % s;
      let mut y = n / s;
      if y == s { // pass
        x = 19;
        y = 19;
      }
      let code = format!("{}[{}{}]", ['W', 'B'][turn], KIFU_TABLE[x], KIFU_TABLE[y]);
      kifu.push_str(&code);
      kifu.push_str(";");
    }
    kifu.push_str(")");
    kifu
  }
  pub fn action_xy(&mut self, x: u32, y: u32, turn: Turn) {
    let mov = (x - 1) + (y - 1) * self.size as u32;
    self.action(mov, turn);
  }
  pub fn action(&mut self, mov: u32, turn: Turn) {
    self.step += 1;
    self.kifu.push((mov + 1) as i32 * self.turn as i32);
    for i in (1..HIST_SIZE).rev() {
      self.history_black[i] = self.history_black[i-1];
      self.history_white[i] = self.history_white[i-1];
    }
    self.history_black[0] = self.black;
    self.history_white[0] = self.white;

    if mov == self.action_size() as u32 - 1 { // pass
      self.turn = turn.rev();
      self.pass_cnt += 1;
      return;
    }
    self.pass_cnt = 0;

    let stones = self.stones(turn);
    // hit stone
    let addStone = Stones::new(self.size);
    self.set_stones(turn, stones | 1 << mov);
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
      (bw >> s as u128 | self.edge(Dir::Down)) &
      (bw << s as u128 | self.edge(Dir::Up)));
    let mut sc = ps & sc;
    // alive when any aliveStone beside
    let mut death = bw;
    while death != sc {
      death = sc;
      let alive = sc ^ ps;
      sc = sc & !(
        sc & (alive >> 1 & !self.edge(Dir::Right)) |
        sc & (alive << 1 & !self.edge(Dir::Left)) |
        sc & (alive >> s as u128 & !self.edge(Dir::Down)) |
        sc & (alive << s as u128 & !self.edge(Dir::Up))
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
  pub fn input(&self) -> Vec<f32> {
    let mut color = self.turn as i32;
    if color < 0 { color = 0; }
    let s = self.size() as usize;
    let bs = s.pow(2);
    let mut vec: Vec<f32> = Vec::new();
    vec.append(&mut self.black.vec());
    vec.append(&mut self.white.vec());
    for i in 0..3 {
      vec.append(&mut self.history_black[i].vec());
      vec.append(&mut self.history_white[i].vec());
    }
    vec.append(&mut vec![color as f32; bs]);
    let valids: Vec<f32> = self.vec_valid_moves(self.turn).iter().map(|&v| v as i32 as f32).collect();
    vec.append(&mut valids[0..bs].to_vec());
    vec.append(&mut self.kill_point(self.turn).vec());
    vec.append(&mut vec![self.pass_cnt as f32; bs]);
    vec
  }
  pub fn canonical_form(&self, turn: Turn) -> Board {
    let mut board = self.clone();
    if turn == Turn::White { board.rev(); }
    board
  }
  pub fn flip_diag(&mut self) -> &mut Board {
    self.black = self.black.flip_diag();
    self.white = self.white.flip_diag();
    for i in 0..3 {
      self.history_black[i] = self.history_black[i].flip_diag();
      self.history_white[i] = self.history_white[i].flip_diag();
    }
    self
  }
  pub fn flip_vert(&mut self) -> &mut Board {
    self.black = self.black.flip_vert();
    self.white = self.white.flip_vert();
    for i in 0..3 {
      self.history_black[i] = self.history_black[i].flip_vert();
      self.history_white[i] = self.history_white[i].flip_vert();
    }
    self
  }
  pub fn flip_diag_pi(&self, pi: &Vec<f32>) -> Vec<f32> {
    let s = self.size as usize;
    let mut idxs = Vec::new();
    for i in 0..s {
      let mut row: Vec<usize> = (0..s*s-i).rev().step_by(s).collect();
      idxs.append(&mut row);
    }
    idxs.iter().map(|i| pi[*i]).collect()
  }
  pub fn flip_vert_pi(&self, pi: &Vec<f32>) -> Vec<f32> {
    let s = self.size as usize;
    let mut res = Vec::new();
    for chunk in pi.chunks(s).rev() {
      res.extend_from_slice(chunk);
    }
    res
  }
  pub fn symmetries(&self, pi: Vec<f32>) -> Vec<(Vec<f32>, Vec<f32>)> {
    let mut f1 = self.clone();
    let mut f2 = self.clone();
    let fd = f1.flip_diag().input();
    let fv = f2.flip_vert().input();
    let fdv = f1.flip_vert().input();
    let fvd = f2.flip_diag().input();
    let fdvd = f1.flip_diag().input();
    let fvdv = f2.flip_vert().input();
    let fvdvd = f2.flip_diag().input();

    assert!(pi.len() == self.action_size());
    let mut p = pi.clone();
    let pass = match p.pop() {
      Some(x) => x,
      None => 0.0
    };
    let mut pfd = self.flip_diag_pi(&p);
    let mut pfv = self.flip_vert_pi(&p);
    let mut pfdv = self.flip_vert_pi(&pfd);
    let mut pfvd = self.flip_diag_pi(&pfv);
    let mut pfdvd = self.flip_diag_pi(&pfdv);
    let mut pfvdv = self.flip_vert_pi(&pfvd);
    let mut pfvdvd = self.flip_diag_pi(&pfvdv);
    p.push(pass);
    pfd.push(pass);
    pfv.push(pass);
    pfdv.push(pass);
    pfvd.push(pass);
    pfdvd.push(pass);
    pfvdv.push(pass);
    pfvdvd.push(pass);
    
    vec![
      (self.input(), pi),
      (fd, pfd),
      (fv, pfv),
      (fdv, pfdv),
      (fvd, pfvd),
      (fdvd, pfdvd),
      (fvdv, pfvdv),
      (fvdvd, pfvdvd),
    ]
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
        let b = self.black >> a as u128 & 1;
        let w = self.white >> a as u128 & 1;
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
