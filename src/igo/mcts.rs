use super::*;
use std::cmp::Ordering;

fn max_idx(vals: &Vec<f32>) -> usize {
  let index_of_max: Option<usize> = vals
    .iter()
    .enumerate()
    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
    .map(|(index, _)| index);
    match index_of_max {
      None => 0,
      Some(n) => n,
    }
}

impl MCTS {
  pub fn new(size: BoardSize) -> MCTS {
    MCTS {
      sim_num: 10,
      cpuct: 1.0,
      board: Board::new(size),
      qsa: HashMap::new(), // Q values
      nsa: HashMap::new(), // edge visited times
      ns: HashMap::new(), // board visited times
      ps: HashMap::new(), // initial policy (returned by neural net)
      es: HashMap::new(), // game ended
      vs: HashMap::new(), // valid moves
      search_cnt: 0,
      cache_hit_cnt: 0,
      nnet: NNet::new(),
    }
  }
  pub fn get_action_prob(&mut self, c_board: &Board, temp: f32) -> Vec<f32> {
    for _ in 0..self.sim_num {
      let mut b = c_board.clone();
      self.search(&mut b);
    }
    let amax = c_board.action_size();
    let mut counts = Vec::new();
    let s = c_board.calc_hash();
    for a in 0..amax {
      let mut val = 0;
      if self.nsa.contains_key(&(s, a)) {
        val = self.nsa[&(s, a)];
      }
      counts.push(val);
    }
    if temp == 0.0 {
      let mut probs = vec![0.0; amax];
      let best_idx = max_idx(&probs);
      probs[best_idx] = 1.0;
      return probs;
    }
    let counts: Vec<f32> = counts.iter().map(|x| (*x as f32).powf(1.0 / temp)).collect();
    let counts_sum: f32 = counts.iter().sum();
    let probs: Vec<f32> = counts.iter().map(|x| x / counts_sum).collect();
    probs
  }
  pub fn search(&mut self, c_board: &mut Board) -> i8 {
    let s = c_board.calc_hash();
    if !self.es.contains_key(&s) {
      self.es.insert(s, c_board.game_ended());
    }
    if self.es[&s] != 0 {
      // terminal node
      return -self.es[&s];
    }
    self.search_cnt += 1;
    if !self.ps.contains_key(&s) {
      // leaf node
      let (ps, v) = self.nnet.predict();
      let valids = c_board.vec_valid_moves(Turn::Black);
      let mut masked_valids: Vec<f32> = valids.iter().enumerate().map(|(i, x)| *x as i32 as f32 * ps[i]).collect();
      let sum_ps_s: f32 = masked_valids.iter().sum();
      if sum_ps_s > 0.0 {
        masked_valids = masked_valids.iter().map(|x| x / sum_ps_s).collect();
      } else {
        println!("all valids moves were masked");
        let sum_ps_s: i32 = valids.iter().map(|&x| x as i32).sum();
        masked_valids = valids.iter().map(|&x| x as i32 as f32 / sum_ps_s as f32).collect();
      }
      self.ps.insert(s, masked_valids);
      self.vs.insert(s, valids);
      self.ns.insert(s, 0);
      return -v;
    } else {
      self.cache_hit_cnt += 1;
    }
    let valids = &self.vs[&s];
    let mut cur_best = f32::MIN;
    let mut best_act: isize = -1;

    // pick best action
    let amax = c_board.action_size();
    for a in 0..amax {
      if !valids[a] { continue; }
      let u: f32;
      let sa = &(s, a);
      if self.qsa.contains_key(sa) {
        u = self.qsa[sa] + self.cpuct * self.ps[&s][a] * (self.ns[&s] as f32 / (1 + self.nsa[sa]) as f32).sqrt()
      } else {
        u = self.cpuct * self.ps[&s][a] * (self.ns[&s] as f32 + 1e-8).sqrt();
      }
      if u > cur_best {
        cur_best = u;
        best_act = a as isize;
      }
    }
    let a = best_act;
    if a < 0 { panic!("no valid moves"); }
    let a = a as usize;

    // play one step
    let mut next_s = c_board.next_state(a as u32);

    // search until leaf node
    let v = self.search(&mut next_s);

    // move back up the tree
    let sa = (s, a);
    if self.qsa.contains_key(&sa) {
      self.qsa.insert(sa, (self.nsa[&sa] as f32 * self.qsa[&sa] + v as f32) / (self.nsa[&sa] + 1) as f32);
      self.nsa.insert(sa, self.nsa[&sa] + 1);
    } else {
      self.qsa.insert(sa, v as f32);
      self.nsa.insert(sa, 1);
    }
    self.ns.insert(s, self.ns[&s] + 1);
    
    -v
  }
}