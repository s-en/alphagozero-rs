use super::*;
use std::cmp::Ordering;
use std::future::Future;

pub fn max_idx(vals: &Vec<f32>) -> usize {
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
  pub fn new(sim_num: u32, cpuct: f32) -> MCTS {
    MCTS {
      sim_num: sim_num,
      cpuct: cpuct,
      qsa: HashMap::new(), // Q values
      nsa: HashMap::new(), // edge visited times
      wsa: HashMap::new(), // win probability
      ns: HashMap::new(), // board visited times
      ps: HashMap::new(), // initial policy (returned by neural net)
      es: HashMap::new(), // game ended
      vs: HashMap::new(), // valid moves
    }
  }
  pub fn get_win_rate(&self, c_board: &Board) -> f32 {
    let s = c_board.calc_hash();
    let amax = c_board.action_size();
    let mut qsas = Vec::new();
    for a in 0..amax {
      let mut val = 0.0;
      if self.qsa.contains_key(&(s, a)) {
        val = self.qsa[&(s, a)];
      }
      qsas.push(val);
    }
    //println!("qsas {:?}", qsas);
    //println!("");
    0.0
  }
  fn predict_leaf<F>(&mut self, nodes: &Vec<Vec<((u64, usize), f32)>>, inputs: &Vec<Vec<f32>>, hashs: &Vec<u64>, predict: &F)
    where 
      F: Fn(Vec<Vec<f32>>) -> Vec<(Vec<f32>, f32)>
    {
    let iv = inputs.to_vec();
    let batch_size = iv.len();
    let predicts = predict(iv);
    for b in 0..batch_size {
      let (ps, v) = &predicts[b];
      let s = hashs[b];
      let valids = &self.vs[&s];
      let mut masked_valids: Vec<f32> = valids.iter().enumerate().map(|(i, x)| *x as i32 as f32 * ps[i]).collect();
      let sum_ps_s: f32 = masked_valids.iter().sum();
      if sum_ps_s > 0.0 {
        masked_valids = masked_valids.iter().map(|x| x / sum_ps_s).collect();
      } else {
        println!("all valids moves were masked {:?}", ps);
        let sum_ps_s: i32 = valids.iter().map(|&x| x as i32).sum();
        masked_valids = valids.iter().map(|&x| x as i32 as f32 / sum_ps_s as f32).collect();
      }
      self.ps.insert(s, masked_valids);
      // move back down the tree
      let sections = &nodes[b];
      for section in sections {
        let (sa, turn) = section;
        let win = v * turn * 1.0;
        self.wsa.insert(*sa, self.wsa[&sa] + win);
        self.qsa.insert(*sa, self.wsa[&sa] / self.nsa[&sa] as f32);
      }
    }
  }
  pub fn get_action_prob<F>(&mut self, c_board: &Board, temp: f32, predict: &F) -> Vec<f32>
    where 
      F: Fn(Vec<Vec<f32>>) -> Vec<(Vec<f32>, f32)>
    {
    let s = c_board.calc_hash();
    let amax = c_board.action_size();
    let mut sn = self.sim_num;
    let root_turn = c_board.turn;
    // if root_turn == Turn::White {
    //   // handicap for white player
    //   sn *= 2;
    // }
    let mut cnt = 0;
    let mut inputs: Vec<Vec<f32>> = Vec::new();
    let mut hashs: Vec<u64> = Vec::new();
    let mut nodes: Vec<Vec<((u64, usize), f32)>> = Vec::new();
    while cnt <= sn {
      let mut nodes_inside: Vec<((u64, usize), f32)> = Vec::new();
      let mut b = c_board.clone();
      let (_, leaf) = self.search(&mut b, &mut nodes_inside, root_turn);
      nodes.push(nodes_inside);
      if let Some(x) = leaf {
        let (input, s) = x;
        inputs.push(input);
        hashs.push(s);
      }
      cnt += 1;
      // multiple predicts in one step
      if (cnt < sn && inputs.len() >= 16) || cnt == sn && inputs.len() >= 1 {
        self.predict_leaf(&nodes, &inputs, &hashs, predict);
        inputs = Vec::new();
        hashs= Vec::new();
        nodes = Vec::new();
      }
    }
    let mut counts = Vec::new();
    for a in 0..amax {
      let mut val = 0;
      if self.nsa.contains_key(&(s, a)) {
        val = self.nsa[&(s, a)];
      }
      counts.push(val as f32);
    }
    // println!("counts: {:?}", counts);
    if temp == 0.0 {
      let mut probs = vec![0.0; amax];
      let best_idx = max_idx(&counts);
      probs[best_idx] = 1.0;
      return probs;
    }
    let counts: Vec<f32> = counts.iter().map(|&x| x.powf(1.0 / temp)).collect();
    let counts_sum: f32 = counts.iter().sum();
    let probs: Vec<f32> = counts.iter().map(|x| x / counts_sum).collect();
    // let end = start.elapsed();
    // println!("getactionprob {}.{:03}秒", end.as_secs(), end.subsec_nanos() / 1_000_000);
    probs
  }
  pub fn search(&mut self, c_board: &mut Board, nodes: &mut Vec<((u64, usize), f32)>, root_turn: Turn) -> (f32, Option<(Vec<f32>, u64)>) {
    let s = c_board.calc_hash();
    //println!("search {} {:?}", c_board, s);
    if !self.es.contains_key(&s) {
      self.es.insert(s, c_board.game_ended() as f32);
    }
    if self.es[&s] != 0.0 {
      // terminal node
      return (self.es[&s], None);
    }
    if !self.ps.contains_key(&s) {
      // leaf node
      let valids = c_board.vec_valid_moves(c_board.turn);
      self.ps.insert(s, valids.iter().map(|&v| v as i32 as f32).collect());
      self.vs.insert(s, valids);
      self.ns.insert(s, 0);
      let v_loss = 0.0;//root_turn as i32 as f32 * -1.0;
      let leaf = Some((c_board.input(), s));
      return (v_loss, leaf);
    }
    let valids = &self.vs[&s];
    let mut cur_best = f32::MIN;
    let mut best_act: isize = -1;

    // pick best action
    let amax = c_board.action_size();
    let mut temp: Vec<f32> = Vec::new();
    for a in 0..amax {
      if !valids[a] { continue; }
      let u: f32;
      let sa = &(s, a);
      if self.qsa.contains_key(sa) {
        u = self.qsa[sa] + self.cpuct * self.ps[&s][a] * (self.ns[&s] as f32).sqrt() / (1 + self.nsa[sa]) as f32;
      } else {
        u = self.cpuct * self.ps[&s][a] * (self.ns[&s] as f32 + 1e-8).sqrt();
      }
      //println!("s:{:?} a:{:?} u:{:?}", s, a, u);
      temp.push(u);
      if u > cur_best {
        cur_best = u;
        best_act = a as isize;
      }
    }
    let a = best_act;
    // println!("valids {:?}", valids);
    if a < 0 { panic!("no valid moves"); }
    let a = a as usize;
    //println!("action {}", a);

    // play one step
    let turn = c_board.turn as i32 as f32;
    c_board.action(a as u32, c_board.turn);

    // maximum step
    if c_board.step > 50 {
      // println!("step action {}", a);
      // println!("{:?}", c_board.get_kifu_sgf());
      // println!("{}", c_board);
      return (0.0, None);
    }

    // search until leaf node
    let sa = (s, a);
    nodes.push((sa, turn));
    let (v, leaf) = self.search(c_board, nodes, root_turn);

    // move back up the tree
    let mut win = v * turn;
    // if win < 0.0 {
    //   win = 0.0;
    // }
    if self.nsa.contains_key(&sa) {
      self.wsa.insert(sa, self.wsa[&sa] + win);
      self.nsa.insert(sa, self.nsa[&sa] + 1);
    } else {
      self.wsa.insert(sa, win);
      self.nsa.insert(sa, 1);
    }
    self.qsa.insert(sa, self.wsa[&sa] / self.nsa[&sa] as f32);
    self.ns.insert(s, self.ns[&s] + 1);
    //println!("qsa {:?} = {:?}/{:?}",self.qsa, self.wsa[&sa], self.nsa[&sa]);
    
    (v, leaf)
  }
}