use super::*;
use std::cmp::Ordering;
use std::cmp::min;
use std::cmp::max;
use rand::distributions::WeightedIndex;
use rand::distributions::Distribution;

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
      wsa: HashMap::new(), // win times
      ns: HashMap::new(), // board visited times
      ps: HashMap::new(), // initial policy (returned by neural net)
      es: HashMap::new(), // game ended
      vs: HashMap::new(), // valid moves
    }
  }
  pub fn duplicate(mcts: &MCTS) -> MCTS {
    MCTS {
      sim_num: mcts.sim_num,
      cpuct: mcts.cpuct,
      qsa: HashMap::new(), // Q values
      nsa: HashMap::new(), // edge visited times
      wsa: HashMap::new(), // win times
      ns: HashMap::new(), // board visited times
      ps: HashMap::new(), // initial policy (returned by neural net)
      es: HashMap::new(), // game ended
      vs: HashMap::new(), // valid moves
    }
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
        // v*turn: 予測勝率で更新
        let win = v * turn + 0.1;
        self.wsa.insert(*sa, self.wsa[&sa] + win);
        self.qsa.insert(*sa, self.wsa[&sa] / self.nsa[&sa] as f32);
      }
    }
  }
  pub fn get_action_prob<F>(&mut self, c_board: &Board, temp: f32, predict: &F, prioritize_kill: bool,  for_train: bool, self_play: bool, komi: i32) -> Vec<f32>
    where 
      F: Fn(Vec<Vec<f32>>) -> Vec<(Vec<f32>, f32)>
    {
    let s = c_board.calc_hash();
    let amax = c_board.action_size();
    let sn = self.sim_num;
    let root_turn = c_board.turn;
    // init
    self.qsa = HashMap::new();
    self.nsa = HashMap::new();
    self.wsa = HashMap::new();
    self.ns = HashMap::new();
    let mut cnt = 0;
    let mut inputs: Vec<Vec<f32>> = Vec::new();
    let mut hashs: Vec<u64> = Vec::new();
    let mut nodes: Vec<Vec<((u64, usize), f32)>> = Vec::new();
    while cnt <= sn {
      let mut nodes_inside: Vec<((u64, usize), f32)> = Vec::new();
      let mut b = c_board.clone();
      let (_, leaf) = self.search(&mut b, &mut nodes_inside, prioritize_kill, for_train, self_play, komi);
      nodes.push(nodes_inside);
      if let Some(x) = leaf {
        let (input, s) = x;
        inputs.push(input);
        hashs.push(s);
      }
      cnt += 1;
      // multiple predicts in one step
      if (cnt < sn && inputs.len() >= 3) || cnt == sn && inputs.len() >= 1 {
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
        val = min(self.nsa[&(s, a)], 1000000); // inf not allowed
      }
      counts.push(val as f32);
    }
    //println!("counts: {:?}", counts);
    if temp == 0.0 {
      let mut probs = vec![0.0; amax];
      let best_idx = max_idx(&counts);
      probs[best_idx] = 1.0;
      return probs;
    }
    let counts: Vec<f32> = counts.iter().map(|&x| x.powf(1.0 / temp)).collect();
    let counts_sum: f32 = counts.iter().sum();
    let probs: Vec<f32>;
    if counts_sum == 0.0 {
      // avoid devide by zero
      probs = c_board.vec_valid_moves(root_turn).iter().map(|&x| x as i32 as f32).collect(); // random move
    } else {
      probs = counts.iter().map(|x| x / counts_sum).collect();
    }
    // let end = start.elapsed();
    // println!("getactionprob {}.{:03}秒", end.as_secs(), end.subsec_nanos() / 1_000_000);
    probs
  }
  pub fn search(&mut self, c_board: &mut Board, nodes: &mut Vec<((u64, usize), f32)>, prioritize_kill: bool, for_train: bool, self_play:bool, komi: i32) -> (f32, Option<(Vec<f32>, u64)>) {
    let s = c_board.calc_hash();
    //println!("search {} {:?}", c_board, s);
    if !self.es.contains_key(&s) {
      let auto_resign = true;
      self.es.insert(s, c_board.game_ended(auto_resign, komi) as f32);
    }
    if self.es[&s] != 0.0 {
      // terminal node
      // println!("game end {:?}", self.es[&s]);
      return (self.es[&s], None);
    }
    if !self.ns.contains_key(&s) {
      // leaf node
      let valids;
      if for_train {
        valids = c_board.vec_valid_moves_for_train(c_board.turn);
      } else {
        valids = c_board.vec_valid_moves(c_board.turn);
      }
      self.ps.insert(s, valids.iter().map(|&v| v as i32 as f32).collect());
      self.vs.insert(s, valids);
      self.ns.insert(s, 0);
      // バーチャルロスを反映
      for n in nodes {
        let (sa, _) = n;
        let win = -0.1;
        if self.nsa.contains_key(sa) {
          self.wsa.insert(*sa, self.wsa[sa] + win);
        } else {
          self.wsa.insert(*sa, win);
          self.nsa.insert(*sa, 0);
        }
      }
      let leaf = Some((c_board.input(), s));
      return (0.0, leaf);
    }
    let turn = c_board.turn as i32 as f32;
    let valids = &self.vs[&s];
    // pick best action
    let amax = c_board.action_size();
    let mut probs: Vec<f32> = vec![];
    for a in 0..amax {
      if !valids[a] { 
        probs.push(0.0);
        continue;
      }
      let u: f32;
      let sa = &(s, a);
      if self.qsa.contains_key(sa) {
        u = self.qsa[sa] + self.cpuct * self.ps[&s][a] * (self.ns[&s] as f32).sqrt() / (1 + self.nsa[sa]) as f32;
      } else {
        u = self.cpuct * self.ps[&s][a] * (self.ns[&s] as f32 + 1e-8).sqrt();
      }
      probs.push(u);
    }
    if prioritize_kill {
      // 石を殺すことを優先する
      let kp = c_board.kill_point(c_board.turn).vec();
      for i in 0..kp.len() {
        probs[i] += kp[i];
      }
    }
    // 次の手を選ぶ
    let a: usize;
    if self_play && c_board.step < c_board.size as u32 {
      // 確率で次の手を選ぶ
      let pmin = probs.iter().fold(f32::INFINITY, |m, v| v.min(m));
      probs = probs.iter().map(|&p| if p==0.0 {0.0} else {p-pmin+0.0000001}).collect(); // マイナスを許容しない
      let rng = &mut rand::thread_rng();
      let dist = WeightedIndex::new(&probs).unwrap();
      a = dist.sample(rng);
    } else {
      // 最良の手を選ぶ
      probs = probs.iter().map(|&p| if p==0.0 {-100.0} else {p}).collect(); // ゼロは無視
      a = max_idx(&probs);
    }
    // if c_board.step == 1  && a==25{
    //   println!("probs {:?}", probs);
    // }
    // println!("valids {:?}", valids);

    // play one step
    c_board.action(a as u32, c_board.turn);

    // maximum step
    if c_board.step > c_board.size as u32 * c_board.size as u32 * 2  {
      // println!("step action {}", a);
      // println!("{:?}", c_board.get_kifu_sgf());
      // println!("{}", c_board);
      return (0.0, None);
    }

    // search until leaf node
    let sa = (s, a);
    nodes.push((sa, turn));
    let sstep = c_board.step;
    let (v, leaf) = self.search(c_board, nodes, prioritize_kill, for_train, self_play, komi);
    // if sstep == 1 && a == 25 {
    //   if self.nsa.contains_key(&sa) {
    //     println!("pass wsa {:?} / nsa {:?} = {:?} v {:?}", self.wsa[&sa], self.nsa[&sa], self.wsa[&sa]/self.nsa[&sa] as f32, v);
    //   }
    // }

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