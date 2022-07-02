use super::*;
use std::cmp::Ordering;
use std::cmp::min;
use std::cmp::max;
use rand::distributions::WeightedIndex;
use rand::distributions::Distribution;
use rand::Rng;
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
      wsa: HashMap::new(), // win times
      ns: HashMap::new(), // board visited times
      ps: HashMap::new(), // initial policy (returned by neural net)
      psv: HashMap::new(),
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
      // ps: mcts.ps.clone(), // initial policy (returned by neural net)
      // psv: mcts.psv.clone(),
      ps: HashMap::new(), // initial policy (returned by neural net)
      psv: HashMap::new(),
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
    
    let mut hash_cnt = 0;
    for s in hashs {
      if self.psv.contains_key(&s) {
        hash_cnt += 1;
      }
    }
    let mut predicts: Vec<(Vec<f32>, f32)>;
    if hash_cnt == batch_size {
      // 全部キャッシュにある場合
      predicts = Vec::new();
      for b in 0..batch_size {
        let s = hashs[b];
        predicts.push((self.ps[&s].clone(), self.psv[&s]));
      }
    } else {
      predicts = predict(iv);
    }
    for b in 0..batch_size {
      let s = hashs[b];
      let (ps, v) = &predicts[b];
      // ----------------------
      // パスは確率を下げる
      // let mut ps = ps2.clone();
      // let pass_idx = ps.len()-1;
      // ps[pass_idx] = ps[pass_idx] * 0.1;
      // ----------------------
      let valids = &self.vs[&s];
      let mut masked_valids: Vec<f32> = valids.iter().enumerate().map(|(i, x)| *x as i32 as f32 * ps[i] + 1e-20).collect();
      let sum_ps_s: f32 = masked_valids.iter().sum();
      if sum_ps_s > 0.0 {
        masked_valids = masked_valids.iter().map(|x| x / sum_ps_s).collect();
      } else {
        println!("all valids moves were masked {:?}", ps);
        println!("valids {:?}", valids);
        let sum_ps_s: i32 = valids.iter().map(|&x| x as i32).sum();
        masked_valids = valids.iter().map(|&x| x as i32 as f32 / sum_ps_s as f32).collect();
      }
      self.ps.insert(s, masked_valids);
      self.psv.insert(s, *v);
      // move back down the tree
      let sections = &nodes[b];
      if sections.len() == 0 {
        continue;
      }
      let (leaf_sa, turn) = sections.last().unwrap();
      let mut wv:f32 = *v * turn; // 1.0はvirtual loss分
      for section in sections.into_iter().rev() {
        let (sa, turn) = section;
        let mut win = wv;// + turn;
        // if sa == leaf_sa {
           win += 1.0;
        // }
        self.wsa.insert(*sa, self.wsa[&sa] + win);
        self.qsa.insert(*sa, self.wsa[&sa] / self.nsa[&sa] as f32);
        wv *= -1.0;
      }
    }
  }
  pub fn get_action_prob<F>(&mut self, c_board: &Board, temp: f32, predict: &F, prioritize_kill: bool,  for_train: bool, self_play: bool, komi: i32) -> Vec<f32>
    where 
      F: Fn(Vec<Vec<f32>>) -> Vec<(Vec<f32>, f32)>
    {
    let s = c_board.calc_hash();
    let amax = c_board.action_size();
    let mut sn = self.sim_num;
    let root_turn = c_board.turn;
    // if temp > 0.11 && root_turn == Turn::Black {
    //   // 黒番だけ読みを深く入れる
    //   sn *= 50;
    // }
    // init
    self.qsa = HashMap::new();
    self.nsa = HashMap::new();
    self.wsa = HashMap::new();
    self.ns = HashMap::new();
    let mut cnt = 0;
    let mut inputs: Vec<Vec<f32>> = Vec::new();
    let mut hashs: Vec<u64> = Vec::new();
    let mut nodes: Vec<Vec<((u64, usize), f32)>> = Vec::new();
    let mut simNum = 1; // 初期盤面はすぐにpredictする
    while cnt < sn {
      let mut nodes_inside: Vec<((u64, usize), f32)> = Vec::new();
      let mut b = c_board.clone();
      //println!("------ call search ---------------");
      let (_, leaf) = self.search(&mut b, &mut nodes_inside, prioritize_kill, for_train, self_play, komi);
      nodes.push(nodes_inside);
      if let Some(x) = leaf {
        let (input, s) = x;
        inputs.push(input);
        hashs.push(s);
      }
      cnt += 1;
      //println!("cnt {:?} inputs {:?} ------------------------", cnt, inputs.len());
      // multiple predicts in one step
      if (cnt < sn && inputs.len() >= simNum) || (cnt == sn && inputs.len() >= 1) {
        self.predict_leaf(&nodes, &inputs, &hashs, predict);
        inputs = Vec::new();
        hashs= Vec::new();
        nodes = Vec::new();
        simNum = 8; // 次回以降は8つ同時にpredictする
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
    // println!("wsa: {:?}", self.wsa);
    //println!("nsa: {:?}", self.nsa);
    // println!("counts: {:?}", counts);
    //println!("qsa: {:?}", self.qsa);
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
      probs = c_board.vec_valid_moves_for_cpu(root_turn).iter().map(|&x| x as i32 as f32).collect(); // random move
    } else {
      probs = counts.iter().map(|x| x / counts_sum).collect();
    }
    // let end = start.elapsed();
    // println!("getactionprob {}.{:03}秒", end.as_secs(), end.subsec_nanos() / 1_000_000);
    probs
  }
  async fn predict_leaf_async<F, Fut>(&mut self, nodes: &Vec<Vec<((u64, usize), f32)>>, inputs: &Vec<Vec<f32>>, hashs: &Vec<u64>, predict: &F)
    where 
      F: Fn(Vec<Vec<f32>>) -> Fut,
      Fut: Future<Output = Vec<(Vec<f32>, f32)>>
    {
    let iv = inputs.to_vec();
    let batch_size = iv.len();
    
    let mut hash_cnt = 0;
    for s in hashs {
      if self.psv.contains_key(&s) {
        hash_cnt += 1;
      }
    }
    let mut predicts: Vec<(Vec<f32>, f32)>;
    if hash_cnt == batch_size {
      // 全部キャッシュにある場合
      predicts = Vec::new();
      for b in 0..batch_size {
        let s = hashs[b];
        predicts.push((self.ps[&s].clone(), self.psv[&s]));
      }
    } else {
      predicts = predict(iv).await;
    }
    for b in 0..batch_size {
      let s = hashs[b];
      let (ps, v) = &predicts[b];
      // ----------------------
      // パスは確率を下げる
      // let mut ps = ps2.clone();
      // let pass_idx = ps.len()-1;
      // ps[pass_idx] = ps[pass_idx] * 0.1;
      // ----------------------
      let valids = &self.vs[&s];
      let mut masked_valids: Vec<f32> = valids.iter().enumerate().map(|(i, x)| *x as i32 as f32 * ps[i] + 1e-20).collect();
      let sum_ps_s: f32 = masked_valids.iter().sum();
      if sum_ps_s > 0.0 {
        masked_valids = masked_valids.iter().map(|x| x / sum_ps_s).collect();
      } else {
        println!("all valids moves were masked {:?}", ps);
        println!("valids {:?}", valids);
        let sum_ps_s: i32 = valids.iter().map(|&x| x as i32).sum();
        masked_valids = valids.iter().map(|&x| x as i32 as f32 / sum_ps_s as f32).collect();
      }
      self.ps.insert(s, masked_valids);
      self.psv.insert(s, *v);
      // move back down the tree
      let sections = &nodes[b];
      if sections.len() == 0 {
        continue;
      }
      let (leaf_sa, turn) = sections.last().unwrap();
      let mut wv:f32 = *v * turn; // 1.0はvirtual loss分
      for section in sections.into_iter().rev() {
        let (sa, turn) = section;
        let mut win = wv;// + turn;
        // if sa == leaf_sa {
           win += 1.0;
        // }
        self.wsa.insert(*sa, self.wsa[&sa] + win);
        self.qsa.insert(*sa, self.wsa[&sa] / self.nsa[&sa] as f32);
        wv *= -1.0;
      }
    }
  }
  pub async fn get_action_prob_async<F, Fut>(&mut self, c_board: &Board, temp: f32, predict: &F, prioritize_kill: bool,  for_train: bool, self_play: bool, komi: i32) -> Vec<f32>
    where 
      F: Fn(Vec<Vec<f32>>) -> Fut,
      Fut: Future<Output = Vec<(Vec<f32>, f32)>>
    {
    let s = c_board.calc_hash();
    let amax = c_board.action_size();
    let mut sn = self.sim_num;
    let root_turn = c_board.turn;
    // if temp > 0.11 && root_turn == Turn::Black {
    //   // 黒番だけ読みを深く入れる
    //   sn *= 50;
    // }
    // init
    self.qsa = HashMap::new();
    self.nsa = HashMap::new();
    self.wsa = HashMap::new();
    self.ns = HashMap::new();
    let mut cnt = 0;
    let mut inputs: Vec<Vec<f32>> = Vec::new();
    let mut hashs: Vec<u64> = Vec::new();
    let mut nodes: Vec<Vec<((u64, usize), f32)>> = Vec::new();
    let mut simNum = 1; // 初期盤面はすぐにpredictする
    while cnt < sn {
      let mut nodes_inside: Vec<((u64, usize), f32)> = Vec::new();
      let mut b = c_board.clone();
      //println!("------ call search ---------------");
      let (_, leaf) = self.search(&mut b, &mut nodes_inside, prioritize_kill, for_train, self_play, komi);
      nodes.push(nodes_inside);
      if let Some(x) = leaf {
        let (input, s) = x;
        inputs.push(input);
        hashs.push(s);
      }
      cnt += 1;
      //println!("cnt {:?} inputs {:?} ------------------------", cnt, inputs.len());
      // multiple predicts in one step
      if (cnt < sn && inputs.len() >= simNum) || (cnt == sn && inputs.len() >= 1) {
        self.predict_leaf_async(&nodes, &inputs, &hashs, predict).await;
        inputs = Vec::new();
        hashs= Vec::new();
        nodes = Vec::new();
        simNum = 8; // 次回以降は8つ同時にpredictする
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
    // println!("wsa: {:?}", self.wsa);
    //println!("nsa: {:?}", self.nsa);
    //println!("counts: {:?}", counts);
    //panic!("nsa: {:?}", self.nsa);
    //println!("qsa: {:?}", self.qsa);
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
      probs = c_board.vec_valid_moves_for_cpu(root_turn).iter().map(|&x| x as i32 as f32).collect(); // random move
    } else {
      probs = counts.iter().map(|x| x / counts_sum).collect();
    }
    // let end = start.elapsed();
    // println!("getactionprob {}.{:03}秒", end.as_secs(), end.subsec_nanos() / 1_000_000);
    probs
  }
  pub fn search(&mut self, c_board: &mut Board, nodes: &mut Vec<((u64, usize), f32)>, prioritize_kill: bool, for_train: bool, self_play:bool, komi: i32) -> (f32, Option<(Vec<f32>, u64)>) {
    let s = c_board.calc_hash();
    let turn = c_board.turn as i32 as f32;
    if !self.es.contains_key(&s) {
      let auto_resign = self_play;
      self.es.insert(s, c_board.game_ended(auto_resign, komi) as f32);
    }
    if self.es[&s] != 0.0 {
      // terminal node
      return (-self.es[&s]*turn*2.0, None);
    }
    if !self.ns.contains_key(&s) {
      // leaf node
      let valids;
      if for_train || prioritize_kill {
        valids = c_board.vec_valid_moves_for_train(c_board.turn);
      } else {
        valids = c_board.vec_valid_moves_for_cpu(c_board.turn);
      }
      
      self.vs.insert(s, valids);
      // if self.psv.contains_key(&s) {
      //   // predict済みのものはキャッシュを使う
      //   //println!("hit cache {:?}", &s);
      //   return (-self.psv[&s], None);
      // }
      let mut inputs: Vec<Vec<f32>> = Vec::new();
      inputs.push(c_board.input());
      
      self.ps.insert(s, self.vs[&s].iter().map(|&v| if v {0.1} else {0.0}).collect());
      self.ns.insert(s, 0);
      let leaf = Some((c_board.input(), s));
      //println!("vloss");
      // virtual loss
      return (-10.0, leaf);
    }
    let valids = &self.vs[&s];
    // pick best action
    let amax = c_board.action_size();
    let mut probs: Vec<f32> = vec![];
    let mut rng = rand::thread_rng();
    let c_base = 2000;
    let c_init = 1.25;
    for a in 0..amax {
      if !valids[a] { 
        // 合法手のみに絞る
        probs.push(-10000000.0);
        continue;
      }
      let u: f32;
      let sa = &(s, a);
      let c: f32 = ((1 + self.ns[&s]+c_base) as f32 / c_base as f32).log(2.7182818284) + c_init;
      if self.qsa.contains_key(sa) {
        u = self.qsa[sa] + c * (self.ps[&s][a] + 1e-32) * (self.ns[&s] as f32).sqrt() / (1 + self.nsa[sa]) as f32;
      } else {
        u = c * (self.ps[&s][a] + 1e-32) * (self.ns[&s] as f32 + 1e-32).sqrt();
      }
      //println!("u {:?}", u);
      // add noise
      let noise: f32 = rng.gen();
      probs.push(u);
      //probs.push(u + noise / 100000.0);
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
    // if self_play && c_board.step < c_board.size as u32 {
    //   // 確率で次の手を選ぶ
    //   let pmin = probs.iter().fold(f32::INFINITY, |m, v| v.min(m));
    //   probs = probs.iter().map(|&p| if p==0.0 {0.0} else {p-pmin+1e-20}).collect(); // マイナスを許容しない
    //   let rng = &mut rand::thread_rng();
    //   let dist = WeightedIndex::new(&probs).unwrap();
    //   a = dist.sample(rng);
    // } else {
      // 最良の手を選ぶ
      a = max_idx(&probs);
    // }
    // println!("probs {:?} {:?} {:?}", probs[27], probs[48], probs[49]);
    // if self.qsa.contains_key(&(s, 27)) {
    //   println!("wsa27 {:?} nsa27 {:?} qsa27 {:?}", self.wsa[&(s, 27)], self.nsa[&(s, 27)], self.qsa[&(s, 27)]);
    // }
    // if self.qsa.contains_key(&(s, 48)) {
    //   println!("wsa48 {:?} nsa48 {:?}", self.wsa[&(s, 48)], self.nsa[&(s, 48)]);
    // }
    // if self.qsa.contains_key(&(s, 49)) {
    //   println!("wsa49 {:?} nsa49 {:?}", self.wsa[&(s, 49)], self.nsa[&(s, 49)]);
    // }
    // //println!("ps {:?} {:?} {:?}", self.ps[&s][27], self.ps[&s][48], self.ps[&s][49]);
    // println!("===action {:?} turn {:?}", a, turn);

    // play one step
    c_board.action(a as u32, c_board.turn);
    // maximum step
    if c_board.step > c_board.size as u32 * c_board.size as u32 * 2  {
      let score = c_board.count_diff();
      if score > 0 {
        return (-turn, None);
      }
      return (turn, None);
    }

    // search until leaf node
    let sa = (s, a);
    nodes.push((sa, turn));
    let (mut v, leaf) = self.search(c_board, nodes, prioritize_kill, for_train, self_play, komi);
    let mut win = v;
    if v < -5.0 {
      // virtual loss
      win = -1.0;
    }
    //println!("a {:?} win {:?}", a, win);
    // move back up the tree
    if self.nsa.contains_key(&sa) {
      self.wsa.insert(sa, self.wsa[&sa] + win);
      self.nsa.insert(sa, self.nsa[&sa] + 1);
    } else {
      self.wsa.insert(sa, win);
      self.nsa.insert(sa, 1);
    }
    self.qsa.insert(sa, self.wsa[&sa] / self.nsa[&sa] as f32);
    self.ns.insert(s, self.ns[&s] + 1);
    //println!("win {:?} qsa {:?} sa:{:?}", win, self.qsa[&sa], sa);
    if v < -5.0 {
      // virtual lossはそのまま伝播
      return (v, leaf);
    }
    (-v, leaf)
  }
}