use super::*;

use rand::distributions::WeightedIndex;
use std::collections::VecDeque;
use std::thread;
use std::sync::mpsc;
use std::sync::{Mutex, Arc};
use std::time::Instant;
use std::cmp;
use std::collections::HashMap;

extern crate savefile;

const KOMI: i32 = 0;
const BOARD_SIZE: BoardSize = BoardSize::S7;

fn self_play_sim(
  arc_examples: &mut Arc<Mutex<Vec<Example>>>, 
  board_size: i64, 
  action_size: i64, 
  num_channels: i64, 
  num_eps: i32, 
  sim_num: i32, 
  mcts: &mut MCTS) {
  let maxlen_of_queue = 1000000;
  let mut rng = rand::thread_rng();
  let (tx, rx) = mpsc::channel();
  for _ in 0..sim_num {
    let tx1 = mpsc::Sender::clone(&tx);
    let mut each_mcts = MCTS::duplicate(&mcts);
    thread::spawn(move || {
      let examples = self_play(board_size, action_size, num_channels, num_eps, &mut each_mcts);
      tx1.send(examples).unwrap();
    });
  }
  drop(tx);
  {
    let mut results = vec![];
    for examples in rx {
      results.extend(examples);
    }
    println!("self play done. we've got {} examples", results.len());
    let train_examples = &mut *arc_examples.lock().unwrap();
    train_examples.extend(results);
    let tlen = train_examples.len();
    if tlen > maxlen_of_queue {
      let cut = tlen - maxlen_of_queue;
      train_examples.drain(..cut);
    }
    train_examples.shuffle(&mut rng);
  }
}

fn self_play(board_size: i64, 
  action_size: i64, 
  num_channels: i64, 
  num_eps: i32,
  mcts: &mut MCTS) -> Vec<Example> {
  let max_history_queue = 40000;
  let rng = &mut rand::thread_rng();
  let mut black_win_history = Examples {
    values: VecDeque::with_capacity(max_history_queue)
  };
  let mut white_win_history = Examples {
    values: VecDeque::with_capacity(max_history_queue)
  };
  let mut tnet = NNet::new(board_size, action_size, num_channels);
  tnet.load("temp/best.pt");
  let mut each_mcts = MCTS::duplicate(mcts);
  for i in 0..num_eps {
    each_mcts = MCTS::extend(each_mcts); // reset search tree
    let ep_results = execute_episode(rng, &mut each_mcts, &tnet, i);
    mcts.append(&each_mcts);
    //println!("before await");
    let (examples, r) = ep_results;
    //println!("after await examples:{}", examples.len());
    for ex in examples {
      if r == 1 {
        black_win_history.values.push_back(ex);
      } else if r == -1 {
        white_win_history.values.push_back(ex);
      }
    }
  }
  let bw_min = cmp::min(cmp::min(black_win_history.values.len(), white_win_history.values.len()), max_history_queue);
  if black_win_history.values.len() > bw_min {
    let cut = black_win_history.values.len() - bw_min;
    black_win_history.values.drain(..cut);
  }
  if white_win_history.values.len() > bw_min {
    let cut = white_win_history.values.len() - bw_min;
    white_win_history.values.drain(..cut);
  }
  // println!("example black {}, white {}", black_win_history.values.len(), white_win_history.values.len());

  // shuffle examples before training
  let mut results: Vec<Example> = Vec::new();
  //let train_examples = &mut *ex_arc_mut.lock().unwrap();
  for e in black_win_history.values {
    results.push(e);
  }
  for e in white_win_history.values {
    results.push(e);
  }
  results
}
fn execute_episode(rng: &mut ThreadRng, mcts: &mut MCTS, net: &NNet, eps_cnt: i32) -> (Vec<Example>, i8) {
  //let start = Instant::now();
  let mut examples = Vec::new();
  let mut board = Board::new(BOARD_SIZE);
  let mut episode_step = 0;
  let temp_threshold = 5;
  let mut kcnt = 0;
  let predict32 = |inputs: Vec<Vec<f32>>| {
    NNet::predict32(net, inputs)
  };
  let for_train = eps_cnt % 2 == 0;
  loop {
    episode_step += 1;
    let mut temp = 0.5;
    if episode_step < temp_threshold {
      temp = 2.0;
    } else if episode_step < temp_threshold*2 {
      temp = 1.0;
    }
    //println!("step {:?} turn {:?}", board.step, board.turn as i32);
    //let mstart = Instant::now();
    let pi = mcts.get_action_prob(&board, temp, &predict32, for_train, KOMI);
    // let mend = mstart.elapsed();
    // println!("get_action_prob {}.{:03}秒", mend.as_secs(), mend.subsec_nanos() / 1_000_000);
    
    let dist = WeightedIndex::new(&pi).unwrap();
    let sym = board.symmetries(pi);
    for (b, p) in sym {
      let ex = Example {
        board: b,
        pi: p,
        v: 0.0
      };
      examples.push(ex);
    }
    let action = dist.sample(rng) as u32;
    board.action(action, board.turn);

    let r = board.game_ended(true, KOMI);

    if r != 0 {
      // println!("{} {}", r, board.get_kifu_sgf());
      // std::process::exit(0x0100);
      let v = r as i32;
      for mut ex in &mut examples {
        // v: 1.0 black won
        // v: -1.0 white won
         ex.v = v as f32;
      }
      // println!("{:?}", examples);
      // let end = start.elapsed();
      // println!("execute_episode {}.{:03}秒", end.as_secs(), end.subsec_nanos() / 1_000_000);
      return (examples, r);
    }
  }
}
fn train_net(ex_arc_mut: &mut Arc<Mutex<Vec<Example>>>, board_size: i64, action_size: i64, num_channels: i64, lr: f64) {
  let mut net = NNet::new(board_size, action_size, num_channels);
  let board = Board::new(BOARD_SIZE);
  net.load_trainable("temp/best.pt");
  let pi = NNet::predict(&net, board.input());
  println!("before {:?}", pi);

  let mut examples: Vec<Example> = Vec::new();
  {
    let train_examples: &Vec<Example> = &*ex_arc_mut.lock().unwrap();
    for e in train_examples {
      let ex = Example {
        board: e.board.clone(),
        pi: e.pi.clone(),
        v: e.v
      };
      examples.push(ex);
    }
  }
  let ex = examples.iter().map(|x| x).collect();
  let _ = net.train(ex, lr);
  net.save("temp/trained.pt");

  let pi = NNet::predict(&net, board.input());
  println!("after {:?}", pi);
}
fn arena(mcts_sim_num: u32, 
  board_size: i64, 
  action_size: i64, 
  num_channels: i64,
  mcts: &mut MCTS) {
  let update_threshold = 0.6;
  
  // training new network, keeping a copy of the old one
  println!("PITTING AGAINST PREVIOUS VERSION");
  let game_result;
  {
    game_result = play_games(80, mcts_sim_num, board_size, action_size, num_channels, mcts);
  }
  let (pwins, nwins, draws) = game_result;

  println!("NEW/PREV WINS : {} / {} ; DRAWS : {}", nwins, pwins, draws);
  let mut net = NNet::new(board_size, action_size, num_channels);
  if pwins + nwins == 0 || nwins * 100 / (pwins + nwins) < (update_threshold * 100.0) as u32 {
    println!("REJECTING NEW MODEL");
  } else {
    println!("ACCEPTING NEW MODEL");
    net.load_trainable("temp/trained.pt");
    net.save("temp/best.pt");
  }
}
fn player<'a>(_mcts: &'a mut MCTS, _net: &'a NNet, temp: f32, count: u64) -> Box<dyn FnMut(&mut Board) -> usize + 'a> {
  let mut rng = rand::thread_rng();
  Box::new(move |x: &mut Board| -> usize {
    if temp >= 1.0 {
      let mut valids: Vec<usize> = x.vec_valid_moves(x.turn).iter().enumerate().map(|(i, &val)| {
        if val { i } else { 10000 }
      }).filter(|&r| r < 10000).collect();
      valids.shuffle(&mut rng);
      // println!("{:?}", valids);
      valids[0]
    } else {
      let predict32 = |inputs: Vec<Vec<f32>>| {
        NNet::predict32(_net, inputs)
      };
      let for_train = false;
      let probs = _mcts.get_action_prob(&x, temp, &predict32, for_train, KOMI);
      let dist = WeightedIndex::new(&probs).unwrap();
      dist.sample(&mut rng)
    }
  })
}
pub fn play_game<F: FnMut(&mut Board) -> usize>(player1: &mut F, player2: & mut F, count: u32) -> i8 {
  let players = [player2, player1];
  let mut cur_player = 1;
  let mut board = Board::new(BOARD_SIZE);
  let mut it = 0;
  while board.game_ended(false, KOMI) == 0 {
    it += 1;
    let action = players[cur_player](&mut board);
    let valids = board.vec_valid_moves(board.turn);
    if !valids[action] {
      println!("Action {} is not valid!", action);
      println!("valids = {:?}", valids);
      println!("{}", board.get_kifu_sgf());
      println!("{}", board);
      assert!(valids[action]);
    }
    board.action(action as u32, board.turn);
    cur_player = (board.turn as i32 + 1) as usize / 2;
  }
  if count <= 2 {
    println!("{} {}", board.game_ended(false, KOMI) ,board.get_kifu_sgf());
    println!("");
  }
  board.game_ended(false, KOMI)
}
pub fn play_games(
  num: u32, 
  mcts_sim_num: u32, 
  board_size: i64, 
  action_size: i64, 
  num_channels: i64,
  mcts: &mut MCTS) -> (u32, u32, u32) {
  let num = num / 2;
  let mut one_won = 0;
  let mut two_won = 0;
  let mut draws = 0;

  let (tx, rx) = mpsc::channel();
  let tx2 = mpsc::Sender::clone(&tx);
  let temp = 0.2;
  let each_mcts = MCTS::duplicate(mcts);
  thread::spawn(move || {
    let mut a = 0;
    let mut b = 0;
    let mut c = 0;
    let mut net = NNet::new(board_size, action_size, num_channels);
    let mut pnet = NNet::new(board_size, action_size, num_channels);
    net.load("temp/trained.pt");
    pnet.load("temp/best.pt");
    let mut pmcts = MCTS::duplicate(&each_mcts);
    let mut nmcts = MCTS::duplicate(&each_mcts);
    for i in 0..num {
      pmcts = MCTS::extend(pmcts); // reset search tree
      nmcts = MCTS::extend(nmcts); // reset search tree
      let mut player1 = player(&mut pmcts, &pnet, temp, i as u64);
      let mut player2 = player(&mut nmcts, &net, temp, i as u64);
      let game_result = play_game(&mut player1, &mut player2, i);
      match game_result {
        1 => a += 1,
        -1 => b += 1,
        _ => c += 1
      }
    }
    tx.send((a, b, c)).unwrap();
  });

  let each_mcts = MCTS::duplicate(mcts);
  thread::spawn(move || {
    let mut a = 0;
    let mut b = 0;
    let mut c = 0;
    let mut net = NNet::new(board_size, action_size, num_channels);
    let mut pnet = NNet::new(board_size, action_size, num_channels);
    net.load("temp/trained.pt");
    pnet.load("temp/best.pt");
    let mut pmcts = MCTS::duplicate(&each_mcts);
    let mut nmcts = MCTS::duplicate(&each_mcts);
    for i in 0..num {
      pmcts = MCTS::extend(pmcts); // reset search tree
      nmcts = MCTS::extend(nmcts); // reset search tree
      let mut player1 = player(&mut pmcts, &pnet, temp, i as u64 + 123445);
      let mut player2 = player(&mut nmcts, &net, temp, i as u64 + 123445);
      let game_result = play_game(&mut player2, &mut player1, i);
      match game_result {
        -1 => a += 1,
        1 => b += 1,
        _ => c += 1
      }
    }
    tx2.send((a, b, c)).unwrap();
  });
  for results in rx {
    let (a, b, c) = results;
    println!("Arena Results {:?} {:?} {:?}", a, b, c);
    one_won += a;
    two_won += b;
    draws += c;
  }
  (one_won, two_won, draws)
}
impl Coach {
  pub fn learn(&mut self) {
    let board_size = self.board_size;
    let action_size = self.action_size;
    let num_channels = self.num_channels;
    let train_examples: Vec<Example> = Vec::new();
    let mut ex_arc_mut = Arc::new(Mutex::new(train_examples));
    let mut sp_ex = Arc::clone(&ex_arc_mut);
    let mut tn_ex = Arc::clone(&ex_arc_mut);
    let mcts_sim_num = 600;
    println!("self playing... warming up");
    {
      let mut root_mcts = MCTS::new(mcts_sim_num, 1.0);
      self_play_sim(&mut ex_arc_mut, board_size, action_size, num_channels, 5, 4, &mut root_mcts);
    }
    let self_play_handle = thread::spawn(move || {
      for i in 0..10000 {
        println!("self playing... round:{}", i);
        let mcts_sim_num = 600 + i * 3;
        let mut root_mcts = MCTS::new(mcts_sim_num, 1.0);
        self_play_sim(&mut sp_ex, board_size, action_size, num_channels, 10, 8, &mut root_mcts);
      }
    });
    
    let train_net_handle = thread::spawn(move || {
      for i in 0..10000 {
        println!("start training... round:{}", i);
        let mut lr = 0.0005 - 0.000002 * i as f64;
        if lr < 0.0001 {
          lr = 0.0001;
        }
        let mcts_sim_num: u32 = 300 + i * 2;
        let mut train_mcts = MCTS::new(mcts_sim_num, 1.0);
        train_net(&mut tn_ex, board_size, action_size, num_channels, lr);
        arena(mcts_sim_num, board_size, action_size, num_channels, &mut train_mcts);
      }
    });
    self_play_handle.join().unwrap();
    train_net_handle.join().unwrap();
    println!("learning end");
  }
}