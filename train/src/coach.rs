use super::*;

use rand::distributions::WeightedIndex;
use std::collections::VecDeque;
use std::thread;
use std::sync::mpsc;
use std::sync::{Mutex, Arc};
use std::time::Instant;
use std::cmp;

extern crate savefile;

fn self_play_sim(arc_examples: &mut Arc<Mutex<Vec<Example>>>, board_size: i64, action_size: i64, num_channels: i64, num_eps: i32, sim_num: i32) {
  let maxlen_of_queue = 100000;
  let mut rng = rand::thread_rng();
  let (tx, rx) = mpsc::channel();
  for _ in 0..(sim_num-1) {
    let tx1 = mpsc::Sender::clone(&tx);
    thread::spawn(move || {
      let examples = self_play(board_size, action_size, num_channels, num_eps);
      tx1.send(examples).unwrap();
    });
  }
  thread::spawn(move || {
    let examples = self_play(board_size, action_size, num_channels, num_eps);
    tx.send(examples).unwrap();
  });
  {
    let mut results = vec![];
    for examples in rx {
      results.extend(examples);
    }
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

fn self_play(board_size: i64, action_size: i64, num_channels: i64, num_eps: i32) -> Vec<Example> {
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
  for i in 0..num_eps {
    let mut mcts = MCTS::new(200, 3.0); // reset search tree
    let ep_results = execute_episode(rng, &mut mcts, &tnet, i);
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
  let start = Instant::now();
  let mut examples = Vec::new();
  let mut board = Board::new(BoardSize::S5);
  let mut episode_step = 0;
  let temp_threshold = 5;
  let mut kcnt = 0;
  let predict32 = |inputs: Vec<Vec<f32>>| {
    NNet::predict32(net, inputs)
  };
  let for_train = eps_cnt % 2 == 0;
  loop {
    episode_step += 1;
    let mut temp = 0.2;
    if episode_step < temp_threshold {
      temp = 1.0;
    }else if episode_step < temp_threshold*2 {
      temp = 0.5;
    }
    //println!("step {:?} turn {:?}", board.step, board.turn as i32);
    let mut pi = mcts.get_action_prob(&board, temp, &predict32, for_train);
    // println!("{}", board);
    //println!("pi {:?}", pi);
    // println!("wi {:?}", WeightedIndex::new(&pi));
    
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

    let r = board.game_ended(true);

    if r != 0 {
      // println!("{} {}", r, board.get_kifu_sgf());
      // std::process::exit(0x0100);
      //println!("");
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
  let board = Board::new(BoardSize::S5);
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
fn arena(board_size: i64, action_size: i64, num_channels: i64) {
  let update_threshold = 0.52;
  
  // training new network, keeping a copy of the old one
  println!("PITTING AGAINST PREVIOUS VERSION");
  let game_result;
  {
    game_result = play_games(80, board_size, action_size, num_channels);
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
fn player<'a>(_mcts: &'a mut MCTS, _net: NNet, temp: f32, count: u64) -> Box<dyn FnMut(&mut Board) -> usize + 'a> {
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
        NNet::predict32(&_net, inputs)
      };
      let for_train = false;
      let probs = _mcts.get_action_prob(&x, temp, &predict32, for_train);
      // println!("{:?}", probs);
      // mcts::max_idx(&probs)
      let dist = WeightedIndex::new(&probs).unwrap();
      dist.sample(&mut rng)
    }
  })
}
pub fn play_game<F: FnMut(&mut Board) -> usize>(player1: &mut F, player2: & mut F, count: u32) -> i8 {
  let players = [player2, player1];
  let mut cur_player = 1;
  let mut board = Board::new(BoardSize::S5);
  let mut it = 0;
  while board.game_ended(false) == 0 {
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
    // println!("{}", board.valid_moves(board.turn));
    // if count == 0 {
    //   println!("Turn {} Player {} action {} {}", it, board.turn as i32, action % 5 + 1, action / 5 + 1);
    //   println!("{}", board);
    // }
    board.action(action as u32, board.turn);
    // println!("Turn {} Player {} action {} {}", it, board.turn as i32, action % 5 + 1, action / 5 + 1);
    cur_player = (board.turn as i32 + 1) as usize / 2;
  }
  if count == 0 {
    println!("{} {}", board.game_ended(false) ,board.get_kifu_sgf());
    println!("");
  }
  board.game_ended(false)
}
pub fn play_games(num: u32, board_size: i64, action_size: i64, num_channels: i64) -> (u32, u32, u32) {
  let num = num / 2;
  let mut one_won = 0;
  let mut two_won = 0;
  let mut draws = 0;

  println!("Arena.playGames (1)");
  let (tx, rx) = mpsc::channel();
  let temp = 0.5;
  for i in 0..num {
    let tx1 = mpsc::Sender::clone(&tx);
    thread::spawn(move || {
      let mut net = NNet::new(board_size, action_size, num_channels);
      let mut pnet = NNet::new(board_size, action_size, num_channels);
      net.load("temp/trained.pt");
      pnet.load("temp/best.pt");
      let mut pmcts = MCTS::new(100, 1.0);
      let mut nmcts = MCTS::new(100, 1.0);
      let mut player1 = player(&mut pmcts, pnet, temp, i as u64);
      let mut player2 = player(&mut nmcts, net, temp, i as u64);
      let game_result = play_game(&mut player1, &mut player2, i);
      tx1.send(game_result).unwrap();
    });
  }
  drop(tx);
  for game_result in rx {
    match game_result {
      1 => one_won += 1,
      -1 => two_won += 1,
      _ => draws += 1
    }
  }

  println!("Arena.playGames (2)");
  let (tx2, rx2) = mpsc::channel();
  for i in 0..num {
    let tx1 = mpsc::Sender::clone(&tx2);
    thread::spawn(move || {
      let mut net = NNet::new(board_size, action_size, num_channels);
      let mut pnet = NNet::new(board_size, action_size, num_channels);
      net.load("temp/trained.pt");
      pnet.load("temp/best.pt");
      let mut pmcts = MCTS::new(100, 1.0);
      let mut nmcts = MCTS::new(100, 1.0);
      let mut player1 = player(&mut pmcts, pnet, temp, i as u64 + 123445);
      let mut player2 = player(&mut nmcts, net, temp, i as u64 + 123445);
      let game_result = play_game(&mut player2, &mut player1, i);
      tx1.send(game_result).unwrap();
    });
  }
  drop(tx2);
  for game_result in rx2 {
    match game_result {
      -1 => one_won += 1,
      1 => two_won += 1,
      _ => draws += 1
    }
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
    println!("self playing...");
    {
      self_play_sim(&mut ex_arc_mut, board_size, action_size, num_channels, 10, 12);
      println!("self play done we got {:?} examples", sp_ex.lock().unwrap().len());
    }

    let self_play_handle = thread::spawn(move || {
      for i in 0..1000 {
        println!("self playing... round:{}", i);
        self_play_sim(&mut sp_ex, board_size, action_size, num_channels, 10, 8);
        println!("self play done round:{}", i);
      }
    });
    let train_net_handle = thread::spawn(move || {
      for i in 0..10000 {
        println!("start training... {}", i);
        let mut lr = 0.0005;
        if i > 20 {
          lr = 0.0002;
        }
        train_net(&mut tn_ex, board_size, action_size, num_channels, lr);
        println!("start arena {}", i);
        arena(board_size, action_size, num_channels);
      }
    });
    self_play_handle.join().unwrap();
    train_net_handle.join().unwrap();
    println!("learning end");
  }
}