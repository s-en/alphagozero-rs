use super::*;

use rand::distributions::WeightedIndex;
use std::collections::VecDeque;
use indicatif::ProgressIterator;
use std::thread;
use std::sync::mpsc;
use std::sync::{Mutex, Arc};
use rand::prelude::*;
use std::time::SystemTime;
use std::time::Instant;

extern crate savefile;

fn predict<'a>(net: &'a NNet) -> Box<dyn Fn(Vec<f32>) -> (Vec<f32>, f32) + 'a> {
  Box::new(move |board| -> (Vec<f32>, f32) {
    NNet::predict(net, board)
  })
}
fn predict32<'a>(net: &'a NNet) -> Box<dyn Fn(Vec<Vec<f32>>) -> Vec<(Vec<f32>, f32)> + 'a> {
  Box::new(move |board| -> Vec<(Vec<f32>, f32)> {
    NNet::predict32(net, board)
  })
}

fn pretrain(board_size: i64, action_size: i64, num_channels: i64) {
  // train v
  let mut rng = rand::thread_rng();
  let mut net = NNet::new(board_size, action_size, num_channels);
  net.load_trainable("temp/trained.pt");
  let mut examples: Vec<Example> = Vec::new();
  let max = 1000;
  for i in 0..max {
    let mut board = Board::new(BoardSize::S5);
    // let tb = Stones::new32(rng.gen_range(0, 1 << (action_size - 1)) as u32);
    // let tw = Stones::new32(rng.gen_range(0, 1 << (action_size - 1)) as u32);
    
    // board.action(action_size as u32, Turn::Black);
    // board.action(action_size as u32, Turn::White);
    // board.action(action_size as u32, Turn::Black);
    
    let mut v: f32 = 0.0 ;//board.count_diff() as f32 / action_size as f32 * 10.0;
    // let mut v = 0.0;
    // if board.count_diff() > 0 { v = 1.0; }
    // if board.count_diff() < 0 { v = -1.0; }
    let mut pi = vec![0.0; action_size as usize];
    if i < max / 2 {
      let tb = Stones::new32(0b11111_11111_11111_11111_11111);
      let tw = Stones::new32(0b00000_00000_00000_00000_00000);
      board.set_stones(Turn::Black, tb);
      board.set_stones(Turn::White, tw);
      v = 1.0;
      pi[0] = 1.0;
    } else {
      let tb = Stones::new32(0b00000_00000_00000_00000_00000);
      let tw = Stones::new32(0b11111_11111_11111_11111_11111);
      board.set_stones(Turn::Black, tb);
      board.set_stones(Turn::White, tw);
      v = -1.0;
      pi[1] = 1.0;
    }
    board.action(action_size as u32, Turn::Black);
    board.action(action_size as u32, Turn::Black);
    board.action(action_size as u32, Turn::Black);
    board.turn = Turn::Black;
    if i % 2 == 0 {
      board.turn = Turn::White;
    }
    
    let ex = Example {
      board: board.input(),
      pi: pi,
      v: v
    };
    // println!("v {:?}", v);
    examples.push(ex);
  }
  let ex = examples.iter().map(|x| x).collect();
  // // println!("ex {:?}", ex);
  net.train(ex);
  net.save("temp/trained.pt");

  let mut board = Board::new(BoardSize::S5);
  // let tb = Stones::new32(0b00000_00000_00000_00000_00000);
  // let tw = Stones::new32(0b11111_11111_11111_11111_11111);
  // board.set_stones(Turn::Black, tb);
  // board.set_stones(Turn::White, tw);
  // board.action(action_size as u32, Turn::Black);
  // board.action(action_size as u32, Turn::Black);
  // board.action(action_size as u32, Turn::Black);
  for i in 1..20 {
    board.action(i, Turn::Black);
  }
  // board.turn = Turn::White;
  println!("{:?}", board.input());
  let pi = NNet::predict(&net, board.input());
  println!("predict {:?}", pi);
}
fn self_play(ex_arc_mut: &mut Arc<Mutex<Vec<Example>>>, board_size: i64, action_size: i64, num_channels: i64, seed: u64) {
  let num_eps = 20;
  let maxlen_of_queue = 100000;
  let max_history_queue = 40000;
  let mut rng = rand::thread_rng();
  let mut iteration_train_examples = Examples {
    values: VecDeque::with_capacity(maxlen_of_queue)
  };
  let mut black_win_history = Examples {
    values: VecDeque::with_capacity(max_history_queue)
  };
  let mut white_win_history = Examples {
    values: VecDeque::with_capacity(max_history_queue)
  };
  println!("self playing...");
  for _ in 0..num_eps {
    let (tx, rx) = mpsc::channel();
    for ne in 0..5 {
      let tx1 = mpsc::Sender::clone(&tx);
      thread::spawn(move || {
        let mut tnet = NNet::new(board_size, action_size, num_channels);
        tnet.load("temp/best.pt");
        let mut mcts = MCTS::new(200, 2.0); // reset search tree
        let rng = &mut rand::thread_rng();
        let examples = execute_episode(rng, &mut mcts, &tnet);
        tx1.send(examples).unwrap();
      });
    }
    drop(tx);
    for received in rx {
      iteration_train_examples.values.extend(received);
    }
  }

  // save the iteration examples to the history
  for ex in iteration_train_examples.values {
    if ex.board[0] == 1.0 {
      black_win_history.values.push_back(ex);
    } else {
      white_win_history.values.push_back(ex);
    }
  }
  // backup history to a file
  // NB! the examples were collected using the model from the previous iteration, so (i-1)
  // save_file(&format!("temp/train_examples_{}.bin", i - 1), 0, &train_examples_history).unwrap();
  if black_win_history.values.len() > max_history_queue {
    let cut = black_win_history.values.len() - max_history_queue;
    black_win_history.values.drain(..cut);
  }
  if white_win_history.values.len() > max_history_queue {
    let cut = white_win_history.values.len() - max_history_queue;
    white_win_history.values.drain(..cut);
  }
  println!("example black {}, white {}", black_win_history.values.len(), white_win_history.values.len());

  // shuffle examples before training
  let train_examples = &mut *ex_arc_mut.lock().unwrap();
  for e in black_win_history.values {
    train_examples.push(e);
  }
  for e in white_win_history.values {
    train_examples.push(e);
  }
  let vs = train_examples.iter().map(|x| x.v).collect::<Vec<f32>>();
  //println!("{:?}", vs);
  println!("won {:?}", vs.iter().filter(|&x| x > &0.0).sum::<f32>());
  println!("lose {:?}", vs.iter().filter(|&x| x < &0.0).sum::<f32>());
  let tlen = train_examples.len();
  if tlen > maxlen_of_queue {
    let cut = tlen - maxlen_of_queue;
    train_examples.drain(..cut);
  }
  train_examples.shuffle(&mut rng);
}
fn execute_episode(rng: &mut ThreadRng, mcts: &mut MCTS, net: &NNet) -> Vec<Example> {
  let start = Instant::now();
  let mut examples = Vec::new();
  let mut board = Board::new(BoardSize::S5);
  let mut episode_step = 0;
  let temp_threshold = 5;
  let mut kcnt = 0;
  loop {
    episode_step += 1;
    let mut temp = 0.5;
    if episode_step < temp_threshold {
      temp = 3.0;
    }else if episode_step < temp_threshold*2 {
      temp = 1.0;
    }
    //println!("step {:?} turn {:?}", board.step, board.turn as i32);
    // println!("{}", board);
    let mut pi = mcts.get_action_prob(&board, temp, &predict32(net));
    //mcts.get_win_rate(&board);
    // if episode_step == 1 {
    //   println!("pi {:?}", pi);
    // }
    // if episode_step == 1 {
    //   // 初手天元に固定
    //   pi[12] = 1.0;
    // }
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
    //println!("action {}", action);
    board.action(action, board.turn);

    let r = board.game_ended();

    if r != 0 {
      println!("{} {}", r, board.get_kifu_sgf());
      //println!("");
      let v = r as i32;
      for mut ex in &mut examples {
        // v: 1.0 black won
        // v: -1.0 white won
         ex.v = v as f32;
      }
      //println!("{:?}", examples);
      // let end = start.elapsed();
      // println!("execute_episode {}.{:03}秒", end.as_secs(), end.subsec_nanos() / 1_000_000);
      return examples;
    }
  }
}
fn train_net(ex_arc_mut: &mut Arc<Mutex<Vec<Example>>>, board_size: i64, action_size: i64, num_channels: i64) {
  let mut net = NNet::new(board_size, action_size, num_channels);
  let board = Board::new(BoardSize::S5);
  net.load_trainable("temp/trained.pt");
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
  net.train(ex);
  net.save("temp/trained.pt");

  let pi = NNet::predict(&net, board.input());
  println!("after {:?}", pi);
}
fn arena(board_size: i64, action_size: i64, num_channels: i64) {
  let update_threshold = 0.54;
  
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
    net.load("temp/best.pt");
    net.save("temp/trained.pt");
  } else {
    println!("ACCEPTING NEW MODEL");
    net.load_trainable("temp/trained.pt");
    net.save("temp/best.pt");
  }
}
fn player<'a>(_mcts: &'a mut MCTS, _net: &'a NNet, temp: f32, seed: u64) -> Box<dyn FnMut(&mut Board) -> usize + 'a> {
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
      let probs = _mcts.get_action_prob(&x, temp, &predict32(_net));
      // println!("{:?}", probs);
      // mcts::max_idx(&probs)
      let dist = WeightedIndex::new(&probs).unwrap();
      dist.sample(&mut rng)
    }
  })
}
pub fn play_game<F: FnMut(&mut Board) -> usize>(player1: &mut F, player2: & mut F, rep: u32) -> i8 {
  let players = [player2, player1];
  let mut cur_player = 1;
  let mut board = Board::new(BoardSize::S5);
  let mut it = 0;
  while board.game_ended() == 0 {
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
    // if rep == 0 {
    //   println!("Turn {} Player {} action {} {}", it, board.turn as i32, action % 5 + 1, action / 5 + 1);
    //   println!("{}", board);
    // }
    board.action(action as u32, board.turn);
    // println!("Turn {} Player {} action {} {}", it, board.turn as i32, action % 5 + 1, action / 5 + 1);
    cur_player = (board.turn as i32 + 1) as usize / 2;
  }
  if rep == 0 {
    println!("{} {}", board.game_ended() ,board.get_kifu_sgf());
    println!("");
  }
  board.game_ended()
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
      let mut pmcts = MCTS::new(100, 0.5);
      let mut nmcts = MCTS::new(100, 0.5);
      let mut player1 = player(&mut pmcts, &pnet, temp, i as u64);
      let mut player2 = player(&mut nmcts, &net, temp, i as u64);
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
      let mut pmcts = MCTS::new(100, 0.5);
      let mut nmcts = MCTS::new(100, 0.5);
      let mut player1 = player(&mut pmcts, &pnet, temp, i as u64 + 123445);
      let mut player2 = player(&mut nmcts, &net, temp, i as u64 + 123445);
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
    let ex_arc_mut = Arc::new(Mutex::new(train_examples));
    let mut sp_ex = Arc::clone(&ex_arc_mut);
    pretrain(board_size, action_size, num_channels);
    // self_play(&mut sp_ex, board_size, action_size, num_channels, 0);
    // let self_play_handle = thread::spawn(move || {
    //   for i in 0..1000 {
    //     println!("start self play {}", i);
    //     self_play(&mut sp_ex, board_size, action_size, num_channels, i+1);
    //     println!("end self play {}", i);
    //   }
    // });
    // let mut tn_ex = Arc::clone(&ex_arc_mut);
    // let train_net_handle = thread::spawn(move || {
    //   for i in 0..1000 {
    //     println!("start train {}", i);
    //     train_net(&mut tn_ex, board_size, action_size, num_channels);
    //     println!("end train {}", i);
    //     println!("start arena {}", i);
    //     arena(board_size, action_size, num_channels);
    //     println!("end arena {}", i);
    //   }
    // });
    // println!("before join");
    // self_play_handle.join().unwrap();
    // train_net_handle.join().unwrap();
    // println!("learning end");
  }
}