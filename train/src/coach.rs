use super::*;

use rand::distributions::WeightedIndex;
use std::collections::VecDeque;
use std::thread;
use std::sync::mpsc;
use std::sync::{Mutex, Arc};
use std::time;
use std::cmp::Ordering;
use std::process;

extern crate savefile;

const KOMI: i32 = 0;
const BOARD_SIZE: BoardSize = BoardSize::S7;
const TRAINED_MODEL: &str = "7x7/trained";
const BEST_MODEL: &str = "7x7/dualnet7x7_8conv.pt";//"7x7/dualnet7x7_4conv_dep16.pt";
const MAX_EXAMPLES: usize = 2000000;
const FOR_TRAIN: bool = false;

fn self_play_sim(
  arc_examples: &mut Arc<Mutex<Vec<Example>>>, 
  board_size: i64, 
  action_size: i64, 
  num_channels: i64, 
  num_eps: i32, 
  sim_num: i32, 
  mcts: &mut MCTS) {
  let mut rng = rand::thread_rng();
  let (tx, rx) = mpsc::channel();
  for i in 0..sim_num {
    let tx1 = mpsc::Sender::clone(&tx);
    let mut each_mcts = MCTS::duplicate(&mcts);
    thread::spawn(move || {
      thread::sleep(time::Duration::from_millis(i as u64 * 500));
      println!("self_play_sim {:?}", i);
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
    if tlen > MAX_EXAMPLES {
      let cut = tlen - MAX_EXAMPLES;
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
  let max_history_queue = MAX_EXAMPLES/10;
  let rng = &mut rand::thread_rng();
  let mut black_win_history = Examples {
    values: VecDeque::with_capacity(max_history_queue)
  };
  let mut white_win_history = Examples {
    values: VecDeque::with_capacity(max_history_queue)
  };
  let mut tnet = NNet::new(board_size, action_size, num_channels);
  tnet.load(BEST_MODEL);
  let mut each_mcts = MCTS::duplicate(mcts);
  for i in 0..num_eps {
    let ep_results = execute_episode(rng, &mut each_mcts, &tnet, i);
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
    each_mcts = MCTS::duplicate(&each_mcts);
  }
  // ??????????????????????????????
  // let bw_min = cmp::min(cmp::min(black_win_history.values.len(), white_win_history.values.len()), max_history_queue);
  // if black_win_history.values.len() > bw_min {
  //   let cut = black_win_history.values.len() - bw_min;
  //   black_win_history.values.drain(..cut);
  // }
  // if white_win_history.values.len() > bw_min {
  //   let cut = white_win_history.values.len() - bw_min;
  //   white_win_history.values.drain(..cut);
  // }
  // ?????????????????????????????????
  // let bw_min = cmp::min(cmp::min(black_win_history.values.len(), 10), max_history_queue);
  // if white_win_history.values.len() > bw_min {
  //   let cut = white_win_history.values.len() - 10;
  //   white_win_history.values.drain(..cut);
  // }
  println!("example black {}, white {}", black_win_history.values.len(), white_win_history.values.len());

  let mut results: Vec<Example> = Vec::from(black_win_history.values);
  results.append(&mut Vec::from(white_win_history.values));
  results
}
fn execute_episode(rng: &mut ThreadRng, mcts: &mut MCTS, net: &NNet, eps_cnt: i32) -> (Vec<Example>, i8) {
  //let start = Instant::now();
  let mut examples = Vec::new();
  let mut board = Board::new(BOARD_SIZE);
  let mut episode_step = 0;
  let predict32 = |inputs: Vec<Vec<f32>>| {
    NNet::predict32(net, inputs)
  };
  let for_train = eps_cnt%2 != 0;//FOR_TRAIN;
  let prioritize_kill = false;
  let self_play = false;
  let stemp = mcts.sim_num;
  loop {
    episode_step += 1;
    let mut temp = 1.0;
    // if eps_cnt%2 == episode_step%2 {
    //   // ??????????????????????????????
    //   mcts.sim_num = stemp * 5;
    // } else {
    //   mcts.sim_num = stemp;
    // }
    let mstep = BOARD_SIZE as i32 * 3;
    if episode_step == 1 {
      temp = 3.0;
    } else if episode_step <= mstep {
      temp = (1.0 - (episode_step as f32 / mstep as f32)) * 2.0 + 0.1;
    } else {
      temp = 0.1;
    }
    // println!("step {:?} turn {:?}", board.step, board.turn as i32);
    // let mstart = Instant::now();
    let turn = board.turn;
    let mut pi: Vec<f32>;
    // if episode_step == 1 && eps_cnt % 1 == 0 {
    //   // ??????????????????
    //   pi = vec![0.0; 50];
    //   // pi[24] = 1.0;
    //   // ??????????????????
    //   let random_int: usize = rng.gen_range(0, BOARD_SIZE as usize * BOARD_SIZE as usize);
    //   pi[random_int] = 1.0;
    // } else if rng.gen_range(0,50) == 0 {
    //   // ??????????????????????????????
    //   pi = board.vec_valid_moves_for_search(board.turn).iter().map(|&m| m as i32 as f32).collect();
    // } else {
      pi = mcts.get_action_prob(&board, temp, &predict32, prioritize_kill, for_train, self_play, KOMI);
    // }

    //let mut pi = mcts.get_action_prob(&board, temp, &predict32, prioritize_kill, for_train, self_play, KOMI);
    // let mend = mstart.elapsed();
    // println!("get_action_prob {}.{:03}???", mend.as_secs(), mend.subsec_nanos() / 1_000_000);

    let dist = WeightedIndex::new(&pi).unwrap();
    let mut action = dist.sample(rng) as u32;
    // if action == 49 {
    //   println!("{}: {}", eps_cnt, board.get_kifu_sgf());
    //   println!("{:?}", board.input());
    // }

    // ????????????????????????????????????
      // let s = board.calc_hash();
      // let sa = (s, action as usize);
      // let bs = BOARD_SIZE as u32;
      // if mcts.qsa.contains_key(&sa) {
      //   if turn == Turn::Black && mcts.qsa[&sa] > 0.99 {
      //     action = bs * bs;
      //   }
      //   if turn == Turn::White && mcts.qsa[&sa] < -0.99 {
      //     action = bs * bs;
      //   }
      // }
    // if action == 49 {
    //   // pass
    //   println!("probs {:?}", pi);
    // }

    board.action(action, board.turn);

    let r = board.game_ended(false, KOMI);
    if board.pass_cnt >= 2 {
      board.pass_cnt = 1;
    }

    let sym = board.symmetries(pi);
    for (b, p) in sym {
      let ex = Example {
        board: b,
        pi: p,
        v: 0.0
      };
      examples.push(ex);
    }

    if r != 0 {
      //print!("{} ", r);
      // println!("{} {}", r, board.get_kifu_sgf());
      // std::process::exit(0x0100);
      // if eps_cnt == 0 {
      //   println!("{}: {} {}", eps_cnt, r, board.get_kifu_sgf());
      //   // println!("{:?}", board.input());
      // }
      let mut v = r as i32;
      for mut ex in &mut examples {
        // v: 1.0 black won
        // v: -1.0 white won
         ex.v = v as f32;
      }
      mcts.sim_num = stemp;
      // println!("{:?}", examples);
      // let end = start.elapsed();
      // println!("execute_episode {}.{:03}???", end.as_secs(), end.subsec_nanos() / 1_000_000);
      return (examples, r);
    }
  }
}
fn nakade_data() -> Vec<Example> {
  let mut examples = Vec::new();
  let mut add_example = |board: &Board, pi:Vec<f32>, v: f32| {
    // println!("---- v: {:?} turn: {:?} ----", v, board.turn as i32);
    // println!("pi {:?}", pi);
    // println!("{:}", board);
    let sym = board.symmetries(pi);
    for (b, p) in sym {
      let ex = Example {
        board: b,
        pi: p,
        v: v
      };
      examples.push(ex);
    }
  };
  // ????????????
  for teban in [-1, 1] {
    for color in [-1, 1] {
      for x in 0..3 {
        for y in 0..4 {
          let mut board = Board::new(BoardSize::S7);
          let mut s1: u64 = 0b1111111_1111111_1111111_1111111_1111111_1111111_1111111;
          s1 = s1 ^ (1 << (x+y*7));
          s1 = s1 ^ (1 << (x+1+y*7));
          s1 = s1 ^ (1 << (x+2+y*7));
          let tb;
          let tw;
          if color == -1 {
            tb = Stones::new64(0);
            tw = Stones::new64(s1);
          } else {
            tb = Stones::new64(s1);
            tw = Stones::new64(0);
          }
          board.set_stones(Turn::Black, tb);
          board.set_stones(Turn::White, tw);
          for _ in 0..6 {
            board.action(49, board.turn);
          }
          board.pass_cnt = 0;
          board.turn = if teban == -1 {Turn::White} else {Turn::Black};
          //println!("{:}", board);
          let action = x + 1 + y*7;
          //println!("action {:?}", action);
          let mut pi: Vec<f32> = vec![0.0; 50];
          pi[action] = 1.0;
          // ?????????????????????????????????
          add_example(&board, pi.clone(), teban as f32);
          board.turn = if teban == -1 {Turn::Black} else {Turn::White};
          add_example(&board, pi, 0.0);
          board.turn = if teban == -1 {Turn::White} else {Turn::Black};
          // ?????????????????????
          board.action(action as u32, board.turn);
          board.action(49, board.turn);
          pi = vec![0.0; 50];
          if color == teban {
            // ?????????????????????
            pi[49] = 1.0;
            add_example(&board, pi, teban as f32);
          } else {
            // ??????????????????????????????
            pi[action-1] = 0.5;
            pi[action+1] = 0.5;
            add_example(&board, pi, 0.0);
          }
          // ??????????????????????????????
          board.turn = if teban == -1 {Turn::Black} else {Turn::White};
          pi = vec![0.0; 50];
          pi[49] = 1.0; // ??????????????????????????????????????????
          add_example(&board, pi, teban as f32);
          // ??????????????????????????????
          board.turn = if teban == -1 {Turn::White} else {Turn::Black};
          board.action(action as u32 - 1, board.turn);
          pi = vec![0.0; 50];
          pi[action+1] = 1.0;
          add_example(&board, pi, 0.0);
        }
      }
    }
  }
  // ????????????

  println!("examples {:?}", examples.len());
  return examples;
}
fn pass_data() -> Vec<Example> {
  let mut examples = Vec::new();
  let mut add_example = |board: &Board, pi:Vec<f32>, v: f32| {
    println!("---- pass: {:?} turn: {:?} = v: {:?}  ----", board.pass_cnt, board.turn as i32, v);
    // println!("{:?}", &board.input()[49*12-1]);
    let sym = board.symmetries(pi);
    for (b, p) in sym {
      let ex = Example {
        board: b,
        pi: p,
        v: v
      };
      examples.push(ex);
    }
  };
  let mut rng = rand::thread_rng(); // ??????????????????????????????????????????????????????
  for t in 0..2 {
    for p in 0..2 {
      let mut board = Board::new(BoardSize::S7);
      if t == 0 {
        board.turn = Turn::White;
      }
      board.pass_cnt = p;
      let mut pi = vec![0.0; 50];
      let mut v = 0.0;
      // xor
      // 0 0 = 0
      if board.pass_cnt == 0 && board.turn == Turn::White {
        v = 0.0;
      }
      // 0 1 = 1
      if board.pass_cnt == 0 && board.turn == Turn::Black {
        v = 1.0;
      }
      // 1 0
      if board.pass_cnt == 1 && board.turn == Turn::White {
        v = 1.0;
      }
      // 1 1 = 0
      if board.pass_cnt == 1 && board.turn == Turn::Black {
        v = 0.0;
      }
      add_example(&board, pi, v);
    }
  }
  println!("examples {:?}", examples.len());
  return examples;
}
fn train_net(ex_arc_mut: &mut Arc<Mutex<Vec<Example>>>, board_size: i64, action_size: i64, num_channels: i64, lr: f64) {
  let mut net = NNet::new(board_size, action_size, num_channels);
  let mut board = Board::new(BOARD_SIZE);
  net.load_trainable(BEST_MODEL);
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
  match net.train(ex, lr) {
    Ok(_) => {},
    Err(e) => panic!("{:?}", e)
  }
  net.save(String::from(TRAINED_MODEL) + &process::id().to_string() + ".pt");

  let pi = NNet::predict(&net, board.input());
  println!("after black pass0 1 0 = {:?}", pi.1);
  board.turn = Turn::White;
  let pi = NNet::predict(&net, board.input());
  println!("after white pass0 0 0 = {:?}", pi.1);
  board.turn = Turn::Black;
  board.action(49, board.turn);
  board.turn = Turn::Black;
  let pi = NNet::predict(&net, board.input());
  println!("after black pass1 1 1 = {:?}", pi.1);
  board.turn = Turn::White;
  let pi = NNet::predict(&net, board.input());
  println!("after white pass1 0 1 = {:?}", pi.1);
}
fn arena(mcts_sim_num: u32, 
  board_size: i64, 
  action_size: i64, 
  num_channels: i64,
  mcts: &mut MCTS) {
  let update_threshold = 0.56;
  
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
    net.load_trainable(String::from(TRAINED_MODEL) + &process::id().to_string() + ".pt");
    net.save(BEST_MODEL);
  }
}
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
fn player<'a>(_mcts: &'a mut MCTS, _net: &'a NNet, temp: f32) -> Box<dyn FnMut(&mut Board, u64) -> usize + 'a> {
  let mut rng = rand::thread_rng();
  Box::new(move |x: &mut Board, count: u64| -> usize {
    let predict32 = |inputs: Vec<Vec<f32>>| {
      NNet::predict32(_net, inputs)
    };
    let mut for_train = FOR_TRAIN;
    let self_play = false;
    let prioritize_kill = false;
    // if count % 10 < 7 {
    //   for_train = true;
    // }
    let mut t = temp;
    let turn = x.turn;
    let probs = _mcts.get_action_prob(&x, t, &predict32, prioritize_kill, for_train, self_play, KOMI);
    let mut action: usize;
    if temp == 0.0 {
      action = max_idx(&probs);
    } else {
      let dist = WeightedIndex::new(&probs).unwrap();
      action = dist.sample(&mut rng);
    }

    // ????????????????????????????????????
    // let s = x.calc_hash();
    // let sa = (s, action as usize);
    // let bs = BOARD_SIZE as usize;
    // if _mcts.qsa.contains_key(&sa) {
    //   if turn == Turn::Black && _mcts.qsa[&sa] > 0.99 {
    //     action = bs * bs;
    //   }
    //   if turn == Turn::White && _mcts.qsa[&sa] < -0.99 {
    //     action = bs * bs;
    //   }
    // }
    action
  })
}
pub fn play_game<F: FnMut(&mut Board, u64) -> usize>(player1: &mut F, player2: & mut F, episode: u32) -> i8 {
  let players = [player2, player1];
  let mut cur_player = 1;
  let mut board = Board::new(BOARD_SIZE);
  let mut count = 0;
  while board.game_ended(false, KOMI) == 0 {
    count += 1;
    let action = players[cur_player](&mut board, count);
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
  if episode <= 2 {
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

  let temp = 0.1;
  let mut pmcts = MCTS::duplicate(&mcts);
  let mut nmcts = MCTS::duplicate(&mcts);
  // white cpu
  let mut a = 0;
  let mut b = 0;
  let mut c = 0;
  let mut net = NNet::new(board_size, action_size, num_channels);
  let mut pnet = NNet::new(board_size, action_size, num_channels);
  net.load(String::from(TRAINED_MODEL) + &process::id().to_string() + ".pt");
  pnet.load(BEST_MODEL);
  
  for i in 0..num {
    pmcts = MCTS::duplicate(&pmcts);
    nmcts = MCTS::duplicate(&nmcts);
    let mut player1 = player(&mut pmcts, &pnet, temp);
    let mut player2 = player(&mut nmcts, &net, temp);
    let game_result = play_game(&mut player1, &mut player2, i);
    match game_result {
      1 => a += 1,
      -1 => b += 1,
      _ => c += 1
    };
    // if a > num * 2 / 3 {
    //   // skip
    //   return (80, two_won, draws)
    // }
  }
  println!("Arena Results WHITE CPU WIN {:?}", b);
  one_won += a;
  two_won += b;
  draws += c;

  let mut pmcts = MCTS::duplicate(&mcts);
  let mut nmcts = MCTS::duplicate(&mcts);
  let mut a = 0;
  let mut b = 0;
  let mut c = 0;
  let mut net = NNet::new(board_size, action_size, num_channels);
  let mut pnet = NNet::new(board_size, action_size, num_channels);
  net.load(String::from(TRAINED_MODEL) + &process::id().to_string() + ".pt");
  pnet.load(BEST_MODEL);
  for i in 0..num {
    pmcts = MCTS::duplicate(&pmcts);
    nmcts = MCTS::duplicate(&nmcts);
    let mut player1 = player(&mut pmcts, &pnet, temp);
    let mut player2 = player(&mut nmcts, &net, temp);
    let game_result = play_game(&mut player2, &mut player1, i);
    match game_result {
      -1 => a += 1,
      1 => b += 1,
      _ => c += 1
    };
    // if one_won + a > num * 2 * 2 / 3 {
    //   // skip
    //   return (80, two_won, draws)
    // }
  }
  println!("Arena Results BLACK CPU WIN {:?}", b);
  one_won += a;
  two_won += b;
  draws += c;
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
    let mcts_sim_num = 2;
    for i in 0..10000 {
      println!("self playing... round:{}", i);
      // {
      //   // ???????????????
      //   let tex = &mut *sp_ex.lock().unwrap();
      //   tex.extend(nakade_data());
      // }
      // let mut mcts_sim_num = 300 + i*5;
      // if mcts_sim_num > 1400 {
      //   mcts_sim_num = 1400;
      // }
      // {
      //   let tex = &mut *sp_ex.lock().unwrap();
      //   for _ in 0..100 {
      //     tex.extend(pass_data());
      //   }
      // }
      let mut mcts_sim_num = 200;
      let mut root_mcts = MCTS::new(mcts_sim_num, 1.0);
      self_play_sim(&mut sp_ex, board_size, action_size, num_channels, 14, 64, &mut root_mcts);

      println!("start training... round:{}", i);
      let lr = 8e-5;
      // let mut mcts_sim_num: u32 = 200 + i*3;
      // if mcts_sim_num > 300 {
      //   mcts_sim_num = 300;
      // }
      let mut mcts_sim_num: u32 = 200;
      let mut train_mcts = MCTS::new(mcts_sim_num, 1.0);
      train_net(&mut tn_ex, board_size, action_size, num_channels, lr);

      // let mut board = Board::new(BoardSize::S7);
      // let tb = Stones::new64(0b0000000_0000000_0000000_0000000_0000000_0000000_0000000);
      // let tw = Stones::new64(0b1111111_1111111_1111111_1111111_1111111_1111111_1111000);
      // board.set_stones(Turn::Black, tb);
      // board.set_stones(Turn::White, tw);
      // for _ in 0..6 {
      //   board.action(49, board.turn);
      // }
      // board.pass_cnt = 0;
      // println!("{:}", board);
      // let mut net = NNet::new(board_size, action_size, num_channels);
      // //net.load_trainable("7x7/best2.pt");
      // net.load_trainable(String::from(TRAINED_MODEL) + &process::id().to_string() + ".pt");
      // board.turn = Turn::Black;
      // let pi = NNet::predict(&net, board.input());
      // //println!("input {:?}", board.input());
      // println!("v {:?}", pi);
      // board.turn = Turn::White;
      // let pi = NNet::predict(&net, board.input());
      // println!("v {:?}", pi);

      // board.set_stones(Turn::Black, tw);
      // board.set_stones(Turn::White, tb);
      // for _ in 0..6 {
      //   board.action(49, board.turn);
      // }
      // board.pass_cnt = 0;
      // println!("{:}", board);
      // board.turn = Turn::Black;
      // let pi = NNet::predict(&net, board.input());
      // //println!("input {:?}", board.input());
      // println!("v {:?}", pi);
      // board.turn = Turn::White;
      // let pi = NNet::predict(&net, board.input());
      // println!("v {:?}", pi);



      // let mut board = Board::new(BoardSize::S7);
      // let tb = Stones::new64(0b0000000_0000000_0000000_0000000_0000000_0000000_0000000);
      // let tw = Stones::new64(0b1111111_1111111_1111111_1111111_1111111_1111111_1111010);
      // board.set_stones(Turn::Black, tb);
      // board.set_stones(Turn::White, tw);
      // for _ in 0..6 {
      //   board.action(49, board.turn);
      // }
      // board.pass_cnt = 0;
      // println!("{:}", board);
      // //net.load_trainable("7x7/best2.pt");
      // board.turn = Turn::Black;
      // let pi = NNet::predict(&net, board.input());
      // //println!("input {:?}", board.input());
      // println!("v {:?}", pi);
      // board.turn = Turn::White;
      // let pi = NNet::predict(&net, board.input());
      // println!("v {:?}", pi);

      // board.set_stones(Turn::Black, tw);
      // board.set_stones(Turn::White, tb);
      // for _ in 0..6 {
      //   board.action(49, board.turn);
      // }
      // board.pass_cnt = 0;
      // println!("{:}", board);
      // board.turn = Turn::Black;
      // let pi = NNet::predict(&net, board.input());
      // //println!("input {:?}", board.input());
      // println!("v {:?}", pi);
      // board.turn = Turn::White;
      // let pi = NNet::predict(&net, board.input());
      // println!("v {:?}", pi);

      // return;

      arena(mcts_sim_num, board_size, action_size, num_channels, &mut train_mcts);
    }
    println!("learning end");
  }
}