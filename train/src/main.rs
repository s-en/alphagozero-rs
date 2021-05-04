// extern crate tch;
use tch::{nn, kind, nn::Module, nn::OptimizerConfig, Device, Tensor};
use rand::prelude::*;
use rand::distributions::WeightedIndex;
use std::collections::VecDeque;
use indicatif::ProgressIterator;
use az_game::igo::*;

extern crate savefile;
#[macro_use]
extern crate savefile_derive;
use savefile::prelude::*;

pub mod nnet;

#[derive(Debug)]
pub struct NNet {
  board_size: i64,
  action_size: i64,
  num_channels: i64,
  vs: nn::VarStore,
  conv1: nn::Conv2D,
  conv2: nn::Conv2D,
  conv3: nn::Conv2D,
  conv4: nn::Conv2D,
  bn1: nn::BatchNorm,
  bn2: nn::BatchNorm,
  bn3: nn::BatchNorm,
  bn4: nn::BatchNorm,
  fc1: nn::Linear,
  fc_bn1: nn::BatchNorm,
  fc2: nn::Linear,
  fc_bn2: nn::BatchNorm,
  fc3: nn::Linear,
  fc4: nn::Linear,
}

#[derive(Savefile, Debug)]
pub struct Example {
  pub board: Vec<f32>,
  pub pi: Vec<f32>,
  pub v: f32
}

#[derive(Savefile, Debug)]
pub struct Examples {
  pub values: VecDeque<Example>
}

// fn main() {
//   println!("{}",tch::Cuda::is_available());
//   let t = Tensor::randn(&[26, 1], kind::FLOAT_CPU);
//   t.print();
//   let vs = nn::VarStore::new(Device::Cpu);
//   let board_size: i64 = 5;
//   let action_size: i64 = 26;
//   let num_channels: i64 = 32;
//   let net  = NNet::new(&vs.root(), board_size, action_size, num_channels);
//   let ex1 = Example {
//     board: nnet::randfloat(10, 25*2),
//     pi: nnet::randfloat(10, 26),
//     v: nnet::randfloat(1, 1)[0]
//   };
//   net.train(vec![&ex1]);
//   println!("ok");
// }

fn main() {
  println!("{}",tch::Cuda::is_available());
  learn();
}

fn predict<'a>(net: &'a NNet) -> Box<dyn Fn(Vec<f32>) -> (Vec<f32>, i8) + 'a> {
  Box::new(move |board| -> (Vec<f32>, i8) {
    NNet::predict(net, board)
  })
}

fn execute_episode(mcts: &mut MCTS, net: &NNet) -> Vec<Example> {
  let mut examples = Vec::new();
  let mut board = Board::new(BoardSize::S5);
  let mut episode_step = 0;
  let temp_threshold = 5 * 2;
  let mut rng = thread_rng();

  loop {
    episode_step += 1;
    let mut temp = 0.0;
    if episode_step < temp_threshold {
      temp = 1.0;
    }
    let pi = mcts.get_action_prob(&board, temp, &predict(net));
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
    let action = dist.sample(&mut rng) as u32;
    board.action(action, board.turn);

    let r = board.game_ended();

    if r != 0 {
      // println!("{}", board);
      for mut ex in &mut examples {
        // v: 1.0 black won
        // v: -1.0 white won
         ex.v = r as f32;
      }
      return examples;
    }
  }
}

fn player<'a>(_mcts: &'a mut MCTS, _net: &'a NNet, temp: f32) -> Box<dyn FnMut(&mut Board) -> usize + 'a> {
  Box::new(move |x: &mut Board| -> usize {
    if temp >= 1.0 {
      let mut rng = rand::thread_rng();
      let mut valids: Vec<usize> = x.vec_valid_moves(x.turn).iter().enumerate().map(|(i, &val)| {
        if val { i } else { 10000 }
      }).filter(|&r| r < 10000).collect();
      valids.shuffle(&mut rng);
      // println!("{:?}", valids);
      valids[0]
    } else {
      let probs = _mcts.get_action_prob(&x, temp, &predict(_net));
      println!("{:?}", probs);
      mcts::max_idx(&probs)
    }
  })
}

fn learn() {
  let num_iters = 50;
  let num_eps = 100;
  let maxlen_of_queue = 2000;
  let num_iters_for_train_examples_history = 10000;
  let update_threshold = 0.55;
  let skip_first_self_play = false;
  let mut train_examples_history = Examples {
    values: VecDeque::with_capacity(num_iters_for_train_examples_history)
  };
  let mut rng = thread_rng();

  let board_size: i64 = 5;
  let action_size: i64 = 26;
  let num_channels: i64 = 32;
  let mut net = NNet::new(board_size, action_size, num_channels);
  let mut pnet = NNet::new(board_size, action_size, num_channels);
  // let board = Board::new(BoardSize::S5);
  net.load("temp/best.pt");

  for i in 1..(num_iters + 1) {
    println!("Starting Iter #{} ...", i);
    //if skip_first_self_play || i > 1 {
    {
      let mut iteration_train_examples = Examples {
        values: VecDeque::with_capacity(maxlen_of_queue)
      };
      println!("self playing...");
      for _ in (0..num_eps).progress() {
        let mut mcts = MCTS::new(action_size); // reset search tree
        let examples = execute_episode(&mut mcts, &net);
        iteration_train_examples.values.extend(examples);
      }

      // save the iteration examples to the history 
      train_examples_history.values.extend(iteration_train_examples.values);
    }
    // backup history to a file
    // NB! the examples were collected using the model from the previous iteration, so (i-1)
    // save_file(&format!("temp/train_examples_{}.bin", i - 1), 0, &train_examples_history).unwrap();

    // shuffle examples before training
    let mut train_examples = Vec::new();
    for e in &train_examples_history.values {
      train_examples.push(e);
    }
    train_examples.shuffle(&mut rng);

    // training new network, keeping a copy of the old one
    net.save("temp/temp.pt");
    pnet.load("temp/temp.pt");
    let mut pmcts = MCTS::new(action_size);

    net.train(train_examples);
    let mut nmcts = MCTS::new(action_size);

    println!("PITTING AGAINST PREVIOUS VERSION");
    let temp = 0.0;
    
    let game_result;
    {
      let mut player1 = player(&mut pmcts, &pnet, temp);
      let mut player2 = player(&mut nmcts, &net, temp);
      game_result = play_games(100, &mut player1, &mut player2);
    }
    let (pwins, nwins, draws) = game_result;

    println!("NEW/PREV WINS : {} / {} ; DRAWS : {}", nwins, pwins, draws);
    if pwins + nwins == 0 || nwins * 100 / (pwins + nwins) < (update_threshold * 100.0) as u32 {
      println!("REJECTING NEW MODEL");
      net.load("temp/temp.pt");
    } else {
      println!("ACCEPTING NEW MODEL");
      net.save(format!("temp/checkpoint_{}.pt", i));
      net.save("temp/best.pt");
    }
  }
}

fn play_game<F: FnMut(&mut Board) -> usize>(player1: &mut F, player2: & mut F) -> i8 {
  let players = [player2, player1];
  let mut cur_player = 1;
  let mut board = Board::new(BoardSize::S5);
  let mut it = 0;
  let verbose = true;
  while board.game_ended() == 0 {
    it += 1;
    // if verbose {
    //   println!("Turn {} Player {}", it, board.turn as i32);
    // }
    let action = players[cur_player](&mut board);

    let valids = board.vec_valid_moves(board.turn);
    if !valids[action] {
      println!("Action {} is not valid!", action);
      println!("valids = {:?}", valids);
      assert!(valids[action]);
    }
    // println!("{}", board.valid_moves(board.turn));
    // println!("Turn {} Player {} action {} {}", it, board.turn as i32, action % 5 + 1, action / 5 + 1);
    // println!("{}", board);
    board.action(action as u32, board.turn);
    // println!("Turn {} Player {} action {} {}", it, board.turn as i32, action % 5 + 1, action / 5 + 1);
    cur_player = (board.turn as i32 + 1) as usize / 2;
  }
  // if verbose {
  //   println!("Game over: Turn {} Result {}", it, board.game_ended());
  //   println!("");
  //   // println!("{}", board);
  // }
  board.game_ended()
}
fn play_games<F: FnMut(&mut Board) -> usize>(num: u32, player1: &mut F, player2: &mut F) -> (u32, u32, u32) {
  let num = num / 2;
  let mut one_won = 0;
  let mut two_won = 0;
  let mut draws = 0;

  println!("Arena.playGames (1)");
  for _ in (0..num).progress() {
    let game_result = play_game(player1, player2);
    match game_result {
      1 => one_won += 1,
      -1 => two_won += 1,
      _ => draws += 1
    }
  }

  println!("Arena.playGames (2)");
  for _ in (0..num).progress() {
    let game_result = play_game(player2, player1);
    match game_result {
      -1 => one_won += 1,
      1 => two_won += 1,
      _ => draws += 1
    }
  }

  (one_won, two_won, draws)
}

// fn main() {
//   let mut board = Board::new(BoardSize::S5);
//   let b = Turn::Black;
//   let w = Turn::White;
//   let tb = Stones::new32(0b00000_00000_00000_00000_10000);
//   let tw = Stones::new32(0b01100_10011_01010_00101_00010);
//   board.set_stones(b, tb);
//   board.set_stones(w, tw);
//   board.remove_death_stones(b);
//   println!("{}", board);
// }
