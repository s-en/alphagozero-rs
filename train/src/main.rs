// extern crate tch;
use tch::nn;
use rand::prelude::*;
use std::collections::VecDeque;
use az_game::igo::*;

extern crate savefile;
#[macro_use]
extern crate savefile_derive;

pub mod nnet;
pub mod coach;

#[derive(Debug)]
pub struct NNet {
  board_size: i64,
  action_size: i64,
  num_channels: i64,
  vs: nn::VarStore,
  model: Option<CModule>,
  tmodel: Option<TrainableCModule>
}

pub struct Coach {
  board_size: i64,
  action_size: i64,
  num_channels: i64,
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

use anyhow::Result;
use tch::TrainableCModule;
use tch::CModule;

use rand::distributions::WeightedIndex;
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

fn main() {
  unsafe{ torch_sys::dummy_cuda_dependency(); }
  println!("cuda is_available: {}",tch::Cuda::is_available());
  let bsize = 7;
  let mut coach = Coach {
    board_size: bsize,
    action_size: bsize*bsize+1,
    num_channels: 32,
  };
  coach.learn();


  // let board_size: i64 = 7;
  // let action_size: i64 = 50;
  // let num_channels: i64 = 32;
  // let mut net = NNet::new(board_size, action_size, num_channels);
  // net.load_trainable("7x7/best.pt");
  // let mut mcts = MCTS::new(200, 1.0);
  // let mut board = Board::new(BoardSize::S7);
  // board.action(49, board.turn);
  // // board.action(1, board.turn);
  // // board.action(0, board.turn);
  // // board.action(7, board.turn);
  // // board.action(49, board.turn);
  // let predict32 = |inputs: Vec<Vec<f32>>| {
  //   NNet::predict32(&net, inputs)
  // };
  // let temp = 1.0;
  // let probs = mcts.get_action_prob(&board, temp, &predict32, false, false, false, 1);
  // println!("probs {:?}", probs);


  // while board.game_ended(true, 0) == 0 {
  //   let probs = mcts.get_action_prob(&board, temp, &predict32, false, true, 0);
  //   //println!("probs {:?}", probs);
  //   let a = max_idx(&probs) as u32;
  //   board.action(a, board.turn);
  //   //println!("{}", board);
  //   println!("{:?}", a);
  // }
  // // board.action(9, board.turn);
  // // println!("{}", board);
  // // let pi = NNet::predict(&net, board.input());
  // // println!("pi {:?}", pi);
}

#[test]
fn nnet_test() {
  let board_size: i64 = 5;
  let action_size: i64 = 26;
  let num_channels: i64 = 32;
  let mut net = NNet::new(board_size, action_size, num_channels);
  net.load_trainable("temp/dualnet.pt");

  let ex = Example {
    board: vec![
      1.0, 1.0, 1.0, 1.0, 1.0,
      1.0, 1.0, 1.0, 1.0, 1.0,
      1.0, 1.0, 1.0, 1.0, 1.0,
      1.0, 1.0, 1.0, 1.0, 1.0,
      1.0, 1.0, 1.0, 1.0, 1.0,

      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,

      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,

      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,

      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,

      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,

      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,

      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,

      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    pi: vec![
      1.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.
    ],
    v: 1.0
  };
  let board = Board::new(BoardSize::S5);
  let pi = NNet::predict(&net, board.input());
  println!("predict {:?}", pi);
  let mut examples = Vec::new();
  examples.push(&ex);
  examples.push(&ex);
  examples.push(&ex);
  net.train(examples, 0.01);

  let pi = NNet::predict(&net, board.input());
  println!("predict {:?}", pi);
  let p: Vec<f32> = pi.0.iter().map(|x| x.round()).collect();
  let v = pi.1.round();

  assert_eq!(
    p,
    vec![
      1.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.
    ]
  );
  assert_eq!(
    v, 1.0
  );
}
