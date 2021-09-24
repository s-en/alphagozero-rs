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
  let bsize = 5;
  let mut coach = Coach {
    board_size: bsize,
    action_size: bsize*bsize+1,
    num_channels: 32,
  };
  coach.learn();


  // let mut probs: Vec<f32> = vec![];
  // probs.push(0.5);
  // for a in 0..5 {
  //   probs.push(0.0);
  // }
  // let pmin = probs.iter().fold(f32::INFINITY, |m, v| v.min(m));
  // probs = probs.iter().map(|&p| if p==0.0 {0.0} else {p-pmin+0.01}).collect();
  // println!("probs {:?}", probs);


  // let board_size: i64 = 5;
  // let action_size: i64 = 26;
  // let num_channels: i64 = 32;
  // let mut net = NNet::new(board_size, action_size, num_channels);
  // net.load_trainable("temp/best.pt");
  // let mut mcts = MCTS::new(900, 1.0);
  // let mut board = Board::new(BoardSize::S5);
  // let b = Turn::Black;
  // let w = Turn::White;
  // //board.action(25, board.turn);
  // // board.action(17, board.turn);
  // // board.action(19, board.turn);
  // // board.action(6, board.turn);
  // // board.action(8, board.turn);
  // // board.action(23, board.turn);
  // // board.action(11, board.turn);
  // // board.action(7, board.turn);
  // // board.action(15, board.turn);
  // // board.action(12, board.turn);
  // // board.action(3, board.turn);
  // // board.action(22, board.turn);
  // // board.action(13, board.turn);
  // // board.action(4, board.turn);
  // // board.action(21, board.turn);
  // // board.action(1, board.turn);
  // // board.action(5, board.turn);
  // // // let tb = Stones::new32(0b01110_00101_00011_00011_00001);
  // // // let tw = Stones::new32(0b10000_11000_01100_11100_01110);
  // // //let tb = Stones::new32(0b11110_00101_00010_00010_00010);
  // // // //let tw = Stones::new32(0b00000_11000_10100_01101_00100);
  // // // let tw = Stones::new32(0b00000_11000_11100_01101_11100);
  // // board.turn = b;
  // // //board.set_stones(b, tb);
  // // // board.set_stones(w, tw);
  // println!("turn {:?}", board.turn as i32);
  // println!("diff {:?}", board.count_diff());
  // println!("step {:?}", board.step);
  // println!("{}", board);
  // let predict32 = |inputs: Vec<Vec<f32>>| {
  //   NNet::predict32(&net, inputs)
  // };
  // let temp = 0.0;
  // let pi = NNet::predict(&net, board.input());
  // println!("pi {:?}", pi);
  // for i in 0..1 {
  //   let probs = mcts.get_action_prob(&board, temp, &predict32, false, true, 0);
  //   println!("probs {:?}", probs[25]);
  // }


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
