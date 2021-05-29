// extern crate tch;
use tch::{nn, nn::ModuleT};
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
  rng: rand::rngs::StdRng,
  train_examples: Vec<Example>,
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
use tch::Tensor;
use tch::nn::{Adam, OptimizerConfig, VarStore};
use tch::vision::dataset::Dataset;
use tch::TrainableCModule;
use tch::{CModule, Device};
use std::path::Path;

fn main() {
  unsafe{ torch_sys::dummy_cuda_dependency(); }
  println!("cuda is_available: {}",tch::Cuda::is_available());
  tch::manual_seed(42);
  let mut coach = Coach {
    rng: rand::SeedableRng::from_seed([42; 32]),
    train_examples: Vec::new(),
    board_size: 5,
    action_size: 26,
    num_channels: 32,
  };
  coach.learn();
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
  net.train(examples);

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
