use tch::{nn};

pub mod nnet;

#[derive(Debug)]
pub struct NNet {
  board_size: i64,
  action_size: i64,
  num_channels: i64,
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

pub struct Example {
  pub board: Vec<f32>,
  pub pi: Vec<f32>,
  pub v: Vec<f32>
}
