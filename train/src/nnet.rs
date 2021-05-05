use super::*;
use tch::{nn, Tensor, Device, nn::OptimizerConfig, no_grad, Kind, TchError};
use anyhow::Result;
use indicatif::ProgressIterator;

fn randint(max: usize, size: usize, rng: &mut rand::rngs::StdRng) -> Vec<usize> {
  let mut vec: Vec<usize> = Vec::with_capacity(size);
  for _ in 0..size {
    let rnd: f32 = rng.gen();
    vec.push((rnd * max as f32) as usize);
  };
  vec
}

pub fn randfloat(max: usize, size: usize, rng: &mut rand::rngs::StdRng) -> Vec<f32> {
  let mut vec: Vec<f32> = Vec::with_capacity(size);
  for _ in 0..size {
    let rnd: f32 = rng.gen();
    vec.push(rnd * max as f32);
  };
  vec
}

impl NNet {
  pub fn new(board_size: i64, action_size: i64, num_channels: i64) -> NNet {
    let conv2d_cfg = nn::ConvConfig {
      stride: 1,
      padding: 1,
      ..Default::default()
    };
    let conv2d_cfg_after = nn::ConvConfig {
      stride: 1,
      padding: 0,
      ..Default::default()
    };
    let vs = nn::VarStore::new(Device::Cpu);
    let root = &vs.root();
    let conv1 = nn::conv2d(root, 9, num_channels, 3, conv2d_cfg);
    let conv2 = nn::conv2d(root, num_channels, num_channels, 3, conv2d_cfg);
    let conv3 = nn::conv2d(root, num_channels, num_channels, 3, conv2d_cfg_after);
    let conv4 = nn::conv2d(root, num_channels, num_channels, 3, conv2d_cfg_after);
    let bn1 = nn::batch_norm2d(root, num_channels, Default::default());
    let bn2 = nn::batch_norm2d(root, num_channels, Default::default());
    let bn3 = nn::batch_norm2d(root, num_channels, Default::default());
    let bn4 = nn::batch_norm2d(root, num_channels, Default::default());
    let fc1 = nn::linear(root, num_channels*(board_size-4)*(board_size-4), 1024, Default::default());
    let fc_bn1 = nn::batch_norm1d(root, 1024, Default::default());
    let fc2 = nn::linear(root, 1024, 512, Default::default());
    let fc_bn2 = nn::batch_norm1d(root, 512, Default::default());
    let fc3 = nn::linear(root, 512, action_size, Default::default());
    let fc4 = nn::linear(root, 512, 1, Default::default());
    NNet {
      board_size,
      action_size,
      num_channels,
      vs,
      conv1,
      conv2,
      conv3,
      conv4,
      bn1,
      bn2,
      bn3,
      bn4,
      fc1,
      fc_bn1,
      fc2,
      fc_bn2,
      fc3,
      fc4,
    }
  }
  pub fn forward(&self, xs: &Tensor, train: bool) -> (Tensor, Tensor) {
    let s= xs.view([-1, 9, self.board_size, self.board_size])
        .apply(&self.conv1).apply_t(&self.bn1, train).relu()
        .apply(&self.conv2).apply_t(&self.bn2, train).relu()
        .apply(&self.conv3).apply_t(&self.bn3, train).relu()
        .apply(&self.conv4).apply_t(&self.bn4, train).relu()
        .view([-1, self.num_channels*(self.board_size-4)*(self.board_size-4)])
        .apply(&self.fc1).apply_t(&self.fc_bn1, train).relu().dropout_(0.3, train)
        .apply(&self.fc2).apply_t(&self.fc_bn2, train).relu().dropout_(0.3, train);
    let pi = s.apply(&self.fc3);
    let v = s.apply(&self.fc4);
    (pi.log_softmax(1, Kind::Float), v.tanh())
  }
  pub fn predict(net: &NNet, board: Vec<f32>) -> (Vec<f32>, f32) {
    let b = Tensor::of_slice(&board);
    net.predict_tensor(b)
  }
  pub fn predict_tensor(&self, board: Tensor) -> (Vec<f32>, f32) {
    // todo cuda
    let b = board.view([9, self.board_size, self.board_size]);
    let mut pi: Tensor = Tensor::zeros(&[1, self.board_size*self.board_size+1], tch::kind::FLOAT_CPU);
    let mut v: Tensor = Tensor::zeros(&[1, 1], tch::kind::FLOAT_CPU);
    no_grad(|| {
      let (pis, vs) = self.forward(&b, false);
      pi += pis;
      v += vs;
    });
    let r1 = Vec::<f32>::from(pi.exp());
    let r2 = v.double_value(&[0, 0]) as f32;
    (r1, r2)
  }
  pub fn train(&self, examples: Vec<&Example>) -> Result<()> {
    // examples: list of examples, each example is of form (board, pi, v)
    tch::manual_seed(42);
    let mut optimizer = nn::Adam::default().build(&self.vs, 1e-3)?;
    let epochs = 100;
    let batch_size = 32;
    println!("start train");
    let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([42; 32]);
    for _ in (0..epochs).progress() {
      for _ in (0..10).progress() {
        let sample_ids = randint(examples.len(), batch_size, &mut rng);
        let ex: Vec<&Example> = examples.iter().enumerate().filter(|(i, _)| sample_ids.contains(i)).map(|(_, e)| *e).collect();
        let boards = Tensor::of_slice2(&ex.iter().map(|x| &x.board).collect::<Vec<&Vec<f32>>>());
        let target_pis = Tensor::of_slice2(&ex.iter().map(|x| &x.pi).collect::<Vec<&Vec<f32>>>());
        let target_vs = Tensor::of_slice(&ex.iter().map(|x| x.v).collect::<Vec<f32>>());
        // todo cuda

        // compute output
        let (out_pi, out_v) = self.forward(&boards, true);
        let l_pi = -f64::from((&target_pis * &out_pi).sum(tch::Kind::Float)) / target_pis.size()[0] as f64;
        let l_v = (&target_vs - &out_v.view(-1)).pow(2).sum(tch::Kind::Float) / target_vs.size()[0] as f64;
        let total_loss = l_pi + l_v;

        optimizer.zero_grad();
        total_loss.backward();
        optimizer.step();
      }
    }
    Ok(())
  }
  pub fn save<T: AsRef<std::path::Path>>(&self, path: T) -> Result<(), TchError> {
    self.vs.save(path)
  }
  pub fn load<T: AsRef<std::path::Path>>(&mut self, path: T) -> Result<(), TchError> {
    self.vs.load(path)
  }
}
