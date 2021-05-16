use super::*;
use tch::{nn, Tensor, Device, nn::OptimizerConfig, no_grad, Kind, TchError, Reduction, nn::Conv2D, nn::FuncT, nn::ModuleT};
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

fn conv2d(p: &nn::Path, c_in: i64, c_out: i64, ksize: i64, padding: i64, stride: i64) -> Conv2D {
  let conv2d_cfg = nn::ConvConfig {
      stride,
      padding,
      bias: false,
      ..Default::default()
  };
  nn::conv2d(p, c_in, c_out, ksize, conv2d_cfg)
}

fn downsample(p: &nn::Path, c_in: i64, c_out: i64, stride: i64) -> impl ModuleT {
  if stride != 1 || c_in != c_out {
      nn::seq_t()
          .add(conv2d(p, c_in, c_out, 1, 0, stride))
          .add(nn::batch_norm2d(p, c_out, Default::default()))
  } else {
      nn::seq_t()
  }
}

fn basic_block(p: &nn::Path, c_in: i64, c_out: i64, stride: i64) -> impl ModuleT {
  let conv1 = conv2d(p, c_in, c_out, 3, 1, stride);
  let bn1 = nn::batch_norm2d(p, c_out, Default::default());
  let conv2 = conv2d(p, c_out, c_out, 3, 1, 1);
  let bn2 = nn::batch_norm2d(p, c_out, Default::default());
  let downsample = downsample(p, c_in, c_out, stride);
  nn::func_t(move |xs, train| {
      let ys = xs
          .apply(&conv1)
          .apply_t(&bn1, train)
          .relu()
          .apply(&conv2)
          .apply_t(&bn2, train);
      (xs.apply_t(&downsample, train) + ys).relu()
  })
}

fn basic_layer(p: &nn::Path, c_in: i64, c_out: i64, stride: i64, cnt: i64) -> impl ModuleT {
  let mut layer = nn::seq_t().add(basic_block(p, c_in, c_out, stride));
  for _ in 1..cnt {
    layer = layer.add(basic_block(p, c_out, c_out, 1))
  }
  layer
}

impl NNet {
  pub fn new(board_size: i64, action_size: i64, num_channels: i64) -> NNet {
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let root = vs.root();
    let bs = board_size;
    let header_t = nn::seq_t()
        .add(conv2d(&root, 9, num_channels, 3, 1, 1))
        .add(nn::batch_norm2d(&root, num_channels, Default::default()))
        .add_fn(|xs| xs.relu());
    let blocks_t = nn::seq_t()
        .add(basic_layer(&root, num_channels, num_channels, 1, 10));
    let p_t = nn::seq_t()
        .add(conv2d(&root, num_channels, 32, 1, 0, 1)) // [1, 32, 5, 5]
        .add(nn::batch_norm2d(&root, 32, Default::default())) // [1, 32, 5, 5]
        .add_fn(move |xs| xs.relu().view([-1, bs*bs*32])) // [bs*bs*32]
        .add(nn::linear(&root, bs*bs*32, action_size, Default::default())) // [1, action_size]
        .add_fn(|xs| xs.log_softmax(-1, Kind::Float).exp()); // [1, action_size]
    let v_t = nn::seq_t()
        .add(conv2d(&root, num_channels, 3, 1, 0, 1))
        .add(nn::batch_norm2d(&root, 3, Default::default()))
        .add_fn(move |xs| xs.relu().view([-1, bs*bs*3]))
        .add(nn::linear(&root, bs*bs*3, 32, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(&root, 32, 1, Default::default()))
        .add_fn(|xs| xs.tanh());
    NNet {
      board_size,
      action_size,
      num_channels,
      vs,
      headerT: header_t,
      blocksT: blocks_t,
      pT: p_t,
      vT: v_t,
    }
  }
  pub fn forward(&self, xs: &Tensor, train: bool) -> (Tensor, Tensor) {
    let s= xs.view([-1, 9, self.board_size, self.board_size])
          .apply_t(&self.headerT, train)
          .apply_t(&self.blocksT, train);
    let pi = s.apply_t(&self.pT, train);
    let v = s.apply_t(&self.vT, train);
    (pi, v)
  }
  pub fn predict(net: &NNet, board: Vec<f32>) -> (Vec<f32>, f32) {
    let b = Tensor::of_slice(&board).to_device(net.vs.device());
    net.predict_tensor(b)
  }
  pub fn predict_tensor(&self, board: Tensor) -> (Vec<f32>, f32) {
    let b = board.view([9, self.board_size, self.board_size]);
    let mut pi: Tensor = Tensor::zeros(&[1, self.board_size*self.board_size+1], tch::kind::FLOAT_CUDA);
    let mut v: Tensor = Tensor::zeros(&[1, 1], tch::kind::FLOAT_CUDA);
    no_grad(|| {
      let (pis, vs) = self.forward(&b, false);
      pi = pis;
      v = vs;
    });
    let r1 = Vec::<f32>::from(&pi);
    let r2 = v.double_value(&[0]) as f32;
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
        let boards = Tensor::of_slice2(&ex.iter().map(|x| &x.board).collect::<Vec<&Vec<f32>>>()).to_device(self.vs.device());
        let target_pis = Tensor::of_slice2(&ex.iter().map(|x| &x.pi).collect::<Vec<&Vec<f32>>>()).to_device(self.vs.device());
        let target_vs = Tensor::of_slice(&ex.iter().map(|x| x.v).collect::<Vec<f32>>()).to_device(self.vs.device());
        // todo cuda
        // compute output
        let (out_pi, out_v) = self.forward(&boards, true);
        let l_pi = -(&target_pis * &out_pi.log()).sum(tch::Kind::Float) / target_pis.size()[0] as f64;
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
