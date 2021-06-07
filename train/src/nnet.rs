use super::*;
use tch::{nn, Tensor, Device, IValue, nn::OptimizerConfig, no_grad, Kind, TchError, Reduction, nn::Conv2D, nn::FuncT, nn::ModuleT};
use anyhow::{bail, Result};
use indicatif::ProgressIterator;
use std::time::SystemTime;
use tch::IndexOp;
use std::time::Instant;

fn randint(max: usize, size: usize, rnd: &mut ThreadRng) -> Vec<usize> {
  let mut vec: Vec<usize> = Vec::with_capacity(size);
  for _ in 0..size {
    vec.push(rnd.gen_range(0, max) as usize);
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

pub fn learning_rate(epoch: i64) -> f64 {
  if epoch < 80 {
    0.005
  } else {
    0.002
  }
}

impl NNet {
  pub fn new(board_size: i64, action_size: i64, num_channels: i64) -> NNet {
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let root = vs.root();
    NNet {
      board_size,
      action_size,
      num_channels,
      vs,
      model: None,
      tmodel: None
    }
  }
  pub fn predict(net: &NNet, board: Vec<f32>) -> (Vec<f32>, f32) {
    let b = Tensor::of_slice(&board).to_device(net.vs.device());
    net.predict_tensor(b)
  }
  pub fn predict_tensor(&self, board: Tensor) -> (Vec<f32>, f32) {
    let b = board.view([12, self.board_size, self.board_size]);
    let mut pi: Tensor = Tensor::zeros(&[1, self.action_size], tch::kind::FLOAT_CUDA);
    let mut v: Tensor = Tensor::zeros(&[1, 1], tch::kind::FLOAT_CUDA);
    if let Some(model) = &self.model {
      let output = model.forward_t(&b, false);
      pi = output.narrow(1, 0, self.action_size);
      v = output.narrow(1, self.action_size, 1);
    } else if let Some(model) = &self.tmodel {
      let output = model.forward_t(&b, false);
      pi = output.narrow(1, 0, self.action_size);
      v = output.narrow(1, self.action_size, 1);
    } else {
      println!("predict_tensor model not found");
    }
    let rs1 = pi.view([self.action_size]);
    let rs2 = v.view([1]);
    let r1 = Vec::<f32>::from(rs1);
    let r2 = rs2.double_value(&[0]) as f32;
    (r1, r2)
  }
  pub fn predict32(net: &NNet, board: Vec<Vec<f32>>) -> Vec<(Vec<f32>, f32)> {
    let start = Instant::now();
    let b = Tensor::of_slice2(&board).to_device(net.vs.device());
    let res = net.predict32_tensor(b, board.len() as i64);
    let end = start.elapsed();
    // println!("{:?}", board);
    // println!("res {:?}", res);
    //println!("predict32 {}.{:03}秒", end.as_secs(), end.subsec_nanos() / 1_000_000);
    res
  }
  pub fn predict32_tensor(&self, board: Tensor, num: i64) -> Vec<(Vec<f32>, f32)> {
    let b = board.view([num, 12, self.board_size, self.board_size]);
    let mut pi: Tensor = Tensor::zeros(&[num, self.action_size], tch::kind::FLOAT_CUDA);
    let mut v: Tensor = Tensor::zeros(&[num, 1], tch::kind::FLOAT_CUDA);
    if let Some(model) = &self.model {
      //let start = Instant::now();
      let output = model.forward_t(&b, false);
      // let end = start.elapsed();
      // println!("forward_t {}.{:03}秒", end.as_secs(), end.subsec_nanos() / 1_000_000);
      pi = output.narrow(1, 0, self.action_size);
      v = output.narrow(1, self.action_size, 1);
    } else if let Some(model) = &self.tmodel {
      let output = model.forward_t(&b, false);
      pi = output.narrow(1, 0, self.action_size);
      v = output.narrow(1, self.action_size, 1);
    } else {
      println!("predict32_tensor model not found");
    }
    let mut res = Vec::new();
    let rs1 = pi.view([num, self.action_size]);
    let rs2 = v.view([num, 1]);
    for i in 0..num {
      let r1 = Vec::<f32>::from(rs1.narrow(0, i, 1));
      let r2 = rs2.double_value(&[i, 0]) as f32;
      res.push((r1, r2));
    }
    res
  }
  pub fn train(&mut self, examples: Vec<&Example>, lr: f64) -> Result<()> {
    // examples: list of examples, each example is of form (board, pi, v)
    tch::manual_seed(42);
    let trainable_model;
    if let Some(model) = &mut self.tmodel {
      trainable_model = model;
    } else {
      panic!("trainable_model not found");
    }
    let mut optimizer = nn::Adam::default().build(&self.vs, lr)?;
    let epochs = 200;
    let batch_size = 512;
    println!("start train");
    let mut rnd = rand::thread_rng();
    trainable_model.set_train();
    // let mut prev_model;
    // let mut prev_optimizer;
    for i in 0..epochs {
      for j in 0..10 {
        // println!("{} {}", i, j);
        let sample_ids = randint(examples.len(), batch_size, &mut rnd);
        let ex: Vec<&Example> = examples.iter().enumerate().filter(|(i, _)| sample_ids.contains(i)).map(|(_, e)| *e).collect();
        let boards = Tensor::of_slice2(&ex.iter().map(|x| &x.board).collect::<Vec<&Vec<f32>>>()).to_device(self.vs.device());
        let target_pis = Tensor::of_slice2(&ex.iter().map(|x| &x.pi).collect::<Vec<&Vec<f32>>>()).to_device(self.vs.device());
        let target_vs = Tensor::of_slice(&ex.iter().map(|x| x.v).collect::<Vec<f32>>()).to_device(self.vs.device());
        // compute output
        let output = boards.apply_t(trainable_model, true);
        let out_pi = output.narrow(1, 0 , self.action_size);
        let out_v = output.narrow(1, self.action_size, 1);
        let l_pi = -(&target_pis * &out_pi.log()).sum(tch::Kind::Float) / target_pis.size()[0] as f64;
        let l_v = (&target_vs - &out_v.view(-1)).pow(2).sum(tch::Kind::Float) / target_vs.size()[0] as f64;
        let total_loss = l_pi + l_v;
        if i %39 == 0 && j == 5 {
          println!("loss {:?}", Vec::<f32>::from(&total_loss));
        }

        let nan = total_loss.isnan().sum(tch::Kind::Float);
        if i64::from(nan) > 0 {
          println!("has nan");
          let l_pi = -(&target_pis * &out_pi.log()).sum(tch::Kind::Float) / target_pis.size()[0] as f64;
          let l_v = (&target_vs - &out_v.view(-1)).pow(2).sum(tch::Kind::Float) / target_vs.size()[0] as f64;
          total_loss.print();
          l_pi.print();
          l_v.print();
          println!("target_pis.size {:?}", target_pis.size()[0]);
          println!("target_vs.size {:?}", target_vs.size()[0]);
          println!("target_pis {:?}", target_pis);
          println!("out_pi {:?}", out_pi);
        } else {
          optimizer.zero_grad();
          total_loss.backward();
          optimizer.step();
        }
      }
    }
    Ok(())
  }
  pub fn save<T: AsRef<std::path::Path>>(&self, path: T) {
    // self.vs.save(path)
    if let Some(model) = &self.tmodel {
      model.save(path);
    }
  }
  pub fn load<T: AsRef<std::path::Path>>(&mut self, path: T) {
    // self.vs.load(path)
    let model = CModule::load(
      path
    ).expect("failed loading dualnet model");
    self.model = Some(model);
    self.tmodel = None;
  }
  pub fn load_trainable<T: AsRef<std::path::Path>>(&mut self, path: T) {
    // self.vs.load(path)
    let mut model = TrainableCModule::load(
      path, self.vs.root()
    ).expect("failed loading deualnet trainable model");
    self.tmodel = Some(model);
    self.model = None;
  }
}
