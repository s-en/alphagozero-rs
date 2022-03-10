use super::*;
use tch::{nn, Tensor, Device, nn::OptimizerConfig, Kind, nn::Conv2D, nn::ModuleT, autocast};
use anyhow::Result;
use std::time::Instant;

fn randint(max: usize, size: usize, rnd: &mut ThreadRng) -> Vec<usize> {
  let mut vec: Vec<usize> = Vec::with_capacity(size);
  if max <= 0 {
    println!("randint too small!! {}", max);
  }
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
    let b = board.view([12, self.board_size, self.board_size]).totype(Kind::Float);
    let mut pi: Tensor = Tensor::zeros(&[1, self.action_size], (Kind::Float, self.vs.device()));
    let mut v: Tensor = Tensor::zeros(&[1, 1], (Kind::Float, self.vs.device()));
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
    //let start = Instant::now();
    let b = Tensor::of_slice2(&board).to_device(net.vs.device()).totype(Kind::Float);
    let res = net.predict32_tensor(b, board.len() as i64);
    //let end = start.elapsed();
    // println!("{:?}", board);
    // println!("res {:?}", res);
    // println!("predict32 {}.{:03}秒", end.as_secs(), end.subsec_nanos() / 1_000_000);
    res
  }
  pub fn predict32_tensor(&self, board: Tensor, num: i64) -> Vec<(Vec<f32>, f32)> {
    let mut res = Vec::new();
    tch::no_grad(|| {
      let b = board.view([num, 12, self.board_size, self.board_size]);
      let mut pi: Tensor = Tensor::zeros(&[num, self.action_size], (Kind::Float, self.vs.device()));
      let mut v: Tensor = Tensor::zeros(&[num, 1], (Kind::Float, self.vs.device()));
      if let Some(model) = &self.model {
        // let start = Instant::now();
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
      let rs1 = pi.view([num, self.action_size]);
      let rs2 = v.view([num, 1]);
      for i in 0..num {
        let r1 = Vec::<f32>::from(rs1.narrow(0, i, 1));
        let r2 = rs2.double_value(&[i, 0]) as f32;
        res.push((r1, r2));
      }
    });
    res
  }
  pub fn train(&mut self, examples: Vec<&Example>, lr: f64) -> Result<()> {
    // examples: list of examples, each example is of form (board, pi, v)
    let mut trainable_model;
    if let Some(model) = &mut self.tmodel {
      trainable_model = model;
    } else {
      panic!("trainable_model not found");
    }
    let mut optimizer = nn::Adam::default().build(&self.vs, lr)?;
    let batch_size: usize = 512;
    let epochs: i32 = 2000;
    println!("start train examples:{}", examples.len());
    let mut rnd = rand::thread_rng();
    trainable_model.set_train();
    let mut loss_hist = 0.0;
    let mut loss_cnt = 0;
    let eps = 1e-7;
    for i in 0..epochs {
      let ex: Vec<&Example> = examples.choose_multiple(&mut rnd, batch_size).cloned().collect();
      let mut ex_board: Vec<&Vec<f32>> = Vec::new();
      let mut ex_pis: Vec<&Vec<f32>> = Vec::new();
      let mut ex_vs: Vec<f32> = Vec::new();
      for row in ex.iter() {
        ex_board.push(&row.board);
        ex_pis.push(&row.pi);
        ex_vs.push(row.v);
      }
      let boards = Tensor::of_slice2(&ex_board).to_device(self.vs.device());
      let target_pis = Tensor::of_slice2(&ex_pis).to_device(self.vs.device());
      let target_vs = Tensor::of_slice(&ex_vs).to_device(self.vs.device());
      // compute output
      let output = boards.apply_t(trainable_model, true);
      let out_pi = output.narrow(1, 0 , self.action_size);
      let out_v = output.narrow(1, self.action_size, 1);
      let l_pi = -(&target_pis * (&out_pi+eps).log()).sum(tch::Kind::Float) / target_pis.size()[0];
      let l_v = (&target_vs - &out_v.view(-1)).pow(2).sum(tch::Kind::Float) / target_vs.size()[0];
      let total_loss = l_pi + l_v;
      let vloss = Vec::<f32>::from(&total_loss)[0];
      loss_hist += vloss;
      loss_cnt += 1;
      if i % (epochs / 20 + 1) == epochs / 20 {
        println!("loss {:?}", loss_hist / loss_cnt as f32);
        loss_hist = 0.0;
        loss_cnt = 0;
      }

      let nan = total_loss.isnan().sum(tch::Kind::Float);
      if i64::from(nan) > 0 {
        println!("has nan");
        println!("target_pis {:?}", target_pis);
        println!("output {:?}", output.isnan().sum(tch::Kind::Float));
        // output.print();
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
