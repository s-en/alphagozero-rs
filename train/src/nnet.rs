use super::*;
use tch::{nn, Tensor, Device, IValue, nn::OptimizerConfig, no_grad, Kind, TchError, Reduction, nn::Conv2D, nn::FuncT, nn::ModuleT};
use anyhow::{bail, Result};
use indicatif::ProgressIterator;
use std::time::SystemTime;

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

impl NNet {
  pub fn new(board_size: i64, action_size: i64, num_channels: i64) -> NNet {
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let root = vs.root();
    NNet {
      board_size,
      action_size,
      num_channels,
      vs,
      model: None
    }
  }
  pub fn predict(net: &NNet, board: Vec<f32>) -> (Vec<f32>, f32) {
    let b = Tensor::of_slice(&board).to_device(net.vs.device());
    net.predict_tensor(b)
  }
  pub fn predict_tensor(&self, board: Tensor) -> (Vec<f32>, f32) {
    let b = board.view([9, self.board_size, self.board_size]);
    let mut pi: Tensor = Tensor::zeros(&[1, self.board_size*self.board_size+1], tch::kind::FLOAT_CUDA);
    let mut v: Tensor = Tensor::zeros(&[1, 1], tch::kind::FLOAT_CUDA);
    if let Some(model) = &self.model {
      let output = model.forward_is(&[tch::IValue::Tensor(b)]);
      let res = match output {
        Ok(val) => match val {
          IValue::Tuple(ivalues) => match &ivalues[..] {
              [IValue::Tensor(t1), IValue::Tensor(t2)] => Ok((t1.shallow_clone(), t2.shallow_clone())),
              _ => Err("unexpected output"),
          },
          _ => Err("unexpected output"),
        },
        _ => Err("forward_is error"),
      };
      if let Ok((pis, vs)) = res {
        pi = pis;
        v = vs;
      }
    }
    // no_grad(|| {
    //   if let Some(model) = self.model {
    //     let output = model.forward_is(&[tch::IValue::Tensor(b)]);
    //     let (pis, vs) = match output {
    //       Ok(val) => match val {
    //         IValue::Tuple(ivalues) => match &ivalues[..] {
    //             [IValue::Tensor(t1), IValue::Tensor(t2)] => (t1.shallow_clone(), t2.shallow_clone()),
    //             _ => bail!("unexpected output {:?}", ivalues),
    //         },
    //         _ => bail!("unexpected output {:?}", output),
    //       },
    //       _ => bail!("forward_is error"),
    //     };
    //     pi = pis;
    //     v = vs;
    //   }
    // });
    let r1 = Vec::<f32>::from(&pi);
    let r2 = v.double_value(&[0]) as f32;
    (r1, r2)
  }
  pub fn predict32(net: &NNet, board: Vec<Vec<f32>>) -> Vec<(Vec<f32>, f32)> {
    let b = Tensor::of_slice2(&board).to_device(net.vs.device());
    net.predict32_tensor(b, board.len() as i64)
  }
  pub fn predict32_tensor(&self, board: Tensor, num: i64) -> Vec<(Vec<f32>, f32)> {
    let b = board.view([num, 9, self.board_size, self.board_size]);
    let mut pi: Tensor = Tensor::zeros(&[num, self.board_size*self.board_size+1], tch::kind::FLOAT_CUDA);
    let mut v: Tensor = Tensor::zeros(&[num, 1], tch::kind::FLOAT_CUDA);
    if let Some(model) = &self.model {
      let output = model.forward_is(&[tch::IValue::Tensor(b)]);
      let res = match output {
        Ok(val) => match val {
          IValue::Tuple(ivalues) => match &ivalues[..] {
              [IValue::Tensor(t1), IValue::Tensor(t2)] => Ok((t1.shallow_clone(), t2.shallow_clone())),
              _ => Err("unexpected output"),
          },
          _ => Err("unexpected output"),
        },
        _ => Err("forward_is error"),
      };
      if let Ok((pis, vs)) = res {
        pi = pis;
        v = vs;
      }
    }
    // no_grad(|| {
    //   let (pis, vs) = self.forward(&b, false);
    //   pi = pis;
    //   v = vs;
    // });
    let mut res = Vec::new();
    let rs1 = pi.view([num, self.board_size*self.board_size+1]);
    let rs2 = v.view([num, 1]);
    for i in 0..num {
      let r1 = Vec::<f32>::from(rs1.narrow(0, i, 1));
      let r2 = rs2.double_value(&[i, 0]) as f32;
      res.push((r1, r2));
    }
    res
  }
  pub fn train(&self, trainable_model: &mut TrainableCModule, examples: Vec<&Example>) -> Result<()> {
    // examples: list of examples, each example is of form (board, pi, v)
    tch::manual_seed(42);
    let mut optimizer = nn::Adam::default().build(&self.vs, 1e-3)?;
    let epochs = 10;
    let batch_size = 128;
    println!("start train");
    let mut rnd = rand::thread_rng();
    trainable_model.set_train();
    for i in 0..epochs {
      for j in 0..10 {
        // println!("{} {}", i, j);
        let sample_ids = randint(examples.len(), batch_size, &mut rnd);
        let ex: Vec<&Example> = examples.iter().enumerate().filter(|(i, _)| sample_ids.contains(i)).map(|(_, e)| *e).collect();
        let boards = Tensor::of_slice2(&ex.iter().map(|x| &x.board).collect::<Vec<&Vec<f32>>>()).to_device(self.vs.device());
        let target_pis = Tensor::of_slice2(&ex.iter().map(|x| &x.pi).collect::<Vec<&Vec<f32>>>()).to_device(self.vs.device());
        let target_vs = Tensor::of_slice(&ex.iter().map(|x| x.v).collect::<Vec<f32>>()).to_device(self.vs.device());
        // compute output
        let ten = boards.apply_t(trainable_model, true);
        ten.print();
        //let (out_pi, out_v) = boards.apply_t(&trainable_model, true);
        // let l_pi = -(&target_pis * &out_pi.log()).sum(tch::Kind::Float) / target_pis.size()[0] as f64;
        // let l_v = (&target_vs - &out_v.view(-1)).pow(2).sum(tch::Kind::Float) / target_vs.size()[0] as f64;
        // let total_loss = l_pi + l_v;

        // optimizer.zero_grad();
        // total_loss.backward();
        // optimizer.step();
      }
    }
    Ok(())
  }
  pub fn save<T: AsRef<std::path::Path>>(&self, path: T) {
    // self.vs.save(path)
    if let Some(model) = &self.model {
      model.save(path);
    }
  }
  pub fn load<T: AsRef<std::path::Path>>(&mut self, path: T) {
    // self.vs.load(path)
    let model = CModule::load(
      path
    ).expect("failed loading dualnet model");
    self.model = Some(model);
  }
  pub fn load_trainable<T: AsRef<std::path::Path>>(&mut self, path: T) -> TrainableCModule {
    // self.vs.load(path)
    TrainableCModule::load(
      path, self.vs.root()
    ).expect("failed loading deualnet trainable model")
  }
}
