// extern crate tch;
use rand::prelude::*;
use std::collections::VecDeque;
use az_game::igo::*;
use tch::{nn, Tensor, IValue, Device, nn::OptimizerConfig};

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
    num_channels: 8,
  };
  coach.learn();
  return;

  // let vs = nn::VarStore::new(Device::cuda_if_available());
  // let mut model = TrainableCModule::load(
  //   "7x7/dualnet7x7_4conv.pt", vs.root()
  // ).expect("failed loading deualnet trainable model");
  
  // let epochs: i32 = 1000;
  // let lr:f64 = 1e-5;
  // let mut rnd = rand::thread_rng();
  // let mut optimizer = nn::Adam::default().build(&vs, lr).unwrap();
  // fn make(n1:f32, n2:f32) -> Vec<f32> {
  //   let mut b = vec![0.0; 12*49];
  //   b[0] = n1;
  //   b[1] = n2;
  //   return b;
  // }
  // for i in 0..epochs {
  //   let mut ex_board: Vec<Vec<f32>> = Vec::new();
  //   let mut ex_vs: Vec<Vec<f32>> = Vec::new();
  //   for _ in 0..128 {
  //     let r = rnd.gen_range(0, 4) as usize;
  //     let mut b = vec![0.0; 12*49];
  //     if r == 0 {
  //       ex_board.push(make(0.0, 0.0));
  //       ex_vs.push(vec![0.0]);
  //     } else if r == 1 {
  //       ex_board.push(make(1.0, 0.0));
  //       ex_vs.push(vec![1.0]);
  //     } else if r == 2 {
  //       ex_board.push(make(0.0, 1.0));
  //       ex_vs.push(vec![1.0]);
  //     } else {
  //       ex_board.push(make(1.0, 1.0));
  //       ex_vs.push(vec![0.0]);
  //     }
  //   }

  //   let boards = Tensor::of_slice2(&ex_board).to_device(vs.device()).reshape(&[-1, 12, 7, 7]);
  //   let target_vs = Tensor::of_slice2(&ex_vs).to_device(vs.device());
  //   let output = model.forward_is(&[IValue::from(boards)]).unwrap();
  //   let (out_pi, out_v) = match output {
  //     IValue::Tuple(ivalues) => match &ivalues[..] {
  //       [IValue::Tensor(t1), IValue::Tensor(t2)] => (t1.shallow_clone(), t2.shallow_clone()),
  //       _ => panic!("unexpected output {:?}", ivalues),
  //     },
  //     _ => panic!("unexpected output {:?}", output),
  //   };
  //   let l_v = &out_v.mse_loss(&target_vs, tch::Reduction::Mean);
  //   if i % 100 == 0 {
  //     println!("loss {:?}", l_v);
  //   }
  //   optimizer.backward_step(&l_v);
  // }
  // let mut ex_board: Vec<Vec<f32>> = Vec::new();
  // ex_board.push(make(0.0, 0.0));
  // ex_board.push(make(0.0, 1.0));
  // ex_board.push(make(1.0, 0.0));
  // ex_board.push(make(1.0, 1.0));
  // let boards = Tensor::of_slice2(&ex_board).to_device(vs.device()).reshape(&[-1, 12, 7, 7]);
  // let output = model.forward_is(&[IValue::from(boards)]).unwrap();
  // let (out_pi, out_v) = match output {
  //   IValue::Tuple(ivalues) => match &ivalues[..] {
  //     [IValue::Tensor(t1), IValue::Tensor(t2)] => (t1.shallow_clone(), t2.shallow_clone()),
  //     _ => panic!("unexpected output {:?}", ivalues),
  //   },
  //   _ => panic!("unexpected output {:?}", output),
  // };
  // out_v.print();
  
  // return;

  // vの確認
  // let board_size: i64 = 7;
  // let action_size: i64 = 50;
  // let num_channels: i64 = 32;
  // let mut net = NNet::new(board_size, action_size, num_channels);
  // net.load_trainable("7x7/best.pt");
  // //let mut examples = Vec::new();
  // let predict32 = |inputs: Vec<Vec<f32>>| {
  //   NNet::predict32(&net, inputs)
  // };
  // let mut board = Board::new(BoardSize::S7);
  // let tb = Stones::new64(0b0010000_1111111_1111111_1111111_0000000_0000000_0000000);
  // let tw = Stones::new64(0b0000000_0000000_0000000_0100000_1110111_1101111_1111111);
  // board.set_stones(Turn::Black, tb);
  // board.set_stones(Turn::White, tw);
  // board.turn = Turn::Black;
  // println!("{:}", board);
  // let pi = NNet::predict(&net, board.input());
  // println!("pi {:?}", pi);
  // return;

  let board_size: i64 = 7;
  let action_size: i64 = 50;
  let num_channels: i64 = 32;
  let mut net = NNet::new(board_size, action_size, num_channels);
  net.load_trainable("7x7/dualnet7x7_4conv.pt");
  //let mut examples = Vec::new();
  let predict32 = |inputs: Vec<Vec<f32>>| {
    NNet::predict32(&net, inputs)
  };
  let mut board = Board::new(BoardSize::S7);
  // let tb = Stones::new64(0b0000000_0000000_0000000_0000000_0000000_0000000_0000000);
  // let tw = Stones::new64(0b1111111_1101111_1000111_1101111_1111111_1111111_1111111);
  let tb = Stones::new64(0b0000000_0000000_0000000_0000000_1111111_0000000_1111010);
  let tw = Stones::new64(0b1101111_1111111_1111111_1111111_0000000_0000000_0000000);
  // // let tb = Stones::new64(0b0000000_1000000_1100000_0111000_0100111_0000101_0001110);
  // // let tw = Stones::new64(0b1100000_0110000_0011111_0000100_0011000_0101000_0000000);
  board.set_stones(Turn::Black, tb);
  board.set_stones(Turn::White, tw);
  // board.turn = Turn::Black; board.pass_cnt = 0;
  // let pi = NNet::predict(&net, board.input());
  // println!("pi 1 0 = {:?}", pi);
  // board.turn = Turn::White; board.pass_cnt = 0;
  // let pi = NNet::predict(&net, board.input());
  // println!("pi 0 0 = {:?}", pi);
  // board.turn = Turn::Black; board.pass_cnt = 1;
  // let pi = NNet::predict(&net, board.input());
  // println!("pi 1 1 = {:?}", pi);
  // board.turn = Turn::White; board.pass_cnt = 1;
  // let pi = NNet::predict(&net, board.input());
  // println!("pi 0 1 = {:?}", pi);

  // board.action(42, board.turn);
  // board.action(29, board.turn);
  // board.action(3, board.turn);
  // board.action(1, board.turn);
  // board.action(18, board.turn);
  // board.action(27, board.turn);
  // let pi = NNet::predict(&net, board.input());
  // // let pi = NNet::predict32(&net, vec![vec![0.0; 12*49],vec![0.0; 12*49],vec![0.0; 12*49],vec![0.0; 12*49],vec![0.0; 12*49],vec![0.0; 12*49],vec![0.0; 12*49],vec![0.0; 12*49]]);
  // println!("pi {:?}", pi);
  // //let valids = board.vec_valid_moves_for_train(board.turn);
  // println!("{:}", board);
  //println!("{:?}", valids);

  println!("{}", board);
  let mut mcts = MCTS::new(100, 1.0);
  board.turn = Turn::Black;
  let mut probs: Vec<f32> = mcts.get_action_prob(&board, 0.2, &predict32, false, true, false, 0);//board.vec_valid_moves(board.turn).iter().map(|&x| x as i32 as f32).collect();
  println!("probs {:?}", probs);
  return;
  // println!("probs11 {:?}", probs[11]);

  // let pi = NNet::predict(&net, board.input());
  // println!("pi start= {:?}", pi);
  // let mut b2 = board.clone();
  // board.action(46, board.turn);
  // println!("{}", board);
  // let pi = NNet::predict(&net, board.input());
  // println!("pi after= {:?}", pi);

  // b2.action(2, b2.turn);
  // println!("{}", b2);
  // let pi = NNet::predict(&net, b2.input());
  // println!("pi after2= {:?}", pi);
  // let mut probs: Vec<f32> = mcts.get_action_prob(&b2, 0.2, &predict32, false, false, false, 0);//board.vec_valid_moves(board.turn).iter().map(|&x| x as i32 as f32).collect();
  // println!("probs {:?}", probs);

  // b2.action(0, b2.turn);
  // println!("{}", b2);
  // let pi = NNet::predict(&net, b2.input());
  // println!("pi after3= {:?}", pi);

  // board.action(49, Turn::Black);
  // board.action(48, Turn::White);
  // println!("{:}", board);
  // board.turn = Turn::White;
  // let mut mcts = MCTS::new(50, 1.0);
  // let mut probs: Vec<f32> = mcts.get_action_prob(&board, 1.0, &predict32, false, false, false, 0);
  // println!("probs {:?}", probs);
  // // for i in 0..1000 {
  // //   let mut board = Board::new(BoardSize::S5);
  // //   if i%2 == 0 {
  // //     let mut pi = vec![0.0; 26];
  // //     pi[1] = 1.0;
  // //     let ex = Example {
  // //       board: board.input(),
  // //       pi: pi,
  // //       v: 1.0
  // //     };
  // //     examples.push(ex);
  // //   } else {
  // //     let mut pi = vec![0.0; 26];
  // //     pi[2] = 1.0;
  // //     board.turn = Turn::White;
  // //     let ex = Example {
  // //       board: board.input(),
  // //       pi: pi,
  // //       v: -1.0
  // //     };
  // //     examples.push(ex);
  // //   }
  // // }
  // let board_size: i64 = 7;
  // let action_size: i64 = 50;
  // let num_channels: i64 = 32;
  // let mut net = NNet::new(board_size, action_size, num_channels);
  // net.load_trainable("7x7/dualnet7x7.pt");
  // let predict32 = |inputs: Vec<Vec<f32>>| {
  //   NNet::predict32(&net, inputs)
  // };
  // //let mut examples = Vec::new();
  // for i in 0..1 {
  //   let mut board = Board::new(BoardSize::S7);
  //   let tb = Stones::new64(0b0000000_0000000_0000000_0000010_0000011_0000010_0000010);
  //   let tw = Stones::new64(0b1011111_1111111_1111111_1111101_1111100_1111101_1111100);
  //   board.set_stones(Turn::Black, tb);
  //   board.set_stones(Turn::White, tw);
  //   println!("{:}", board);
  //   let mut result = 0;//board.game_ended(true, 0);
  //   let mut mcts = MCTS::new(50, 1.0);
  //   let mut probs: Vec<f32> = mcts.get_action_prob(&board, 0.2, &predict32, false, false, false, 0);//board.vec_valid_moves(board.turn).iter().map(|&x| x as i32 as f32).collect();
  //   println!("probs {:?}", probs);
  // }
  //   let mut cnt = 0;
  //   while result == 0 {
  //     let mut probs: Vec<f32> = mcts.get_action_prob(&board, 1.0, &predict32, false, false, false, 0);//board.vec_valid_moves(board.turn).iter().map(|&x| x as i32 as f32).collect();
  //     println!("probs {:?}", probs);
  //     // 次の手を選ぶ
  //     // let mut probs = vec![0.0; 26];
  //     // probs[0] = 0.1;
  //     // probs[2] = 0.1;
  //     // probs[25] = 0.8;
  //     let rng = &mut rand::thread_rng();
  //     let dist = WeightedIndex::new(&probs).unwrap();
  //     let mut a = dist.sample(rng) as u32;
  //     //let mut aprob: Vec<f32> = vec![0.0; 26];
  //     cnt += 1;
  //     // if cnt >= 5 {
  //     //   a = 0;
  //     //   probs[a as usize] = 1.0;
  //     //   //println!("{:?}", probs);
  //     // }
  //     //println!("{:?}", aprob);
  //     //println!("{:?}", &probs);
  //     let turn = board.turn;
  //     // let ex = Example {
  //     //   board: board.input(),
  //     //   pi: &probs,
  //     //   v: 0.0
  //     // };
  //     //examples.push(ex);
  //     let sym = board.symmetries(probs);

  //     board.action(a, board.turn);
  //     //println!("{}", board);
  //     // println!("{:?}", turn as i32);
  //     //println!("{:?}", a);
  //     // 12で勝ち
  //     // if a == 12 {
  //     //   result = turn as i32;
  //     // }
  //     // パスしたら負け
  //     if a == 25 {
  //       result = turn as i32 * -1;
  //       for (b, p) in sym {
  //         //println!("{:?}", b);
  //         let ex = Example {
  //           board: b,
  //           pi: p,
  //           v: 0.0
  //         };
  //         examples.push(ex);
  //       }
  //     }
  //     //result = board.game_ended(true, 0);
  //   }
  //   println!("{} {}", result, board.get_kifu_sgf());
  //   let v = result as i32;
  //   for mut ex in &mut examples {
  //     // v: 1.0 black won
  //     // v: -1.0 white won
  //       ex.v = v as f32;
  //   }
  //   //println!("{:}", board);
  // }
  //println!("{:?}", &examples[examples.len()-5]);
  //println!("{:?}", &examples[examples.len()-1]);
  // for ex in &examples {
  //   println!("{:?}", ex);
  // }
  // println!("examples {:}", examples.len());
  // // 学習
  // let lr = 1e-5;
  // let ex = examples.iter().map(|x| x).collect();
  // let _ = net.train(ex, lr);

  // // let pi = NNet::predict(&net, vec![1.0; 12*25]);
  // // println!("after {:?}", pi);
  // // let pi = NNet::predict(&net, vec![0.0; 12*25]);
  // // println!("after {:?}", pi);
  // let mut board = Board::new(BoardSize::S5);
  // let pi = NNet::predict(&net, board.input());
  // println!("after {:?}", pi);
  // //println!("after_center {:?}", pi.0[12]);
  // //println!("input {:?}", board.input());

  // //board.turn = Turn::White;
  // board.action(2, board.turn);
  // // // board.action(8, board.turn);
  // // // board.action(3, board.turn);
  // // // board.action(9, board.turn);
  // let pi = NNet::predict(&net, board.input());
  // println!("after {:?}", pi);
  // // println!("after_center {:?}", pi.0[12]);
  // // //println!("input {:?}", board.input());

  // let board_size: i64 = 5;
  // let action_size: i64 = 26;
  // let num_channels: i64 = 32;
  // let mut net = NNet::new(board_size, action_size, num_channels);
  // net.load_trainable("5x5/dualnet5x5.pt");
  // let mut mcts = MCTS::new(50, 1.0);
  // let mut board = Board::new(BoardSize::S5);
  // for i in 0..22 {
  //   board.action(i, Turn::White);
  // }
  // // board.action(1, Turn::Black);
  // // board.action(2, Turn::Black);
  // // board.action(3, Turn::Black);
  // // board.action(5, Turn::Black);
  // // board.action(6, Turn::Black);
  // // board.action(7, Turn::Black);
  // // board.action(8, Turn::Black);
  // // board.action(9, Turn::Black);
  // // board.action(11, Turn::White);
  // // board.action(12, Turn::White);
  // // board.action(13, Turn::White);
  // // board.action(14, Turn::White);
  // // board.action(17, Turn::White);
  // // board.action(18, Turn::White);
  // // board.action(19, Turn::White);
  // // board.action(23, Turn::White);
  // // board.action(24, Turn::White);
  // board.turn = Turn::White;
  // println!("{:}", &board);
  // let predict32 = |inputs: Vec<Vec<f32>>| {
  //   NNet::predict32(&net, inputs)
  // };
  // let temp = 1.0;
  // let probs = mcts.get_action_prob(&board, temp, &predict32, false, true, false, 1);
  // println!("probs {:?}", probs);


  // let board_size: i64 = 7;
  // let action_size: i64 = 50;
  // let num_channels: i64 = 32;
  // let mut net = NNet::new(board_size, action_size, num_channels);
  // net.load_trainable("7x7/best_pass.pt");
  // let mut mcts = MCTS::new(30, 1.0);
  // let mut board = Board::new(BoardSize::S7);
  // // for i in 0..46 {
  // //   board.action(i, Turn::Black);
  // // }
  // board.action(24, board.turn);
  // println!("{:}", &board);
  // //board.action(49, board.turn);
  // // board.action(1, board.turn);
  // // board.action(0, board.turn);
  // // board.action(7, board.turn);
  // // board.action(49, board.turn);
  // let predict32 = |inputs: Vec<Vec<f32>>| {
  //   NNet::predict32(&net, inputs)
  // };
  // let temp = 0.5;
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
