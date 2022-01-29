pub mod igo;
pub use igo::*;

fn main() {
  // let mut x: u32 = 0b11111_11111_11111_00000_00000;
  // let mut t: u32;
  // const K1: u32 = 0b11000_11000_00000_00000_00000;
  // const K2: u32 = 0b00100_00000_10000_00000_00000;
  // const K3: u32 = 0b10010_00100_01000_10010_00000;
  // t  = K1 & (x ^ (x << 18));
  // x ^=       t ^ (t >> 18) ;
  // t  = K2 & (x ^ (x << 12));
  // x ^=       t ^ (t >> 12) ;
  // t  = K3 & (x ^ (x <<  6));
  // x ^=       t ^ (t >>  6) ;
  // println!("{:b}", x);
  let board = Board::new(BoardSize::S7);
  let mut mcts = MCTS::new(4, 1.0); // reset search tree
  let predict = |inputs: Vec<Vec<f32>>| -> Vec<(Vec<f32>, f32)> {
    let len = inputs.len();
    let asize = 50;
    let mut input = Vec::new();
    for row in inputs {
      input.extend(row);
    }
    let mut result = Vec::new();
    let output: Vec<f32> = vec![1.0; 51*1];
    for i in 0..len {
      let pi = output[i*(asize+1)..((i+1)*(asize+1)-1)].to_vec();
      let v = output.get((i+1)*(asize+1)-1).unwrap();
      result.push((pi, *v));
    }
    result
  };
  let temp = 1.0;
  let pi = mcts.get_action_prob(&board, temp, &predict, false, false, false, 0);
  // let pi: Vec<f32> = vec![1.0; 26];
  println!("{:?}", &pi[..]);
}
