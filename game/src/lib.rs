pub mod igo;
pub use igo::*;
extern crate console_error_panic_hook;
use std::panic;
use js_sys::{Promise, Float32Array, Number};
use std::rc::Rc;
use std::cell::{RefCell, RefMut};

extern crate wasm_bindgen;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;
use wasm_bindgen_futures::JsFuture;

static mut BOARDS: Option<Rc<RefCell<Vec<(Vec<f32>, f32)>>>> = None;

pub fn init_boards() {
  unsafe {
    BOARDS = Some(Rc::new(RefCell::new(Vec::new())));
  }
}

pub fn boards_mut() -> RefMut<'static, Vec<(Vec<f32>, f32)>> {
  unsafe { BOARDS.as_ref().unwrap().borrow_mut() }
}

#[wasm_bindgen(module="/src/jspredict.js")]
extern {
  type JSBoard;
  fn jspredict(input: Float32Array) -> Float32Array;
}

fn get_board(board_size: &Number, stones: &Float32Array, turn: &Number, pass_cnt: &Number) -> Board {
  let bsize = match board_size.value_of() as u32 {
    5 => BoardSize::S5,
    7 => BoardSize::S7,
    _ => BoardSize::S9,
  };
  let mut board = Board::new(bsize);
  board.set_vec(stones.to_vec());
  if turn.value_of() < -0.1 {
    board.turn = Turn::White;
  }
  board.pass_cnt = pass_cnt.value_of() as u32;
  board
}

#[wasm_bindgen]
pub fn run(board_size: Number, stones: Float32Array, turn: Number, pass_cnt: Number) -> Float32Array {
  panic::set_hook(Box::new(console_error_panic_hook::hook));
  let board = get_board(&board_size, &stones, &turn, &pass_cnt);
  let mut mcts = MCTS::new(200, 1.0); // reset search tree
  fn predict(inputs: Vec<Vec<f32>>) -> Vec<(Vec<f32>, f32)> {
    let len = inputs.len();
    let mut asize:usize = 0;
    let mut input = Vec::new();
    for row in inputs {
      asize = row.len() / 12 + 1;
      input.extend(row);
    }
    let jsoutput = jspredict(Float32Array::from(&input[..]));
    rspredict(jsoutput, len, asize)
    // let future = JsFuture::from(jsoutput);
    // init_boards();
    // spawn_local(rspredict(future, len, asize));
    // let res = boards_mut().to_vec();
    // panic!("{:?}", boards_mut());
    // res
  }
  let temp = 1.0;
  let pi = mcts.get_action_prob(&board, temp, &predict);
  let pijs = Float32Array::from(&pi[..]);
  return pijs;
}

#[wasm_bindgen]
pub fn action(board_size: Number, stones: Float32Array, turn: Number, pass_cnt: Number, action: Number) -> Float32Array {
  panic::set_hook(Box::new(console_error_panic_hook::hook));
  let mut board = get_board(&board_size, &stones, &turn, &pass_cnt);
  board.action(action.value_of() as u32, board.turn);
  let mut resp = board.to_vec();
  resp.push(board.turn as i32 as f32);
  resp.push(board.pass_cnt as f32);
  Float32Array::from(&resp[..])
}

#[wasm_bindgen]
pub fn game_ended(board_size: Number, stones: Float32Array, turn: Number, pass_cnt: Number) -> Number {
  panic::set_hook(Box::new(console_error_panic_hook::hook));
  let board = get_board(&board_size, &stones, &turn, &pass_cnt);
  Number::from(board.game_ended())
}

pub fn rspredict(board: Float32Array, len: usize, asize: usize) -> Vec<(Vec<f32>, f32)> {
  let mut result = Vec::new();
  let output = board.to_vec();
  for i in 0..len {
    let pi = output[i*(asize+1)..((i+1)*(asize+1)-1)].to_vec();
    let v = output.get((i+1)*(asize+1)-1).unwrap();
    result.push((pi, *v));
  }
  // boards_mut().extend(result);
  result
}
