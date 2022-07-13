pub mod igo;
pub use igo::*;
extern crate console_error_panic_hook;
use std::panic;
use js_sys::{Float32Array, Number, Boolean, Promise};
use std::cmp::Ordering;

extern crate wasm_bindgen;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(module="/src/jspredict.js")]
extern {
  type JSBoard;
  fn jspredict(input: Float32Array) -> Promise;
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

pub fn max_idx(vals: &Vec<f32>) -> usize {
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

#[wasm_bindgen]
pub async fn run(board_size: Number, stones: Float32Array, turn: Number, pass_cnt: Number, sim_num: u32) -> Float32Array {
  panic::set_hook(Box::new(console_error_panic_hook::hook));
  let board = get_board(&board_size, &stones, &turn, &pass_cnt);
  let mut mcts = MCTS::new(sim_num, 1.0); // reset search tree
  async fn predict(inputs: Vec<Vec<f32>>) -> Vec<(Vec<f32>, f32)> {
    let len = inputs.len();
    let mut asize:usize = 0;
    let mut input = Vec::new();
    for row in inputs {
      asize = row.len() / 12 + 1;
      input.extend(row);
    }
    let promise = jspredict(Float32Array::from(&input[..]));
    let jsoutput:Float32Array = wasm_bindgen_futures::JsFuture::from(promise).await.unwrap().into();
    rspredict(jsoutput, len, asize)
  }
  let mut temp = 0.2;
  let for_train = false;
  let self_play = false;
  let prioritize_kill = false;
  let mut komi = -1;
  if board_size == 7 {
    temp = 0.2;
  }
  let stoneCnt = stones.to_vec().iter().fold(0.0, |sum, a| sum + a.abs());
  if stoneCnt < 2.0 {
    // 最初の２手は分散大きく
    temp = 0.4;
  }
  if stoneCnt > board_size.value_of() as f32 * 2.0 {
    // 後半は分散小さく
    temp = 0.1;
    if stoneCnt > board_size.value_of() as f32 * 4.0 {
      temp = 0.0;
    }
  }
  let mut pi = mcts.get_action_prob_async(&board, temp, &predict, prioritize_kill, for_train, self_play, komi).await;
  let s = board.calc_hash();
  // let valids = board.vec_valid_moves_for_cpu(board.turn);
  // let mut masked_pi: Vec<f32> = valids.iter().enumerate().map(|(i, x)| *x as i32 as f32 * pi[i]).collect();
  //panic!("pi {:?}", pi);
  let best_action = max_idx(&pi);
  let sa = (s, best_action);
  if mcts.qsa.contains_key(&sa) {
    pi.push(mcts.qsa[&sa]);
  } else {
    pi.push(0.0);
  }
  let pijs = Float32Array::from(&pi[..]);
  return pijs;
}

#[wasm_bindgen]
pub async fn playout_killed(board_size: Number, stones: Float32Array, turn: Number, pass_cnt: Number, max_play: Number) -> Float32Array {
  panic::set_hook(Box::new(console_error_panic_hook::hook));
  let mut board = get_board(&board_size, &stones, &turn, &pass_cnt);
  let sim_num = 3;
  let mut mcts = MCTS::new(sim_num, 1.0); // reset search tree
  async fn predict(inputs: Vec<Vec<f32>>) -> Vec<(Vec<f32>, f32)> {
    let len = inputs.len();
    let mut asize:usize = 0;
    let mut input = Vec::new();
    for row in inputs {
      asize = row.len() / 12 + 1;
      input.extend(row);
    }
    let promise = jspredict(Float32Array::from(&input[..]));
    let jsoutput:Float32Array = wasm_bindgen_futures::JsFuture::from(promise).await.unwrap().into();
    rspredict(jsoutput, len, asize)
  }
  let temp = 0.0;
  let for_train = false;
  let self_play = false;
  let prioritize_kill = true;
  let komi = 0;
  let mut play_cnt = 0;
  let black_kill = board.kill_point(Turn::Black);
  let white_kill = board.kill_point(Turn::White);
  let mut black_stones = black_kill & 0;
  let mut white_stones = white_kill & 0;
  // アタリを殺した盤面を再現
  board.set_stones(Turn::Black, board.black | black_kill);
  board.remove_death_stones(Turn::White);
  white_stones = white_stones | board.white;
  board = get_board(&board_size, &stones, &turn, &pass_cnt);
  board.set_stones(Turn::White, board.white | white_kill);
  board.remove_death_stones(Turn::Black);
  black_stones = black_stones | board.black;
  board = get_board(&board_size, &stones, &turn, &pass_cnt);
  // 死んだ石を算出
  white_stones = white_stones ^ board.white;
  black_stones = black_stones ^ board.black;
  // 数手進めて死んだ石を算出
  while board.game_ended(false, komi) == 0 && play_cnt < max_play.value_of() as u32 {
    let prev_white = board.white;
    let prev_black = board.black;
    let pi = mcts.get_action_prob_async(&board, temp, &predict, prioritize_kill, for_train, self_play, 0).await;
    let action = max_idx(&pi) as u32;
    board.action(action, board.turn);
    play_cnt += 1;
    if board.turn == Turn::White {
      white_stones = white_stones | (prev_white ^ board.white);
    } else {
      black_stones = black_stones | (prev_black ^ board.black);
    }
  }
  let mut res = black_stones.vec();
  res.append(&mut white_stones.vec());
  return Float32Array::from(&res[..]);
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
  Number::from(board.game_ended(false, 0))
}

#[wasm_bindgen]
pub fn is_valid_move(board_size: Number, stones: Float32Array, turn: Number, pass_cnt: Number, action: Number) -> Boolean {
  panic::set_hook(Box::new(console_error_panic_hook::hook));
  let board = get_board(&board_size, &stones, &turn, &pass_cnt);
  let valids = board.vec_valid_moves(board.turn);
  Boolean::from(valids[action.value_of() as usize])
}

#[wasm_bindgen]
pub fn kou_cnt(board_size: Number, stones: Float32Array, turn: Number, pass_cnt: Number, action: Number) -> Number {
  panic::set_hook(Box::new(console_error_panic_hook::hook));
  let board = get_board(&board_size, &stones, &turn, &pass_cnt);
  let kou_cnt = board.kou_cnt(action.value_of() as u32, board.turn);
  Number::from(kou_cnt)
}

pub fn rspredict(board: Float32Array, len: usize, asize: usize) -> Vec<(Vec<f32>, f32)> {
  let mut result = Vec::new();
  let output = board.to_vec();
  for i in 0..len {
    let pi = output[i*(asize+1)..((i+1)*(asize+1)-1)].to_vec();
    let v = output.get((i+1)*(asize+1)-1).unwrap();
    result.push((pi, *v));
  }
  result
}
