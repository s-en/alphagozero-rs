pub mod igo;
pub use igo::*;
extern crate console_error_panic_hook;
use std::panic;

extern crate wasm_bindgen;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(module="/src/jspredict.js")]
extern {
  fn jspredict(input: js_sys::Float32Array) -> js_sys::Float32Array;
}

// fn convert_to_js(inputs: Vec<(Vec<f32>, f32)>) -> {

// }

#[wasm_bindgen]
pub fn run() -> js_sys::Float32Array {
  panic::set_hook(Box::new(console_error_panic_hook::hook));
  let board = Board::new(BoardSize::S5);
  let mut mcts = MCTS::new(200, 1.0); // reset search tree
  let predict = |inputs: Vec<Vec<f32>>| -> Vec<(Vec<f32>, f32)> {
    let len = inputs.len();
    let asize = 26;
    let mut input = Vec::new();
    for row in inputs {
      input.extend(row);
    }
    let mut result = Vec::new();
    let jsoutput = jspredict(js_sys::Float32Array::from(&input[..]));
    let output = jsoutput.to_vec();
    for i in 0..len {
      let pi = output[i*(asize+1)..((i+1)*(asize+1)-1)].to_vec();
      let v = output.get((i+1)*(asize+1)-1).unwrap();
      result.push((pi, *v));
    }
    result
  };
  let temp = 0.2;
  let pi = mcts.get_action_prob(&board, temp, &predict);
  let pijs = js_sys::Float32Array::from(&pi[..]);
  return pijs;
}

// pub extern fn test() {
//   let mut board = Board::new(BoardSize::S5);
//   let b = Turn::Black;
//   let w = Turn::White;
//   let tb = Stones::new32(0b00000_00000_00000_00000_10000);
//   let tw = Stones::new32(0b01100_10011_01010_00101_00010);
//   board.set_stones(b, tb);
//   board.set_stones(w, tw);
//   board.remove_death_stones(b);
//   println!("{}", board);
//   //func();
//   //println!("{}", add_two(2.3));
//   // unsafe {
//   //   atest(7);
//   //   println!("inside unsafe");
//   // }
// }

// #[no_mangle]
// pub extern fn register_predict<F: Fn(i32) -> i32>(pd: F) -> i32 {
//   println!("register predict");
//   let back: i32 = pd(5);
//   println!("back {}", back);
//   3
// }
