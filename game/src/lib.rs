pub mod igo;

pub use igo::*;

pub extern fn test() {
  let mut board = Board::new(BoardSize::S5);
  let b = Turn::Black;
  let w = Turn::White;
  let tb = Stones::new32(0b00000_00000_00000_00000_10000);
  let tw = Stones::new32(0b01100_10011_01010_00101_00010);
  board.set_stones(b, tb);
  board.set_stones(w, tw);
  board.remove_death_stones(b);
  println!("{}", board);
  //func();
  //println!("{}", add_two(2.3));
  // unsafe {
  //   atest(7);
  //   println!("inside unsafe");
  // }
}

// #[no_mangle]
// pub extern fn register_predict<F: Fn(i32) -> i32>(pd: F) -> i32 {
//   println!("register predict");
//   let back: i32 = pd(5);
//   println!("back {}", back);
//   3
// }
