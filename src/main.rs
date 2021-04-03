use alphagozero_rs::igo::*;

fn main() {
  let mut board = Board::new(BoardSize::S5);
  let b = Turn::Black;
  let w = Turn::White;
  let tb = Stones::new32(0b01000_10000_00000_00000_00000);
  let tw = Stones::new32(0b00100_01000_00000_00000_00000);
  board.set_stones(b, tb);
  board.set_stones(w, tw);

  println!("{}", board);
  board.action_xy(5, 5, w);
  board.action_xy(1, 1, b);
  println!("{}", board);
  let v = board.valid_moves(b);
  println!("{}", v);

  let b = Stones::new64(0b0000000_1111111_111, 0b1111_1111111_1111111_1111111_1111110);
  println!("{}", b + 1);
}
// fn main() {
//   let mut board = board::Board::new(board::BoardSize::S5);
//   let b = board::Turn::Black;
//   let w = board::Turn::White;
//   board.action_xy(3, 3, b);
//   board.action_xy(3, 4, b);
//   board.action_xy(4, 4, b);
//   board.action_xy(2, 3, w);
//   board.action_xy(4, 3, w);
//   board.action_xy(3, 2, w);
//   board.action_xy(2, 4, w);
//   board.action_xy(5, 4, w);
//   board.action_xy(3, 5, w);
//   board.action_xy(4, 5, w);

//   board.action_xy(1, 1, b);
//   board.action_xy(2, 1, w);
//   board.action_xy(1, 2, w);

//   board.action_xy(1, 3, b);
//   board.action_xy(1, 4, w);

//   board.action_xy(5, 5, b);
//   board.action_xy(5, 1, b);
//   println!("{}", board);
//   board.remove_death_stones(b);
//   println!("{}", board);
//   board.valid_moves(w);
// }