use alphagozero_rs::igo::board;

fn main() {
  println!("{}", board::test());
  let mut board = board::Board::new(board::BoardSize::S5);
  let b = board::Turn::Black;
  let w = board::Turn::White;
  board.action_xy(3, 3, b);
  board.action_xy(3, 4, b);
  board.action_xy(4, 4, b);
  board.action_xy(2, 3, w);
  board.action_xy(4, 3, w);
  board.action_xy(3, 2, w);
  board.action_xy(2, 4, w);
  board.action_xy(5, 4, w);
  board.action_xy(3, 5, w);
  board.action_xy(4, 5, w);

  board.action_xy(1, 1, b);
  board.action_xy(2, 1, w);
  board.action_xy(1, 2, w);

  board.action_xy(1, 3, b);
  board.action_xy(1, 4, w);

  board.action_xy(5, 5, b);
  board.action_xy(5, 1, b);
  board.remove_death_stones(b);
  println!("{}", board);
}