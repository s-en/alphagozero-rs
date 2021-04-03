
use alphagozero_rs::igo::*;

#[cfg(test)]
mod board {
  use super::*;

  #[test]
  fn remove_death_stones_5x5() {
      let mut board = Board::new(BoardSize::S5);
      let b = Turn::Black;
      let w = Turn::White;
      let tb = Stones::new32(0b00000_00000_00000_00000_10000);
      let tw = Stones::new32(0b01100_10011_01010_00101_00010);
      board.set_stones(b, tb);
      board.set_stones(w, tw);
      board.remove_death_stones(b);
      assert_eq!(
        format!("{}", board),
        "012345\n\
        1-x--o\n\
        2x-x--\n\
        3-x-x-\n\
        4xx--x\n\
        5--xx-\n"
      );
  }
  #[test]
  fn valid_moves_5x5() {
    let mut board = Board::new(BoardSize::S5);
    let b = Turn::Black;
    let w = Turn::White;
    let tb = Stones::new32(0b10100_01000_00010_00101_01010);
    let tw = Stones::new32(0b00000_10100_11101_01000_10000);
    board.set_stones(b, tb);
    board.set_stones(w, tw);
    let v = board.valid_moves(w);
    assert_eq!(
      format!("{}", v),
      "--o--\n\
       ----o\n\
       -----\n\
       oo---\n\
       oo-o-\n"
    );
  }
  #[test]
  fn check_kou() {
    let mut board = Board::new(BoardSize::S5);
    let b = Turn::Black;
    let w = Turn::White;
    let tb = Stones::new32(0b01000_10000_00000_00000_00000);
    let tw = Stones::new32(0b00100_01000_00000_00000_00000);
    board.set_stones(b, tb);
    board.set_stones(w, tw);

    board.action_xy(5, 5, w);
    let v = board.valid_moves(b);
    assert_eq!(
      format!("{}", v),
      "ooooo\n\
       ooooo\n\
       ooooo\n\
       ooo--\n\
       oo---\n"
    );
    board.action_xy(1, 1, b);
    let v = board.valid_moves(b);
    assert_eq!(
      format!("{}", v),
      "-oooo\n\
       ooooo\n\
       ooooo\n\
       ooo--\n\
       oo-o-\n"
    );
  }
}