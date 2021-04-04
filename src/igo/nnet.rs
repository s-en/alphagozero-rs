use super::*;

impl NNet {
  pub fn new() -> NNet{
    NNet {

    }
  }
  pub fn predict(&self) -> (Vec<f32>, i8) {
    (vec![0.0; 26], 0.0 as i8)
  }
}