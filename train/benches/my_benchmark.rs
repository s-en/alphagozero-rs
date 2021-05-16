#[macro_use]
extern crate criterion;

use criterion::Criterion;
use criterion::black_box;
use az_game::igo::*;
use az_train::*;

fn fibonacci(n: u64) -> u64 {
  let mut a = 0;
  let mut b = 1;

  match n {
      0 => b,
      _ => {
          for _ in 0..n {
              let c = a + b;
              a = b;
              b = c;
          }
          b
      }
  }
}

fn self_play() {

}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("fib 20", |b| b.iter(|| fibonacci(black_box(20))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
