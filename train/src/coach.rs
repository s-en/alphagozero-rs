use super::*;

use rand::distributions::WeightedIndex;
use std::collections::VecDeque;
use indicatif::ProgressIterator;

extern crate savefile;

fn predict<'a>(net: &'a NNet) -> Box<dyn Fn(Vec<f32>) -> (Vec<f32>, f32) + 'a> {
  Box::new(move |board| -> (Vec<f32>, f32) {
    NNet::predict(net, board)
  })
}

impl Coach {
  pub fn execute_episode(&mut self, mcts: &mut MCTS, net: &NNet) -> Vec<Example> {
    let mut examples = Vec::new();
    let mut board = Board::new(BoardSize::S5);
    let mut episode_step = 0;
    let temp_threshold = 5 * 2;
    //let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([42; 32]);
    let mut kcnt = 0;
    loop {
      episode_step += 1;
      let mut temp = 0.0;
      if episode_step < temp_threshold {
        temp = 1.0;
      }
      let pi = mcts.get_action_prob(&board, temp, &predict(net));
      let dist = WeightedIndex::new(&pi).unwrap();
      let sym = board.symmetries(pi);
      for (b, p) in sym {
        let ex = Example {
          board: b,
          pi: p,
          v: 0.0
        };
        examples.push(ex);
      }
      let action = dist.sample(&mut self.rng) as u32;
      //println!("action {}", action);
      board.action(action, board.turn);
  
      let r = board.game_ended();
  
      if r != 0 {
        // if kcnt == 0 {
        //   kcnt += 1;
        // }
        // println!("{:?}", board.count_diff());
        // println!("{}", r);
        // println!("{}", board);
        println!("{} {}", r, board.get_kifu_sgf());
        println!("");
        let v = r as i32;
        for mut ex in &mut examples {
          // v: 1.0 black won
          // v: -1.0 white won
           ex.v = v as f32;
        }
        //println!("{:?}", examples);
        return examples;
      }
    }
  }
  
  pub fn learn(&mut self) {
    let num_iters = 50;
    let num_eps = 10;
    let maxlen_of_queue = 20000;
    let num_iters_for_train_examples_history = 50000;
    let update_threshold = 0.55;
    let skip_first_self_play = false;
    let mut train_examples_history = Examples {
      values: VecDeque::with_capacity(num_iters_for_train_examples_history)
    };
  
    let board_size: i64 = 5;
    let action_size: i64 = 26;
    let num_channels: i64 = 32;
    let mut net = NNet::new(board_size, action_size, num_channels);
    let mut pnet = NNet::new(board_size, action_size, num_channels);
    // net.load("temp/best.pt");
    // pnet.load("temp/best.pt");
  
    for i in 1..(num_iters + 1) {
      println!("Starting Iter #{} ...", i);
      //if skip_first_self_play || i > 1 {
      {
        let mut iteration_train_examples = Examples {
          values: VecDeque::with_capacity(maxlen_of_queue)
        };
        println!("self playing...");
        for _ in (0..num_eps).progress() {
          let mut mcts = MCTS::new(100); // reset search tree
          let examples = self.execute_episode(&mut mcts, &net);
          iteration_train_examples.values.extend(examples);
        }
  
        // save the iteration examples to the history 
        train_examples_history.values.extend(iteration_train_examples.values);
      }
      // backup history to a file
      // NB! the examples were collected using the model from the previous iteration, so (i-1)
      // save_file(&format!("temp/train_examples_{}.bin", i - 1), 0, &train_examples_history).unwrap();
  
      // shuffle examples before training
      let mut train_examples = Vec::new();
      for e in &train_examples_history.values {
        train_examples.push(e);
      }
      // println!("{:?}", train_examples[73]);
      // return;
      let vs = train_examples.iter().map(|x| x.v).collect::<Vec<f32>>();
      //println!("{:?}", vs);
      println!("won {:?}", vs.iter().filter(|&x| x > &0.0).sum::<f32>());
      println!("lose {:?}", vs.iter().filter(|&x| x < &0.0).sum::<f32>());
      train_examples.shuffle(&mut self.rng);
  
      // training new network, keeping a copy of the old one
      net.save("temp/temp.pt");
      pnet.load("temp/temp.pt");
      let mut pmcts = MCTS::new(3);
      let mut nmcts = MCTS::new(3);
  
      let mut board = Board::new(BoardSize::S5);
      let pi = NNet::predict(&net, board.input());
      println!("before {:?}", pi);
      //println!("{:?}", train_examples);
      // for row in &train_examples {
      //   println!("{:?}", &row.board[25..50]);
      //   println!("{:?}", row.v);
      // }

      net.train(train_examples);

      let pi = NNet::predict(&net, board.input());
      println!("after {:?}", pi);
      board.action(3, Turn::Black);
      //board.turn = Turn::White;
      let pi = NNet::predict(&net, board.input());
      println!("white turn {:?}", pi);
  
      println!("PITTING AGAINST PREVIOUS VERSION");
      let temp = 0.0;
      
      let game_result;
      {
        let mut player1 = self.player(&mut pmcts, &pnet, temp);
        let mut player2 = self.player(&mut nmcts, &net, temp);
        game_result = self.play_games(400, &mut player1, &mut player2);
      }
      let (pwins, nwins, draws) = game_result;
  
      println!("NEW/PREV WINS : {} / {} ; DRAWS : {}", nwins, pwins, draws);
      if pwins + nwins == 0 || nwins * 100 / (pwins + nwins) < (update_threshold * 100.0) as u32 {
        println!("REJECTING NEW MODEL");
        net.load("temp/temp.pt");
      } else {
        println!("ACCEPTING NEW MODEL");
        net.save(format!("temp/checkpoint_{}.pt", i));
        net.save("temp/best.pt");
      }
    }
  }
  
  pub fn play_game<F: FnMut(&mut Board) -> usize>(&self, player1: &mut F, player2: & mut F, rep: u32) -> i8 {
    let players = [player2, player1];
    let mut cur_player = 1;
    let mut board = Board::new(BoardSize::S5);
    let mut it = 0;
    while board.game_ended() == 0 {
      it += 1;
      let action = players[cur_player](&mut board);
  
      let valids = board.vec_valid_moves(board.turn);
      if !valids[action] {
        println!("Action {} is not valid!", action);
        println!("valids = {:?}", valids);
        println!("{}", board.get_kifu_sgf());
        assert!(valids[action]);
      }
      // println!("{}", board.valid_moves(board.turn));
      // if rep == 0 {
      //   println!("Turn {} Player {} action {} {}", it, board.turn as i32, action % 5 + 1, action / 5 + 1);
      //   println!("{}", board);
      // }
      board.action(action as u32, board.turn);
      // println!("Turn {} Player {} action {} {}", it, board.turn as i32, action % 5 + 1, action / 5 + 1);
      cur_player = (board.turn as i32 + 1) as usize / 2;
    }
    if rep == 0 {
      println!("{}", board.get_kifu_sgf());
      println!("");
    }
    board.game_ended()
  }
  pub fn play_games<F: FnMut(&mut Board) -> usize>(&self, num: u32, player1: &mut F, player2: &mut F) -> (u32, u32, u32) {
    let num = num / 2;
    let mut one_won = 0;
    let mut two_won = 0;
    let mut draws = 0;
  
    println!("Arena.playGames (1)");
    for i in (0..num).progress() {
      let game_result = self.play_game(player1, player2, i);
      match game_result {
        1 => one_won += 1,
        -1 => two_won += 1,
        _ => draws += 1
      }
    }
  
    println!("Arena.playGames (2)");
    for i in (0..num).progress() {
      let game_result = self.play_game(player2, player1, i);
      match game_result {
        -1 => one_won += 1,
        1 => two_won += 1,
        _ => draws += 1
      }
    }
  
    (one_won, two_won, draws)
  }
  fn player<'a>(&self, _mcts: &'a mut MCTS, _net: &'a NNet, temp: f32) -> Box<dyn FnMut(&mut Board) -> usize + 'a> {
    let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([42; 32]);
    Box::new(move |x: &mut Board| -> usize {
      if temp >= 1.0 {
        let mut valids: Vec<usize> = x.vec_valid_moves(x.turn).iter().enumerate().map(|(i, &val)| {
          if val { i } else { 10000 }
        }).filter(|&r| r < 10000).collect();
        valids.shuffle(&mut rng);
        // println!("{:?}", valids);
        valids[0]
      } else {
        let probs = _mcts.get_action_prob(&x, temp, &predict(_net));
        //println!("{:?}", probs);
        mcts::max_idx(&probs)
      }
    })
  }
}