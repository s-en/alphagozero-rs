importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs');
let nncache = null;

(async function () {
  nncache = await tf.loadGraphModel('/assets/jsmodel/5x5/model.json');
}());

export function jspredict(inputs) {
  console.log("jspredict");
  const BSIZE = 5;
  const len = inputs.length / (12 * BSIZE * BSIZE);
  console.log(inputs);
  let tmax = 16 * 12 * BSIZE * BSIZE;
  let tinputs = [...inputs, ...Array(tmax).fill(0)]; // fill zero for tail data
  tinputs = tinputs.slice(0, tmax);
  const boardSize = [16, 12, BSIZE, BSIZE];
  if (!nncache) {
    console.error('model must be loaded first');
  }
  const nnet = nncache;
  const prediction = nnet.execute({
    'x_1:0': tf.tensor(tinputs).reshape(boardSize)
  });
  const result = prediction.dataSync();
  return result.slice(0, (BSIZE * BSIZE + 2) * len);
};

export class JSBoard {
  constructor(stones, turn, pass_cnt) {
    this.stones = stones;
    this.turn = turn;
    this.pass_cnt = pass_cnt;
  }
};

