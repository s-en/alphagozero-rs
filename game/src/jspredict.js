let nncache = null;
const BSIZE = 5;

(async function () {
  await tf.ready();
  // console.log(`back ${tf.getBackend()}`);
  nncache = await tf.loadGraphModel('/assets/jsmodel/5x5/model.json');
  // warm-up
  const zeros = tf.zeros([16, 12, BSIZE, BSIZE]);
  const prods = nncache.execute({
    'x_1:0': zeros
  });
  prods.dataSync();
  postMessage({msg: (await tf.getBackend())});
  postMessage({isReady: true});
}());

export function jspredict(inputs) {
  // console.log("jspredict");
  const len = inputs.length / (12 * BSIZE * BSIZE);
  //console.log(inputs);
  let tmax = 16 * 12 * BSIZE * BSIZE;
  let tinputs = [...inputs, ...Array(tmax).fill(0)]; // fill zero for tail data
  tinputs = tinputs.slice(0, tmax);
  const boardSize = [16, 12, BSIZE, BSIZE];
  if (!nncache) {
    console.error('model must be loaded first');
    return [];
  }
  const nnet = nncache;
  const prods = tf.tidy(() => {
    const prediction = nnet.execute({
      'x_1:0': tf.tensor(tinputs).reshape(boardSize)
    });
    return prediction;
  });
  const result = prods.dataSync();
  prods.dispose();
  return result.slice(0, (BSIZE * BSIZE + 2) * len);
};

export class JSBoard {
  constructor(stones, turn, pass_cnt) {
    this.stones = stones;
    this.turn = turn;
    this.pass_cnt = pass_cnt;
  }
};

