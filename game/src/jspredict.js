let nncache = null;
let BSIZE = 5;

(async function () {
  console.log('p loading');
  const queryString = window.location.search;
  const urlParams = new URLSearchParams(queryString);
  const size = urlParams.get('size');
  BSIZE = 5;
  if(size && ['5', '7'].includes(size)){
    BSIZE = Number(size);
  }
  if(global.tf){
    // console.log(`back ${tf.getBackend()}`);
    await tf.ready();
    nncache = await tf.loadGraphModel(`/assets/jsmodel/${BSIZE}x${BSIZE}/model.json`);
    console.log('p loaded');
  
    // warm-up
    let simCnt = 8;
    if(BSIZE === 5) simCnt = 16;
    const zeros = tf.zeros([simCnt, 12, BSIZE, BSIZE]);
    const prods = nncache.execute({
      'x_1:0': zeros
    });
    prods.dataSync();
  }
  window.jspredictLoaded = true;
  // postMessage({isReady: true});
}());

export function jspredict(inputs) {
  // console.log("jspredict");
  const len = inputs.length / (12 * BSIZE * BSIZE);
  //console.log(inputs);
  let simCnt = 8;
  if(BSIZE === 5) simCnt = 16;
  let tmax = simCnt * 12 * BSIZE * BSIZE;
  let tinputs = [...inputs, ...Array(tmax).fill(0)]; // fill zero for tail data
  tinputs = tinputs.slice(0, tmax);
  const reshaped = [];
  const boardSize = [simCnt, 12, BSIZE, BSIZE];
  for(let a=0; a<simCnt; a++){
    reshaped[a] = reshaped[a] || [];
    for(let b=0; b<12; b++){
      reshaped[a][b] = reshaped[a][b] || [];
      for(let c=0; c<BSIZE; c++){
        const idx = BSIZE*c + BSIZE*BSIZE*b + BSIZE*BSIZE*12*a;
        reshaped[a][b][c] = tinputs.slice(idx, idx+BSIZE);
      }
    }
  }
  if (!nncache) {
    console.error('model must be loaded first');
    return [];
  }
  
  const nnet = nncache;
  const prods = tf.tidy(() => {
    const prediction = nnet.execute({
      'x_1:0': tf.tensor(reshaped)
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

