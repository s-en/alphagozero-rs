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
  if(ort){
    window.jspredictError = false;
    let status = 0;
    let errStr = '';
    setTimeout(() => {
      // 5秒たっても表示されなければエラーを出す
      if(!window.jspredictLoaded) {
        window.jspredictError = 'エラー: InferenceSession.createに失敗しています st:' + status + ' ' + errStr;
      }
    }, 5000)
  
    // warm-up
    try {
      nncache = await ort.InferenceSession.create('assets/jsmodel/7x7/onnx/dualnet.onnx');
      status = 1;
      console.log('quant loaded');
      let simCnt = 8;
      const zeros = Float32Array.from(Array(simCnt*12*BSIZE*BSIZE).fill(0.0));
      const tensorA = new ort.Tensor('float32', zeros, [simCnt, 12, BSIZE, BSIZE]);
      const feeds = { 
        'x.1': tensorA
      };
      status = 2;
      const results = await nncache.run(feeds);
      status = 3;
      console.log(results);
    } catch (e) {
      errStr = e.toString();
      throw e;
    }
  }
  window.jspredictLoaded = true;
  // postMessage({isReady: true});
}());

const delay = milliseconds => new Promise(resolve, setTimeout(resolve, milliseconds));

export async function jspredict(inputs) {
  // console.log("jspredict");
  const len = inputs.length / (12 * BSIZE * BSIZE);
  //console.log(inputs);
  let simCnt = 8;
  let tmax = simCnt * 12 * BSIZE * BSIZE;
  let tinputs = [...inputs,...inputs,...inputs,...inputs,...inputs,...inputs,...inputs,...inputs, ...Array(tmax).fill(0.0)]; // fill zero for tail data
  tinputs = tinputs.slice(0, tmax);
  // const reshaped = [];
  // const boardSize = [simCnt, 12, BSIZE, BSIZE];
  // for(let a=0; a<simCnt; a++){
  //   reshaped[a] = reshaped[a] || [];
  //   for(let b=0; b<12; b++){
  //     reshaped[a][b] = reshaped[a][b] || [];
  //     for(let c=0; c<BSIZE; c++){
  //       const idx = BSIZE*c + BSIZE*BSIZE*b + BSIZE*BSIZE*12*a;
  //       reshaped[a][b][c] = [...Array(BSIZE).fill(1.0)];//tinputs.slice(idx, idx+BSIZE);
  //     }
  //   }
  // }
  if (!nncache) {
    console.error('model must be loaded first');
    return [];
  }
  // console.log(tinputs);
  // console.log(reshaped.flat(3))
  const nnet = nncache;
  const tensorA = new ort.Tensor('float32', tinputs, [simCnt, 12, BSIZE, BSIZE]);
  const feeds = { 
    'x.1': tensorA
  };
  let result = await nnet.run(feeds);
  // console.log(tinputs);
  // console.log('result');
  // console.log(result[279].data);
  result = [...result[279].data];
  //console.log(result);
  // let bef = -1;
  // for (let i=0; i<1000; i+=100) {
  //   if (bef >= 0) {
  //     tinputs[bef] = 1.0;
  //   }
  //   tinputs[i] = 0.0;
  //   const test = new ort.Tensor('float32', tinputs, [simCnt, 12, BSIZE, BSIZE]);
  //   const feedtest = { 
  //     'x.1': test
  //   };
  //   let rtest = await nnet.run(feedtest);
  //   const r = [...rtest[279].data];
  //   console.log(`${i}: ${r[0]}`);
  //   bef = i;
  // }

  return result.slice(0, (BSIZE * BSIZE + 2) * len);
};

export class JSBoard {
  constructor(stones, turn, pass_cnt) {
    this.stones = stones;
    this.turn = turn;
    this.pass_cnt = pass_cnt;
  }
};

