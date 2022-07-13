let nncache = null;
let BSIZE = 5;
let LV = 2;

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
      nncache = await ort.InferenceSession.create(`assets/jsmodel/${BSIZE}x${BSIZE}/onnx/lv${LV}.onnx`);
      status = 1;
      console.log('quant loaded');
      let simCnt = 8;
      const zeros = Float32Array.from(Array(simCnt*12*BSIZE*BSIZE).fill(0.0));
      const tensorA = new ort.Tensor('float32', zeros, [simCnt, 12, BSIZE, BSIZE]);
      const feeds = { 
        's.1': tensorA
      };
      status = 2;
      const results = await nncache.run(feeds);
      status = 3;
    } catch (e) {
      errStr = e.toString();
      throw e;
    }
  }
  window.jspredictLoaded = true;
  // postMessage({isReady: true});
}());

export async function jspredict(inputs) {
  // console.log("jspredict");
  const len = inputs.length / (12 * BSIZE * BSIZE);
  //console.log(inputs);
  let simCnt = 8;
  let tmax = simCnt * 12 * BSIZE * BSIZE;
  let tinputs = [...inputs,...inputs,...inputs,...inputs,...inputs,...inputs,...inputs,...inputs, ...Array(tmax).fill(0.0)]; // fill zero for tail data
  tinputs = tinputs.slice(0, tmax);
  if (!nncache) {
    console.error('model must be loaded first');
    return [];
  }
  const nnet = nncache;
  const tensorA = new ort.Tensor('float32', tinputs, [simCnt, 12, BSIZE, BSIZE]);
  const feeds = { 
    's.1': tensorA
  };
  let result = await nnet.run(feeds);
  // console.log(result[279].data);
  const pi = [...result[130].data].slice(0, (BSIZE * BSIZE + 1) * len);
  const v = [...result[131].data].slice(0, len);
  let resp = [];
  const piSize = BSIZE * BSIZE + 1;
  for(let i=0; i<len; i++) {
    const eachPi = pi.slice(i*piSize, (i+1)*piSize);
    const eachV = v[i];
    resp = [...resp, ...eachPi, eachV];
  }
  return resp;
};

export class JSBoard {
  constructor(stones, turn, pass_cnt) {
    this.stones = stones;
    this.turn = turn;
    this.pass_cnt = pass_cnt;
  }
};

