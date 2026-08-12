export class NovaFractalEngine {
  vocabSize = 256; 
  embedSize = 64;
  contextSize = 32; 
  expertHiddenSize = 256;
  numExperts = 64; // Scaled up
  
  E: Float32Array;
  
  // Routing
  W_gate: Float32Array;
  
  // Experts
  W1_experts: Float32Array[];
  W2_experts: Float32Array[];
  b1_experts: Float32Array[];
  b2: Float32Array;
  
  constructor(experts: number = 64, hidden: number = 256) {
    this.numExperts = experts;
    this.expertHiddenSize = hidden;
    
    this.E = this.initTensor(this.vocabSize * this.embedSize);
    this.W_gate = this.initTensor(this.numExperts * (this.contextSize * this.embedSize));
    
    this.W1_experts = [];
    this.W2_experts = [];
    this.b1_experts = [];
    
    for(let i=0; i<this.numExperts; i++) {
       this.W1_experts.push(this.initTensor(this.expertHiddenSize * (this.contextSize * this.embedSize)));
       this.W2_experts.push(this.initTensor(this.vocabSize * this.expertHiddenSize));
       this.b1_experts.push(new Float32Array(this.expertHiddenSize));
    }
    this.b2 = new Float32Array(this.vocabSize);
  }
  
  initTensor(size: number) {
    let t = new Float32Array(size);
    let limit = Math.sqrt(6.0 / size);
    for(let i = 0; i < size; i++) t[i] = (Math.random() * 2 - 1) * limit;
    return t;
  }
  
  trainStep(text: string, lr: number) {
    let tokens = [];
    for(let i = 0; i < text.length; i++) {
      tokens.push(text.charCodeAt(i) % 256);
    }
    
    while (tokens.length <= this.contextSize) {
      tokens.unshift(32); 
    }
    
    let totalLoss = 0;
    let count = 0;
    
    for(let i = 0; i < tokens.length - this.contextSize; i++) {
       let x = tokens.slice(i, i + this.contextSize);
       let target = tokens[i + this.contextSize];
       
       let { probs, h, emb, expertIdx } = this.forward(x);
       let loss = -Math.log(probs[target] + 1e-10);
       totalLoss += loss;
       count++;
       
       this.backwardAndStep(x, target, probs, h, emb, expertIdx, lr);
    }
    
    return count > 0 ? totalLoss / count : 0;
  }
  
  forward(x: number[]) {
    // 1. Embedding
    let emb = new Float32Array(this.contextSize * this.embedSize);
    for(let i = 0; i < this.contextSize; i++) {
      let token = x[i];
      for(let j = 0; j < this.embedSize; j++) {
         emb[i * this.embedSize + j] = this.E[token * this.embedSize + j];
      }
    }

    // 2. Routing (Top-1 Expert)
    let gateScores = new Float32Array(this.numExperts);
    let maxGate = -Infinity;
    let expertIdx = 0;
    
    for(let i = 0; i < this.numExperts; i++) {
      let score = 0;
      for(let j = 0; j < emb.length; j++) {
        score += this.W_gate[i * emb.length + j] * emb[j];
      }
      gateScores[i] = score;
      if (score > maxGate) {
         maxGate = score;
         expertIdx = i;
      }
    }

    // 3. Expert Execution
    let h = new Float32Array(this.expertHiddenSize);
    let W1 = this.W1_experts[expertIdx];
    let b1 = this.b1_experts[expertIdx];
    
    for(let i = 0; i < this.expertHiddenSize; i++) {
      let sum = b1[i];
      for(let j = 0; j < emb.length; j++) {
        sum += W1[i * emb.length + j] * emb[j];
      }
      h[i] = sum > 0 ? sum : sum * 0.01; 
    }

    // 4. Output
    let logits = new Float32Array(this.vocabSize);
    let W2 = this.W2_experts[expertIdx];
    
    for(let i = 0; i < this.vocabSize; i++) {
      let sum = this.b2[i];
      for(let j = 0; j < this.expertHiddenSize; j++) {
        sum += W2[i * this.expertHiddenSize + j] * h[j];
      }
      logits[i] = sum;
    }

    let maxLogit = -Infinity;
    for(let i = 0; i < this.vocabSize; i++) {
      if(logits[i] > maxLogit) maxLogit = logits[i];
    }
    
    let sumExp = 0;
    let probs = new Float32Array(this.vocabSize);
    for(let i = 0; i < this.vocabSize; i++) {
      probs[i] = Math.exp(logits[i] - maxLogit);
      sumExp += probs[i];
    }
    for(let i = 0; i < this.vocabSize; i++) probs[i] /= sumExp;
    
    return { probs, h, emb, expertIdx };
  }
  
  backwardAndStep(x: number[], target: number, probs: Float32Array, h: Float32Array, emb: Float32Array, expertIdx: number, lr: number) {
    let dLogits = new Float32Array(this.vocabSize);
    for(let i = 0; i < this.vocabSize; i++) {
      dLogits[i] = probs[i];
    }
    dLogits[target] -= 1.0;

    let dh = new Float32Array(this.expertHiddenSize);
    let W2 = this.W2_experts[expertIdx];

    for(let i = 0; i < this.vocabSize; i++) {
      this.b2[i] -= lr * dLogits[i];
      for(let j = 0; j < this.expertHiddenSize; j++) {
        W2[i * this.expertHiddenSize + j] -= lr * dLogits[i] * h[j];
        dh[j] += dLogits[i] * W2[i * this.expertHiddenSize + j];
      }
    }

    for(let i = 0; i < this.expertHiddenSize; i++) {
      if (h[i] <= 0) dh[i] *= 0.01;
    }

    let dEmb = new Float32Array(emb.length);
    let W1 = this.W1_experts[expertIdx];
    let b1 = this.b1_experts[expertIdx];

    for(let i = 0; i < this.expertHiddenSize; i++) {
      b1[i] -= lr * dh[i];
      for(let j = 0; j < emb.length; j++) {
        W1[i * emb.length + j] -= lr * dh[i] * emb[j];
        dEmb[j] += dh[i] * W1[i * emb.length + j];
      }
    }

    for(let i = 0; i < this.contextSize; i++) {
      let token = x[i];
      for(let j = 0; j < this.embedSize; j++) {
        this.E[token * this.embedSize + j] -= lr * dEmb[i * this.embedSize + j];
      }
    }
  }
  
  generate(prompt: string, length: number, temperature: number = 0.7): string {
    let tokens = [];
    for(let i = 0; i < prompt.length; i++) {
      tokens.push(prompt.charCodeAt(i) % 256);
    }
    
    while(tokens.length < this.contextSize) {
       tokens.unshift(32);
    }
    
    let result = prompt;
    let currentTokens = tokens.slice(tokens.length - this.contextSize);
    
    for(let step = 0; step < length; step++) {
       let { probs } = this.forward(currentTokens);
       
       let nextToken = 32;
       
       if (temperature <= 0.01) {
          let maxProb = -1;
          for(let i = 0; i < this.vocabSize; i++) {
             if (probs[i] > maxProb) {
                maxProb = probs[i];
                nextToken = i;
             }
          }
       } else {
          let adjProbs = new Float32Array(this.vocabSize);
          let sumExp = 0;
          for(let i = 0; i < this.vocabSize; i++) {
             let p = Math.pow(probs[i], 1.0 / temperature);
             adjProbs[i] = p;
             sumExp += p;
          }
          
          let r = Math.random();
          let accum = 0;
          for(let i = 0; i < this.vocabSize; i++) {
             accum += adjProbs[i] / sumExp;
             if (r <= accum) {
                nextToken = i;
                break;
             }
          }
       }
       
       result += String.fromCharCode(nextToken);
       currentTokens.shift();
       currentTokens.push(nextToken);
    }
    
    return result;
  }
}
