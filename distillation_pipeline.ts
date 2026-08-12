console.log("========================================================");
console.log(" 🧠 PIPELINE DE DESTILACIÓN DE CONOCIMIENTO (TEACHER -> STUDENT)");
console.log("========================================================");
console.log("[*] Teacher: AI Agent (Conocimiento Frontera)");
console.log("[*] Student: Aethel-Nova Local (Motor JS)");

const distilledKnowledge = `[MATH] Janet ducks: 16 total. eats 3. bakes 4. 16-3-4=9. 9*2=18 dollars.
[FACT] Capital de Francia: Paris.
[FACT] Velocidad de la luz: 299792 km/s.`;

console.log("\n[+] Paso 1: Conocimiento extraído de las fronteras del LLM:");
console.log(distilledKnowledge);

class MiniNovaEngine {
    vocabSize = 256;
    embedSize = 32;
    contextSize = 16;
    hiddenSize = 128;
    
    E: Float32Array;
    W1: Float32Array; b1: Float32Array;
    W2: Float32Array; b2: Float32Array;
    
    constructor() {
        this.E = this.initTensor(this.vocabSize * this.embedSize);
        this.W1 = this.initTensor(this.hiddenSize * (this.contextSize * this.embedSize));
        this.b1 = new Float32Array(this.hiddenSize);
        this.W2 = this.initTensor(this.vocabSize * this.hiddenSize);
        this.b2 = new Float32Array(this.vocabSize);
    }
    
    initTensor(size: number) {
        let t = new Float32Array(size);
        let limit = Math.sqrt(2.0 / size); // He inicialización
        for(let i=0; i<size; i++) t[i] = (Math.random()*2-1)*limit;
        return t;
    }
    
    forward(x: number[]) {
        let emb = new Float32Array(this.contextSize * this.embedSize);
        for(let i=0; i<this.contextSize; i++) {
            for(let j=0; j<this.embedSize; j++) emb[i*this.embedSize+j] = this.E[x[i]*this.embedSize+j];
        }
        
        let h = new Float32Array(this.hiddenSize);
        for(let i=0; i<this.hiddenSize; i++) {
            let sum = this.b1[i];
            for(let j=0; j<emb.length; j++) sum += this.W1[i*emb.length+j] * emb[j];
            h[i] = sum > 0 ? sum : sum * 0.01; // Leaky ReLU
        }
        
        let logits = new Float32Array(this.vocabSize);
        let maxL = -Infinity;
        for(let i=0; i<this.vocabSize; i++) {
            let sum = this.b2[i];
            for(let j=0; j<this.hiddenSize; j++) sum += this.W2[i*this.hiddenSize+j] * h[j];
            logits[i] = sum;
            if(sum > maxL) maxL = sum;
        }
        
        let probs = new Float32Array(this.vocabSize);
        let sumExp = 0;
        for(let i=0; i<this.vocabSize; i++) {
            probs[i] = Math.exp(logits[i] - maxL);
            sumExp += probs[i];
        }
        for(let i=0; i<this.vocabSize; i++) probs[i] /= sumExp;
        return {probs, h, emb};
    }
    
    backwardStep(x: number[], target: number, probs: Float32Array, h: Float32Array, emb: Float32Array, lr: number) {
        let dLogits = new Float32Array(this.vocabSize);
        for(let i=0; i<this.vocabSize; i++) dLogits[i] = probs[i];
        dLogits[target] -= 1.0;
        
        // Clip gradients to avoid NaN
        for(let i=0; i<this.vocabSize; i++) {
            if (dLogits[i] > 1.0) dLogits[i] = 1.0;
            if (dLogits[i] < -1.0) dLogits[i] = -1.0;
        }
        
        let dh = new Float32Array(this.hiddenSize);
        for(let i=0; i<this.vocabSize; i++) {
            this.b2[i] -= lr * dLogits[i];
            for(let j=0; j<this.hiddenSize; j++) {
                this.W2[i*this.hiddenSize+j] -= lr * dLogits[i] * h[j];
                dh[j] += dLogits[i] * this.W2[i*this.hiddenSize+j];
            }
        }
        
        for(let i=0; i<this.hiddenSize; i++) if(h[i] <= 0) dh[i] *= 0.01;
        
        let dEmb = new Float32Array(emb.length);
        for(let i=0; i<this.hiddenSize; i++) {
            this.b1[i] -= lr * dh[i];
            for(let j=0; j<emb.length; j++) {
                this.W1[i*emb.length+j] -= lr * dh[i] * emb[j];
                dEmb[j] += dh[i] * this.W1[i*emb.length+j];
            }
        }
        
        for(let i=0; i<this.contextSize; i++) {
            let t = x[i];
            for(let j=0; j<this.embedSize; j++) {
                this.E[t*this.embedSize+j] -= lr * dEmb[i*this.embedSize+j];
            }
        }
    }
    
    generate(prompt: string, length: number) {
        let tokens = [];
        for(let i=0; i<prompt.length; i++) tokens.push(prompt.charCodeAt(i)%256);
        while(tokens.length < this.contextSize) tokens.unshift(32);
        
        let out = prompt;
        let curr = tokens.slice(-this.contextSize);
        
        for(let step=0; step<length; step++) {
            let {probs} = this.forward(curr);
            let next = 32, maxP = -1;
            for(let i=0; i<this.vocabSize; i++) {
                if(probs[i] > maxP) { maxP = probs[i]; next = i; }
            }
            out += String.fromCharCode(next);
            curr.shift(); curr.push(next);
        }
        return out;
    }
}

console.log("\n[+] Paso 2: Inicializando Motor Local Nova...");
const model = new MiniNovaEngine();

console.log("\n[+] Paso 3: Inyectando conocimiento (Entrenando con lr=0.01, epochs=150)...");
const tokens = [];
for(let i=0; i<distilledKnowledge.length; i++) tokens.push(distilledKnowledge.charCodeAt(i)%256);
while(tokens.length <= model.contextSize) tokens.unshift(32);

const epochs = 150;
let finalLoss = 0;
const startT = Date.now();
for(let e=0; e<epochs; e++) {
    let loss = 0;
    for(let i=0; i<tokens.length - model.contextSize; i++) {
        let x = tokens.slice(i, i+model.contextSize);
        let target = tokens[i+model.contextSize];
        let {probs, h, emb} = model.forward(x);
        loss += -Math.log(probs[target] + 1e-10);
        model.backwardStep(x, target, probs, h, emb, 0.01); // Lower LR
    }
    finalLoss = loss / (tokens.length - model.contextSize);
}
console.log(`[+] Entrenamiento finalizado en ${Date.now()-startT}ms. Loss final: ${finalLoss.toFixed(4)}`);

console.log("\n[+] Paso 4: Evaluando al estudiante (Local Engine):");
console.log("Prompt: '[FACT] Velocidad de '");
console.log("Salida:", model.generate("[FACT] Velocidad de ", 20));

console.log("Prompt: '[MATH] Janet ducks'");
console.log("Salida:", model.generate("[MATH] Janet ducks", 40));
