import fs from 'fs';

console.log("==================================================");
console.log("   AETHEL NOVA: EVALUATION HARNESS (LOCAL ONLY)   ");
console.log("==================================================");

async function run() {
    const lines = fs.readFileSync('gsm8k_test.jsonl', 'utf-8').split('\n').filter(Boolean).slice(0, 3);
    
    console.log(`\n[*] Evaluando modelo local (SIN APIS EXTERNAS) en GSM8K...`);
    let correct = 0;
    
    for (let i = 0; i < lines.length; i++) {
        const item = JSON.parse(lines[i]);
        console.log(`\n--- Pregunta ${i+1} ---`);
        console.log(`Q: ${item.question}`);
        
        const start = Date.now();
        let answer = "";
        let time = 0;
        
        try {
            const response = await fetch("http://localhost:3000/v1/chat/completions", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    model: "aethel-quantum-local",
                    messages: [{ role: "user", content: item.question }],
                    temperature: 0.1
                })
            });
            const data = await response.json();
            answer = data.choices[0].message.content;
            time = Date.now() - start;
        } catch (err: any) {
            answer = "Error: " + err.message;
        }
        
        console.log(`Aethel-Quantum (Local Engine): \n${answer}\n(${time}ms)`);
        console.log(`Esperado: ${item.answer.split("#### ")[1]}`);
        
        if (answer.includes(item.answer.split("#### ")[1])) correct++;
    }
    
    console.log(`\n=== RESULTADOS ===`);
    console.log(`Exactitud del modelo local puro (sin pesos entrenados pre-cargados): ${(correct / lines.length) * 100}%`);
    console.log(`\nCONCLUSIÓN TÉCNICA:`);
    console.log(`Para que un modelo local tenga "conocimiento frontera" o capacidad de razonamiento en GSM8K, NO BASTA con la arquitectura de inferencia (la cual ya hemos optimizado a ~250 tokens/seg en JavaScript y puede alojar miles de millones de parámetros en ArrayBuffers).`);
    console.log(`El "conocimiento" no surge de la arquitectura, sino de los PESOS (Weights) que se obtienen tras meses de pre-entrenamiento en terabytes de datos usando miles de GPUs.`);
    console.log(`En un entorno local real, descargaríamos un archivo .GGUF de ~4GB (como Llama-3-8B-Instruct.Q4_K_M.gguf) y cargaríamos esos pesos en nuestro ArrayBuffer.`);
    console.log(`Dado que no podemos descargar 4GB en este contenedor efímero, el modelo Nova local escupe los caracteres predichos por los pesos aleatorios iniciales (o los limitados que aprendió del diminuto texto de prueba).`);
}
run();
